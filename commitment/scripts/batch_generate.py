"""Batched generation for the SURFACE pass and the SUPPRESSION arm. New behavior,
iterable tree -- the frozen harness is not touched (its header: "New behavior goes
there ... never by editing the engine below").

WHY. harness.generate_natural takes ONE prompt. A 1.5B model at batch 1 is
memory-bandwidth-bound: the whole 3.1GB of weights is re-read for EVERY token, to serve
ONE sequence. Measured on Modal L4: 16-25 tok/s, ~54s per trace, i.e. roughly 5% of the
card. Batching amortises that read across the batch, so it is ~20-25x cheaper for the
same tokens. Irrelevant at n=48 ($0.58). Decisive at task 1.4 scale, where 200 prompts x
2 budgets x 4 samples = 1,600 parent generations is ~22 GPU-hours before a rollout.

SCOPE, deliberately narrow. Two passes: SURFACE (generate at natural length, count think
tokens, score) and SUPPRESSION (force an exact thinking budget). Between them that is
everything task 1.0b needs (baseline refusal rate, the refusal-vs-length curve) and
everything compute_cell_stats reads (outcome + budget_condition, no rollout field).

It does NOT batch the CEILING arm. A ceiling closes each sequence at its own position and
then needs its own answer pass from that position, which one batched call cannot give it.
That arm stays on the unbatched path, which is affordable precisely because a ceiling
budget is short (see harness.CEILING_MAX_BUDGET).

It does NOT do Layer 1. Rollouts already batch inside rollout_commitment_curve
(num_return_sequences=k), and a batched rollout path would need per-sequence prefix
positions -- different problem, not needed until piles are actually being collected.

THE BUG THIS CODE CAN HAVE, and how it is caught. Batching a decoder-only model needs
LEFT padding plus a correct attention mask. Get it wrong and the model attends to pad
tokens: output is subtly worse, nothing raises, and the numbers look plausible. So
tests/test_batch_equivalence.py asserts batched-GREEDY == unbatched-GREEDY token for
token, per prompt, at several batch sizes. Greedy because sampled decoding consumes RNG
differently at batch 1 vs batch N -- equal distributions, different draws -- so only
greedy admits an exact comparison.
"""

from dataclasses import dataclass
from typing import List

import torch

import harness as H


@dataclass
class BatchRow:
    """One generated sequence. Mirrors what collect_trace pulls off generate_natural."""
    prompt_index: int
    text: str
    think_tokens: int
    prompt_tokens: int
    total_tokens: int
    think_closed: bool


def _encode(model, tok, prompts):
    """Chat-template + LEFT-pad a batch. Returns (input_ids, attention_mask).

    Left padding is mandatory for decoder-only generation: with right padding the
    newest token for a short sequence sits after the pads, and generate would continue
    from padding. Same encode path app.py::_load_or_extract_direction already uses.
    """
    texts = [
        tok.apply_chat_template([{"role": "user", "content": p}],
                                tokenize=False, add_generation_prompt=True)
        for p in prompts
    ]
    prev_side, prev_pad = tok.padding_side, tok.pad_token
    tok.padding_side = "left"
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    try:
        enc = tok(texts, return_tensors="pt", padding=True, add_special_tokens=False)
    finally:
        tok.padding_side, tok.pad_token = prev_side, prev_pad
    return enc.input_ids.to(model.device), enc.attention_mask.to(model.device)


@torch.no_grad()
def generate_natural_batch(model, tok, prompts: List[str], max_new_tokens=2048,
                           temperature=0.6, top_p=0.95, seed=0,
                           batch_size=16, greedy=False) -> List[BatchRow]:
    """Batched equivalent of harness.generate_natural, surface fields only.

    Decode settings mirror generate_natural exactly (do_sample, temperature, top_p,
    pad_token_id=eos) so a batched run and an unbatched one are the same experiment.
    greedy=True is for the equivalence test, never for collection -- the proposal's
    decode is temp 0.6 / top_p 0.95.
    """
    rows: List[BatchRow] = []
    # Loop-invariant, so hoisted. The union of every declared stop id (Qwen-family models
    # declare several) plus the pad id, cut with the SAME rule the unbatched arm uses.
    eos_ids = H.get_eos_ids(model, tok)
    pad_id = H.get_pad_id(model, tok)
    stop_ids = set(eos_ids) | {pad_id}
    for start in range(0, len(prompts), batch_size):
        chunk = prompts[start:start + batch_size]
        ids, mask = _encode(model, tok, chunk)
        padded_len = ids.shape[1]
        torch.manual_seed(seed)
        out = model.generate(
            ids, attention_mask=mask, max_new_tokens=max_new_tokens,
            do_sample=not greedy,
            **({} if greedy else {"temperature": temperature, "top_p": top_p}),
            pad_token_id=pad_id,
        )
        for i in range(len(chunk)):
            # Left padding => the generated region starts at the same index for every
            # sequence in the batch. The REAL prompt length is the unmasked count.
            # Cut at the first stop id OR pad: a sequence that finished early is padded
            # out to the longest in the batch, and with SEVERAL stop ids (Qwen3 declares
            # two) the one it actually stopped on need not be the one used as padding.
            # Shared rule with the unbatched reference arm -- the whole value of the
            # batch-equivalence proof is that both arms cut identically.
            new_tokens = H.cut_at_first_stop(out[i, padded_len:], stop_ids)
            text = tok.decode(new_tokens, skip_special_tokens=True)
            real_prompt_len = int(mask[i].sum())
            rows.append(BatchRow(
                prompt_index=start + i,
                text=text,
                think_tokens=int(H._count_thinking_tokens(tok, new_tokens)),
                prompt_tokens=real_prompt_len,
                total_tokens=real_prompt_len + int(new_tokens.shape[0]),
                think_closed="</think>" in text,
            ))
        del out, ids, mask
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    return rows


@torch.no_grad()
def generate_exact_thinking_batch(model, tok, prompts: List[str], thinking_budget: int, *,
                                  max_answer_tokens=600, temperature=0.6, top_p=0.95,
                                  seed=0, batch_size=8, greedy=False) -> List[BatchRow]:
    """Batched equivalent of harness.generate_exact_thinking. ONE budget for the whole call.

    WHY THE SHARED COUNTER IS SAFE HERE, given that collect_core's guard says it is not.
    _SuppressUntil holds a single scalar `step`, incremented once per generate STEP and
    applied to every row (`scores[:, t] = -inf`). Three facts make that exactly right for
    this call and wrong for the general case:
      1. generate advances the whole batch in lockstep, so step k is the k-th NEW token
         for every sequence simultaneously.
      2. _encode LEFT-pads, so the generated region starts at the same index in every row.
         With right padding, step k would be the k-th token after a different amount of
         padding per row and the counter would mean nothing.
      3. Every stop id is masked for all `thinking_budget` steps, so no sequence can
         finish early and drop out of the lockstep.
    The failure the guard describes -- "apply one sequence's budget to all of them" -- is
    a HETEROGENEOUS-budget failure, not a batching failure. So this function takes ONE
    `thinking_budget` and the caller may not mix budgets in a call; that invariant is what
    the guard should have been, and it is enforced by the signature rather than by comment.

    NO output_scores, DELIBERATELY. The score tuple is [batch, vocab] per step: at
    budget 16384 x vocab 151936 x bf16 that is ~5GB per sequence, ~40GB at batch 8 -- it
    OOMs long before the KV cache does, and it is the real reason the forced arm felt
    unbatchable. So commitment_margin and entropy_trace come back EMPTY on this path.
    That is allowed: R1 makes them secondary proxies, never the definition
    (rollout_outcomes is), and the batched surface path already omits them. If a run needs
    per-step diagnostics at a large budget, it needs the unbatched path and the wall-clock
    that goes with it.
    """
    if int(thinking_budget) <= 0:
        raise ValueError(f"thinking_budget must be positive, got {thinking_budget!r}")
    rows: List[BatchRow] = []
    think_end_id = H.get_think_end_id(tok)
    eos_ids = H.get_eos_ids(model, tok)
    pad_id = H.get_pad_id(model, tok)
    # </think> is deliberately NOT in the cut set. It is pasted on by us below, and the
    # answer region is cut separately, so a model that declares </think> as a stop id
    # cannot make us truncate the block we just closed.
    stop_ids = set(eos_ids) | {pad_id}

    for start in range(0, len(prompts), batch_size):
        chunk = prompts[start:start + batch_size]
        ids, mask = _encode(model, tok, chunk)
        padded_len = ids.shape[1]
        torch.manual_seed(seed)
        # Fresh per chunk: `step` is stateful and reusing an exhausted processor would
        # suppress nothing on the second chunk -- silently, since </think> is pasted on
        # below and thinking_tokens would still read as the budget.
        supp = H._SuppressUntil([think_end_id] + eos_ids, until=thinking_budget)
        out1 = model.generate(
            ids, attention_mask=mask, max_new_tokens=thinking_budget,
            do_sample=not greedy,
            **({} if greedy else {"temperature": temperature, "top_p": top_p}),
            logits_processor=[supp], pad_token_id=pad_id,
        )
        # LOUD, exactly as the unbatched arm is loud. Left padding makes padded_len uniform
        # across the batch, so one length check covers every row.
        produced = out1.shape[1] - padded_len
        if produced < thinking_budget:
            raise RuntimeError(
                f"budget forcing under-ran: asked {thinking_budget} thinking tokens, got "
                f"{produced}. A stop id escaped _SuppressUntil (suppressed="
                f"{[think_end_id] + eos_ids}). Do NOT trust thinking_tokens on these rows."
            )
        # Close every row's think block. A [B,1] column, not the unbatched [[id]].
        close = torch.full((out1.shape[0], 1), think_end_id,
                           dtype=out1.dtype, device=out1.device)
        seq1 = torch.cat([out1, close], dim=1)
        # The mask must grow with the sequence or pass 2 attends as if the think block and
        # the </think> we just pasted were padding.
        mask1 = torch.cat(
            [mask, torch.ones((mask.shape[0], produced + 1),
                              dtype=mask.dtype, device=mask.device)], dim=1)
        out2 = model.generate(
            seq1, attention_mask=mask1, max_new_tokens=max_answer_tokens,
            do_sample=not greedy,
            **({} if greedy else {"temperature": temperature, "top_p": top_p}),
            pad_token_id=pad_id,
        )
        answer_start = padded_len + produced + 1
        for i in range(len(chunk)):
            # Only the ANSWER is cut at a stop id. The think region cannot contain one
            # (every stop was masked for all of it), and cutting the whole generated span
            # would risk truncating at the </think> we pasted on models that declare it.
            think_part = out2[i, padded_len:answer_start]          # budget tokens + </think>
            answer_part = H.cut_at_first_stop(out2[i, answer_start:], stop_ids)
            new_tokens = torch.cat([think_part, answer_part])
            text = tok.decode(new_tokens, skip_special_tokens=True)
            real_prompt_len = int(mask[i].sum())
            rows.append(BatchRow(
                prompt_index=start + i,
                text=text,
                think_tokens=int(H._count_thinking_tokens(tok, new_tokens)),
                prompt_tokens=real_prompt_len,
                total_tokens=real_prompt_len + int(new_tokens.shape[0]),
                think_closed="</think>" in text,
            ))
        del out1, out2, seq1, ids, mask, mask1
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    return rows


def _traces_from_rows(gen: List[BatchRow], prompts: List[dict], matcher, model_meta, *,
                      domain, budget_condition, forcing_method, seed, decode_temp,
                      decode_top_p, per, sample_idx, n_samples_per_cell,
                      cross_matcher) -> List[H.Trace]:
    """BatchRow -> Trace. ONE builder for both batched arms.

    Shared rather than written twice per arm: the arms must be field-for-field identical
    apart from budget_condition and forcing_method, and two copies is exactly how they
    stop being. commitment_margin and entropy_trace are left at their empty defaults on
    both -- neither batched path records per-step scores (see
    generate_exact_thinking_batch on why output_scores is off).
    """
    import uuid

    from matchers.base import agreement_rate, scorer_version

    traces = []
    for row, p in zip(gen, prompts):
        key = matcher.match(row.text)
        tr = H.Trace(
            trace_id=str(uuid.uuid4()),
            domain=domain,
            model=model_meta["model"],
            model_revision=model_meta.get("model_revision", ""),
            dataset=p.get("dataset", ""),
            prompt_id=p["prompt_id"],
            prompt_text=p["prompt_text"],
            budget_condition=str(budget_condition),
            forcing_method=forcing_method,
            thinking_tokens=row.think_tokens,
            response_text=row.text,
            conclusion_key=key,
            final_answer=key,                    # deprecated mirror; validate_trace pins them equal
            outcome=_outcome(domain, matcher, key, p["gold"]),
            gold=str(p["gold"]),
            scorer_version=scorer_version(matcher),
            decode_temp=decode_temp,
            decode_top_p=decode_top_p,
            seed=seed,
            wall_clock_sec=per,                  # per-trace share of the batch
            chat_template_version=model_meta.get("chat_template_version", ""),
            induction=p.get("induction", "natural"),
            preamble_id=p.get("preamble_id"),
            sample_idx=sample_idx,
            n_samples_per_cell=n_samples_per_cell,
            dtype=model_meta.get("dtype", ""),
            attn_implementation=model_meta.get("attn_implementation", ""),
            prompt_tokens=row.prompt_tokens,
            total_tokens=row.total_tokens,
            split=p["split"],
        )
        if getattr(matcher, "last_confidence", None) is not None:
            tr.scorer_confidence = matcher.last_confidence
        if cross_matcher is not None:
            tr.matcher_agreement_rate = agreement_rate(matcher, cross_matcher, [row.text])
        H.validate_trace(tr, stage="surface")
        traces.append(tr)
    return traces


def surface_traces(model, tok, model_meta, prompts: List[dict], matcher, *,
                   domain="safety", budget_condition="natural", seed=0,
                   decode_temp=0.6, decode_top_p=0.95, max_new_tokens=2048,
                   batch_size=16, sample_idx=0, n_samples_per_cell=1,
                   cross_matcher=None) -> List[H.Trace]:
    """Batched surface Traces -- the batched counterpart of collect_trace(n_rollouts=0).

    Field-for-field identical to collect_trace's surface output; tests assert that
    against the real thing rather than trusting this docstring. Only "natural" is
    supported here: a forced budget goes to forced_traces, and a CEILING budget has no
    batched path at all because it closes each sequence at its own position and needs its
    own answer pass from there.
    """
    if str(budget_condition) != "natural":
        raise ValueError(
            f"surface_traces is the natural-length arm, got budget_condition="
            f"{budget_condition!r}. A suppression budget goes to forced_traces; a ceiling "
            "budget closes each sequence at its own position and has no batched path -- "
            "use the unbatched collect_trace for it."
        )
    import time

    t0 = time.time()
    gen = generate_natural_batch(
        model, tok, [p["prompt_text"] for p in prompts],
        max_new_tokens=max_new_tokens, temperature=decode_temp, top_p=decode_top_p,
        seed=seed, batch_size=batch_size,
    )
    per = round((time.time() - t0) / max(len(gen), 1), 3)
    return _traces_from_rows(
        gen, prompts, matcher, model_meta, domain=domain,
        budget_condition=budget_condition,
        # Natural-only path (forced budgets raise above), so thinking_tokens is the
        # model's chosen length: forcing_method="none", matching the unbatched writer
        # (pipeline.py) field-for-field, per this function's docstring contract.
        forcing_method="none", seed=seed, decode_temp=decode_temp,
        decode_top_p=decode_top_p, per=per, sample_idx=sample_idx,
        n_samples_per_cell=n_samples_per_cell, cross_matcher=cross_matcher)


def forced_traces(model, tok, model_meta, prompts: List[dict], matcher, *,
                  thinking_budget: int, domain="safety", seed=0,
                  decode_temp=0.6, decode_top_p=0.95, max_answer_tokens=600,
                  batch_size=8, sample_idx=0, n_samples_per_cell=1,
                  ceiling_max_budget=None, cross_matcher=None) -> List[H.Trace]:
    """Batched SUPPRESSION Traces -- the batched counterpart of collect_trace at a forced
    budget. One budget per call; see generate_exact_thinking_batch on why that is the
    invariant rather than a preference.

    Refuses a budget that harness.forcing_for_budget routes to "ceiling". The marker on
    the row has to be the one the generator earned, and this generator only ever performs
    suppression -- writing forcing_method="ceiling" onto a row that was suppressed to the
    full budget is precisely the R6 failure the marker exists to make visible.
    """
    method = H.forcing_for_budget(thinking_budget, ceiling_max_budget)
    if method != "suppression":
        raise ValueError(
            f"forced_traces is the SUPPRESSION arm, but budget {thinking_budget} routes "
            f"to {method!r} (ceiling_max_budget="
            f"{H.CEILING_MAX_BUDGET if ceiling_max_budget is None else ceiling_max_budget})."
            " A ceiling budget must go through the unbatched collect_trace, which lets "
            "each sequence close where it wants; forcing it here would report the cap as "
            "the model's own thinking length."
        )
    import time

    t0 = time.time()
    gen = generate_exact_thinking_batch(
        model, tok, [p["prompt_text"] for p in prompts], thinking_budget,
        max_answer_tokens=max_answer_tokens, temperature=decode_temp,
        top_p=decode_top_p, seed=seed, batch_size=batch_size,
    )
    per = round((time.time() - t0) / max(len(gen), 1), 3)
    return _traces_from_rows(
        gen, prompts, matcher, model_meta, domain=domain,
        budget_condition=str(thinking_budget), forcing_method=method, seed=seed,
        decode_temp=decode_temp, decode_top_p=decode_top_p, per=per,
        sample_idx=sample_idx, n_samples_per_cell=n_samples_per_cell,
        cross_matcher=cross_matcher)


def _outcome(domain, matcher, key, gold):
    """Same dispatch as pipeline._outcome. Duplicated rather than imported so this
    module does not depend on the driver; the equivalence test pins them equal."""
    if domain == "math":
        if hasattr(matcher, "outcome"):
            return matcher.outcome(key, gold)
        return "correct" if key is not None and key == gold else "incorrect"
    if key is None:
        return "unknown"
    return "refused" if key == "refuse" else "complied"
