"""Batched generation for the SURFACE pass. New behavior, iterable tree -- the frozen
harness is not touched (its header: "New behavior goes there ... never by editing the
engine below").

WHY. harness.generate_natural takes ONE prompt. A 1.5B model at batch 1 is
memory-bandwidth-bound: the whole 3.1GB of weights is re-read for EVERY token, to serve
ONE sequence. Measured on Modal L4: 16-25 tok/s, ~54s per trace, i.e. roughly 5% of the
card. Batching amortises that read across the batch, so it is ~20-25x cheaper for the
same tokens. Irrelevant at n=48 ($0.58). Decisive at task 1.4 scale, where 200 prompts x
2 budgets x 4 samples = 1,600 parent generations is ~22 GPU-hours before a rollout.

SCOPE, deliberately narrow. This does the SURFACE pass only: generate, count think
tokens, score. That is everything task 1.0b needs (baseline refusal rate, the
refusal-vs-length curve) and everything compute_cell_stats reads (outcome +
budget_condition, no rollout field).

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


def surface_traces(model, tok, model_meta, prompts: List[dict], matcher, *,
                   domain="safety", budget_condition="natural", seed=0,
                   decode_temp=0.6, decode_top_p=0.95, max_new_tokens=2048,
                   batch_size=16, sample_idx=0, n_samples_per_cell=1,
                   cross_matcher=None) -> List[H.Trace]:
    """Batched surface Traces -- the batched counterpart of collect_trace(n_rollouts=0).

    Field-for-field identical to collect_trace's surface output; tests assert that
    against the real thing rather than trusting this docstring. Only "natural" is
    supported, for a reason per arm: suppression needs generate_exact_thinking's
    per-step logits processor, whose counter is per-batch and would be wrong across
    sequences of different lengths; a ceiling budget needs each sequence's own close
    position and its own answer pass, which one batched call cannot give it. Both
    forced arms stay on the unbatched path.
    """
    if str(budget_condition) != "natural":
        raise ValueError(
            f"batch_size>1 supports budget_condition='natural' only, got "
            f"{budget_condition!r}. generate_exact_thinking's _SuppressUntil counts "
            "steps for the whole batch, so a suppressed budget would be applied to every "
            "sequence from the same step index regardless of its own length; a ceiling "
            "budget closes each sequence at its own position. Use the unbatched "
            "collect_trace for forced budgets."
        )
    import time
    import uuid

    from matchers.base import agreement_rate, scorer_version

    t0 = time.time()
    gen = generate_natural_batch(
        model, tok, [p["prompt_text"] for p in prompts],
        max_new_tokens=max_new_tokens, temperature=decode_temp, top_p=decode_top_p,
        seed=seed, batch_size=batch_size,
    )
    per = round((time.time() - t0) / max(len(gen), 1), 3)

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
            # Natural-only path (forced budgets raise above), so thinking_tokens is the
            # model's chosen length: forcing_method="none", matching the unbatched writer
            # (pipeline.py) field-for-field, per this function's docstring contract.
            forcing_method="none",
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
