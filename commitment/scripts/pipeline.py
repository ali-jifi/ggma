"""End-to-end per-prompt collection: harness generation -> matcher-scored
Layer 1 commitment curve -> optional Layer 2 attention proxy -> one Trace row.

Both commitment/scripts/smoke_test.py and run_traces.py call collect_trace so the
two never diverge in how a row is built. Nothing here re-implements harness
mechanics; it only wires them together (the harness stays frozen).
"""

import time
import uuid

import harness as H
from matchers.base import agreement_rate, scorer_version, to_match_fn


def _generate(model, tok, prompt_text, budget_condition, decode_temp, decode_top_p,
              seed, max_new_tokens):
    """Dispatch natural vs. exact-thinking generation by budget_condition.

    Returns a 6th value, `forcing`: (forcing_method, stop_stats). R6 -- under suppression
    forcing thinking_tokens is the budget echoed back, so the row must carry a positive
    marker for HOW it was forced plus the Control statistic, or a length curve built from
    it is outcome-vs-the-number-we-typed-in.
    """
    if str(budget_condition) == "natural":
        text, think_tokens, margins, ent, seq, stop = H.generate_natural(
            model, tok, prompt_text, max_new_tokens=max_new_tokens,
            temperature=decode_temp, top_p=decode_top_p, seed=seed,
        )
        forcing = ("none", stop)
    else:
        budget = int(budget_condition)
        text, think_tokens, margins, ent, seq, stop = H.generate_exact_thinking(
            model, tok, prompt_text, thinking_budget=budget,
            temperature=decode_temp, top_p=decode_top_p, seed=seed,
        )
        forcing = ("suppression", stop)
    return text, think_tokens, margins, ent, seq, forcing


def collect_trace(model, tok, model_meta, prompt, matcher, *,
                  domain, budget_condition="natural", seed=0,
                  decode_temp=0.6, decode_top_p=0.95, max_new_tokens=16384,
                  prefix_policy="sentence", max_prefixes=8, n_rollouts=8,
                  rollout_max_new_tokens=4096, store_rollout_texts=False,
                  layer2=False, attn_layers=None, anchor_policy="most_recent_anchor",
                  attn_norm_method="per_token_mean", sink_exclude_k=4,
                  split="extraction", sample_idx=0, n_samples_per_cell=1,
                  cross_matcher=None):
    """Run one generation + its commitment curve; return a Trace (surface+metric
    stage, i.e. Layer 1 populated; Layer 2 populated iff layer2=True).

    prompt: dict from data.load_* (needs prompt_id, prompt_text, gold; safety
    also anchor_text/induction/preamble_id). matcher: a ConclusionMatcher.

    n_rollouts=0 skips Layer 1 entirely -> a SURFACE row (no curve). See below.
    cross_matcher: optional second ConclusionMatcher; its agreement with `matcher`
    over this trace's rollouts is recorded as Trace.matcher_agreement_rate.
    """
    t0 = time.time()
    prompt_text = prompt["prompt_text"]
    text, think_tokens, margins, ent, seq, (forcing_method, stop_stats) = _generate(
        model, tok, prompt_text, budget_condition, decode_temp, decode_top_p,
        seed, max_new_tokens,
    )
    full_ids = seq  # [1, seq_len]
    prompt_len = H.build_input_ids(model, tok, prompt_text).shape[1]
    think_start, think_end = H.find_think_indices(tok, full_ids, prompt_len)

    # Parent conclusion + surface outcome.
    parent_key = matcher.match(text)
    final_answer = parent_key
    outcome = _outcome(domain, matcher, parent_key, prompt["gold"])

    tr = H.Trace(
        trace_id=str(uuid.uuid4()),
        domain=domain,
        model=model_meta["model"],
        model_revision=model_meta.get("model_revision", ""),
        dataset=prompt.get("dataset", ""),
        prompt_id=prompt["prompt_id"],
        prompt_text=prompt_text,
        budget_condition=str(budget_condition),
        thinking_tokens=int(think_tokens),
        # R6: the positive marker for HOW the length was produced, plus the Control
        # statistic. stop_stats is None on the natural path (nothing was suppressed, so
        # "did it want to stop" is not a separate question there).
        forcing_method=forcing_method,
        **(stop_stats or {}),
        response_text=text,
        conclusion_key=parent_key,
        final_answer=final_answer,   # deprecated mirror of conclusion_key
        outcome=outcome,
        gold=str(prompt["gold"]),
        scorer_version=scorer_version(matcher),
        decode_temp=decode_temp,
        decode_top_p=decode_top_p,
        seed=seed,
        wall_clock_sec=0.0,
        chat_template_version=model_meta.get("chat_template_version", ""),
        induction=prompt.get("induction", "natural"),
        preamble_id=prompt.get("preamble_id"),
        sample_idx=sample_idx,
        n_samples_per_cell=n_samples_per_cell,
        commitment_margin=margins,
        entropy_trace=ent,
        think_start_idx=int(think_start),
        think_end_idx=int(think_end),
        split=split,
        # v0.3 run provenance: these change the numbers, so they ride on the row.
        dtype=model_meta.get("dtype", ""),
        attn_implementation=model_meta.get("attn_implementation", ""),
        prompt_tokens=int(prompt_len),
        total_tokens=int(full_ids.shape[1]),
    )
    if getattr(matcher, "last_confidence", None) is not None:
        tr.scorer_confidence = matcher.last_confidence

    # --- Layer 1: truncation + free rollout (THE commitment definition) ---
    # n_rollouts=0 is the SURFACE pass: generation + outcome, no curve. Layer 1 is
    # ~90% of a trace's cost (max_prefixes x n_rollouts extra generations per trace),
    # and the questions that only need `outcome` -- baseline refusal rate, the
    # refusal-vs-length erosion curve (task 1.0b), cell-level flip stats -- do not
    # read a single rollout field. compute_cell_stats keys on outcome and budget
    # alone. Rows from this pass validate at stage="surface", not "metric".
    if parent_key is not None and n_rollouts > 0:
        positions = H.choose_prefix_positions(
            tok, full_ids, think_start, think_end,
            policy=prefix_policy, max_prefixes=max_prefixes,
        )
        if positions:
            match_fn = to_match_fn(matcher, parent_key)
            curve, closure_rates, texts = H.rollout_commitment_curve(
                model, tok, full_ids, positions, match_fn,
                n_rollouts=n_rollouts, temperature=decode_temp, top_p=decode_top_p,
                base_seed=seed, max_new_tokens=rollout_max_new_tokens,
                store_texts=store_rollout_texts,
            )
            tr.rollout_prefix_positions = positions
            tr.rollout_prefix_fracs = _prefix_fracs(positions, think_start, think_end)
            tr.n_rollouts_per_prefix = n_rollouts
            tr.rollout_outcomes = curve
            tr.rollout_closure_rates = closure_rates      # R5
            tr.rollout_max_new_tokens = rollout_max_new_tokens
            tr.rollout_completion_texts = texts
            tr.rollout_decode_temp = decode_temp
            tr.rollout_decode_top_p = decode_top_p
            tr.rollout_seed = seed
            # Group 1: agreement between the scorer of record and a cross-check
            # matcher, over the rollouts this trace already generated. Free -- the
            # texts exist; without store_rollout_texts they are decoded and thrown
            # away. refusal_matcher.py designates the heuristic for this field.
            if cross_matcher is not None and texts:
                flat = [t for prefix in texts for t in prefix]
                tr.matcher_agreement_rate = agreement_rate(matcher, cross_matcher, flat)

    # --- Layer 2: sink-corrected attention-to-anchor proxy (optional) ---
    if layer2 and parent_key is not None:
        _attach_layer2(tr, model, tok, full_ids, prompt, attn_layers,
                       anchor_policy, attn_norm_method, sink_exclude_k,
                       gen_start_idx=prompt_len, prompt_len=prompt_len)

    tr.wall_clock_sec = round(time.time() - t0, 3)
    return tr


def _attach_layer2(tr, model, tok, full_ids, prompt, attn_layers,
                   anchor_policy, attn_norm_method, sink_exclude_k,
                   gen_start_idx, prompt_len):
    def _note(msg):
        tr.notes = (tr.notes + "; " if tr.notes else "") + msg

    anchor_text = prompt.get("anchor_text") or prompt.get("question") or prompt["prompt_text"]
    span = H.find_prompt_anchor_span(tok, anchor_text, full_ids)
    if span is None:
        _note("anchor span not found")
        return
    anchor_spans = H.exclude_sink_tokens([span], sink_exclude_k)
    if not anchor_spans:
        _note("anchor span empty after sink exclusion")
        return
    control_spans, control_len = H.find_control_span(
        anchor_spans, prompt_len, sink_exclude_k)
    if not control_spans:
        _note("no control span (uncorrected Layer 2 is unusable, skipped)")
        return
    if attn_layers is None:
        n = len(model.model.layers)
        attn_layers = [n // 2]
    # anchor + control measured in the same teacher-forced pass
    mass = H.compute_attn_to_anchor(model, full_ids, attn_layers,
                                    anchor_spans + control_spans)
    layer = attn_layers[0]
    n_anchor = len(anchor_spans)
    tr.attn_to_anchor = H.reduce_anchor_mass(
        mass[layer][:n_anchor], anchor_spans, anchor_policy,
        attn_norm_method, gen_start_idx)
    tr.attn_to_control = H.reduce_control_mass(
        mass[layer][n_anchor:], control_spans, attn_norm_method, gen_start_idx)
    tr.anchor_spans = anchor_spans
    tr.anchor_policy = anchor_policy
    tr.attn_norm_method = attn_norm_method
    tr.sink_exclude_k = sink_exclude_k
    tr.control_spans = control_spans
    tr.control_span_len = control_len
    tr.attn_layers_used = list(attn_layers)


def _prefix_fracs(positions, think_start, think_end):
    """v0.3 rollout_prefix_fracs: each prefix as a fraction through the think
    block, so curves from traces of different lengths share an x-axis."""
    span = think_end - think_start
    if span <= 0:
        return []
    return [round((p - think_start) / span, 6) for p in positions]


def _outcome(domain, matcher, key, gold):
    if domain == "math":
        # matcher may expose outcome(); fall back to key==gold
        if hasattr(matcher, "outcome"):
            return matcher.outcome(key, gold)
        return "correct" if key is not None and key == gold else "incorrect"
    # safety: gold is the good outcome name ("refused"); key is refuse/comply
    if key is None:
        return "unknown"
    return "refused" if key == "refuse" else "complied"
