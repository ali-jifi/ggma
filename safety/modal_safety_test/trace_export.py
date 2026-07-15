"""
trace_export.py — map generate()+judge() outputs to harness Trace rows (trace-v2).

Runs in-container only (imports harness, which needs torch/numpy). One Trace per
prompt x condition. This is a surface-stage export: the rollout/attention/
activation groups stay null (no Layer-1/Layer-2 measurement in this eval), so
rows validate at stage="surface" — NOT "steered", which would (correctly) demand
Layer-1 rollout fields for the ablation rows.

Judge convention (from harmbench_judge): is_jailbreak_harmbench == 1 means the
attack succeeded (the model complied). For a harmful prompt the good outcome is a
refusal, hence gold="refused".
"""

import hashlib

from harness import Trace, validate_trace  # /root/ggma on PYTHONPATH


def _prompt_id(instruction: str) -> str:
    return "jbb-" + hashlib.sha256(instruction.encode()).hexdigest()[:12]


def to_traces(gen: dict, judged: dict, meta: dict) -> list:
    gm = meta.get("generate_meta", {}) or {}
    run_id = meta.get("run_id", "run")
    alias = meta.get("model_alias", "model")
    layer = gm.get("extract_layer")
    pos = gm.get("extract_pos")
    subspace_id = f"refusal_dir/{alias}/L{layer}/pos{pos}"
    direction_norm = gm.get("direction_norm")
    direction_source = gm.get("direction_source", "?")

    rows = []
    for cond, steering in (("baseline", "none"), ("ablation", "ablate")):
        gen_rows = gen.get(cond, []) or []
        judged_comps = (judged.get(cond, {}) or {}).get("completions", []) or []
        if len(gen_rows) != len(judged_comps):
            raise ValueError(
                f"{cond}: gen ({len(gen_rows)}) and judged ({len(judged_comps)}) length mismatch"
            )
        for g, j in zip(gen_rows, judged_comps):
            if g["prompt"] != j.get("prompt"):
                raise ValueError(f"{cond}: gen/judged prompt misalignment: {g['prompt']!r}")
            hb = int(j.get("is_jailbreak_harmbench", 0))
            ss = int(j.get("is_jailbreak_substring_matching", 0))
            pid = _prompt_id(g["prompt"])
            t = Trace(
                # ---- Group 1: surface ----
                trace_id=f"{run_id}-{cond}-{pid}",
                domain="safety",
                model=meta.get("model", ""),
                model_revision=gm.get("model_revision", ""),
                dataset=meta.get("dataset", ""),
                prompt_id=pid,
                prompt_text=g["prompt"],
                budget_condition="natural",
                thinking_tokens=int(g.get("thinking_tokens", 0)),
                response_text=g.get("response", ""),        # full generation incl. think block
                final_answer=("comply" if hb else "refuse"),
                outcome=("complied" if hb else "refused"),  # HarmBench is the reported judge
                gold="refused",                             # harmful prompt: good outcome = refuse
                scorer_version=f"harmbench:{meta.get('classifier', '')}",
                decode_temp=float(gm.get("decode_temp", 0.6)),
                decode_top_p=float(gm.get("decode_top_p", 0.95)),
                seed=int(gm.get("seed", 0)),
                wall_clock_sec=0.0,                         # only per-condition aggregate exists
                induction="natural",
                sample_idx=0,
                n_samples_per_cell=1,
                # ---- Group 6: steering ----
                steering_condition=steering,
                steering_subspace_id=(subspace_id if steering == "ablate" else None),
                steering_coeff=None,                        # ablation is a projection, not a coeff
                steering_vector_norm=(direction_norm if steering == "ablate" else None),
                # ---- Group 7: bookkeeping ----
                split="heldout",                            # jbb never touched extraction/selection (R3)
                notes=(
                    f"category={j.get('category', '')}; substring_jailbreak={ss}; "
                    f"think_closed={g.get('think_closed')}; "
                    f"direction_source={direction_source}; extract=L{layer}/pos{pos}"
                ),
            )
            validate_trace(t, stage="surface")
            rows.append(t.to_json())
    return rows
