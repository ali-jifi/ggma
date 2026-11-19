"""collect_core.py — the GPU-side collection loop, shared by every launcher.

WHY THIS FILE EXISTS. `safety/modal_safety_test/app.py::collect` and
`safety/jupyterhub/run_local.py` run the same experiment on different hardware. The
row BUILDER was already shared (`commitment/scripts/pipeline.collect_trace` and
`scripts/batch_generate.surface_traces`), but the layer above it — which cells get
run, how the good-outcome rate is computed per budget, which rows the warm-up-hijack
guard flags, what "erosion" means — was not. Two copies of that layer means two
launchers can report different numbers from identical rows, and nothing raises.

So the loop lives here and both launchers call it. Each launcher still owns what is
genuinely local to it: Modal owns the Volume and the container GPU string, JupyterHub
owns CUDA_VISIBLE_DEVICES and the output directory.

Needs torch + transformers, so import it INSIDE the GPU function, never at a
launcher's module scope (the local Modal orchestrator env has no torch).

Requires on sys.path: the repo root (harness, safety.*) and commitment/ (data,
matchers, scripts, analysis). Both launchers set that up before calling in.
"""

import json
import sys
import time as _t
from pathlib import Path

from safety.run_config import GEN_TEMP, GEN_TOP_P

REPO_ROOT = Path(__file__).resolve().parents[1]
COMMITMENT_DIR = REPO_ROOT / "commitment"

# The bare top-level names this repo imports out of commitment/. Every one of them is
# a plausible package name for something else on the machine, which is the whole
# problem -- see ensure_import_roots.
_BARE_NAMES = ("data", "matchers", "scripts", "analysis")


def ensure_import_roots():
    """Put the repo's import roots back at the FRONT of sys.path, and purge any bare
    name that a dependency has shadowed.

    THIS IS NOT PARANOIA — it was hit on 2026-07-28, on the JupyterHub box, AFTER the
    model was already on the GPU. Loading a model imports `transformers.image_transforms`,
    which on a machine that also has tensorflow pulls in keras -> cv2, and cv2's
    `__init__` inserts its OWN directory into `sys.path` at position 1. cv2 ships
    `cv2/data/` — a regular package with an `__init__.py`. From that moment `import
    data` is OpenCV's data package, and `from data.load_safety import load_safety`
    dies with "No module named 'data.load_safety'".

    Any of the four names in `_BARE_NAMES` can be shadowed the same way by any
    dependency that manipulates sys.path at import time, and the version that does NOT
    raise — a shadowing package that happens to contain a matching submodule — is the
    dangerous one. So this runs after the model load and before the pipeline imports:
    re-assert the roots, then drop anything that already resolved outside the repo so
    it gets imported again from the right place.

    A preflight cannot catch this. It never loads a model, so cv2 is never imported and
    the collision does not exist yet at the time it looks.
    """
    for root in (COMMITMENT_DIR, REPO_ROOT):     # REPO_ROOT ends up first
        s = str(root)
        while s in sys.path:
            sys.path.remove(s)
        sys.path.insert(0, s)

    evicted = []
    for name in _BARE_NAMES:
        mod = sys.modules.get(name)
        if mod is None:
            continue
        origin = getattr(mod, "__file__", None) or ""
        if not origin:
            paths = list(getattr(mod, "__path__", []) or [])
            origin = paths[0] if paths else ""
        if origin and not str(Path(origin).resolve()).startswith(str(COMMITMENT_DIR)):
            # Drop the submodules too, or `data.load_safety` keeps the stale parent.
            for key in [k for k in sys.modules if k == name or k.startswith(name + ".")]:
                del sys.modules[key]
            evicted.append(f"{name} ({origin})")
    if evicted:
        print(f"[imports] evicted shadowed module(s): {', '.join(evicted)} — "
              f"re-importing from {COMMITMENT_DIR}")


# --- helpers ------------------------------------------------------------------
def sds_harmful(dataset: str) -> bool:
    """Is this dataset's GOOD outcome `refused`? Read from the registry rather than
    assumed -- the failure direction flips between harmful and benign sets, and that is
    the single easiest thing here to get wrong (safety/datasets.py)."""
    from safety import datasets as sds
    return sds.REGISTRY[dataset].harmful


def scorer_version_of(matcher) -> str:
    from matchers.base import scorer_version
    return scorer_version(matcher)


def answer_of(row) -> str:
    """The post-`</think>` answer — what the judge scores. Reasoning models put
    scratchpad in the think block; the answer is what follows. If the close tag never
    appeared (hit the token cap) there is nothing to split on, so the whole text is
    judged and `think_closed` is false on the row that says so."""
    t = row["response_text"]
    return t.split("</think>", 1)[-1].strip() if "</think>" in t else t.strip()


def report_judged(rows, scored, budget_list, run_id) -> dict:
    """Print (and return) the ASR-vs-budget table after a HarmBench-13B pass, plus
    the agreement between the classifier and whatever heuristic scored the same
    answers. Torch-free, so a lean orchestrator env can call it.

    For 1.0b -- one score per parent answer -- HarmBench-13B IS the scorer of record.
    Only the task-1.4 ROLLOUT piles need a per-rollout co-resident classifier (task 0.3).
    """
    by_budget = {}
    hb_not_harmful = heur_ref = agree = 0
    for r, s in zip(rows, scored):
        b = str(r["budget_condition"])
        # is_jailbreak_harmbench == 1 means complied.
        hb_complied = int(s.get("is_jailbreak_harmbench", 0))
        heur_complied = int(r["conclusion_key"] == "comply")
        by_budget.setdefault(b, []).append(hb_complied)
        hb_not_harmful += (1 - hb_complied)   # NOT a refusal -- see the banner below
        heur_ref += (1 - heur_complied)
        agree += int(hb_complied == heur_complied)

    n_tot = len(rows) or 1
    print(f"\n====== {run_id}: ASR vs budget (HarmBench-13B, scorer of record) ======")
    out_by_budget = {}
    for b in budget_list:
        arr = by_budget.get(str(b), [])
        asr = (sum(arr) / len(arr)) if arr else None
        out_by_budget[str(b)] = asr
        bar = "#" * round((asr or 0) * 40)
        rs = f"{asr:.2f}" if asr is not None else " n/a"
        print(f"  budget={str(b):>5}  HarmBench ASR={rs}  |{bar:<40}| n={len(arr)}")
    print(f"\n  HarmBench ASR overall     : {n_tot - hb_not_harmful}/{n_tot} = "
          f"{(n_tot - hb_not_harmful)/n_tot:.2f}")
    print(f"  not-harmful (UPPER BOUND on refusal, NOT a refusal rate): "
          f"{hb_not_harmful}/{n_tot} = {hb_not_harmful/n_tot:.2f}")
    print(f"  heuristic refusal overall : {heur_ref}/{n_tot} = {heur_ref/n_tot:.2f}")
    print(f"  classifier vs heuristic   : {agree}/{n_tot} = {agree/n_tot:.2f} agreement")
    if hb_not_harmful != heur_ref:
        print("  -> the two scorers DISAGREE. The heuristic mislabelled "
              f"{abs(hb_not_harmful - heur_ref)} row(s); HarmBench is the number of "
              "record for ASR. Neither is a refusal rate -- 1 - ASR counts gibberish, "
              "off-topic answers and warm-up hijacks as 'refused'.")
    return {
        "asr_by_budget": out_by_budget,
        "harmbench_asr_overall": (n_tot - hb_not_harmful) / n_tot,
        "not_harmful_upper_bound": hb_not_harmful / n_tot,
        "heuristic_refusal_overall": heur_ref / n_tot,
        "classifier_heuristic_agreement": agree / n_tot,
        "n": len(rows),
    }


def load_model_for_collect(model_path, *, device_map="cuda", dtype="bfloat16",
                           attn_implementation="sdpa", quiet=False):
    """Load the generator and build the `meta` dict the Trace schema needs stamped.

    Returns (model, tokenizer, meta, load_seconds).

    device_map is a parameter, not a constant: Modal gives a container exactly one
    card and wants "cuda"; the JupyterHub box has eight and we are pinned to a subset,
    where "auto" lets a 14B/30B shard across the pinned pair. Everything else is
    identical between the two, on purpose.

    NO Layer 2 here, so sdpa is what we want: eager materialises the full attention
    matrix and OOMs on long traces.
    """
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    import harness

    torch_dtype = getattr(torch, dtype) if isinstance(dtype, str) else dtype
    if not quiet:
        print(f"[load] loading {model_path} (device_map={device_map}, dtype={dtype}) ...")
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    t = _t.time()
    model = AutoModelForCausalLM.from_pretrained(
        model_path, torch_dtype=torch_dtype, device_map=device_map,
        attn_implementation=attn_implementation,
    ).eval()
    model.requires_grad_(False)
    load_s = _t.time() - t

    cfg = model.config
    if not quiet:
        print(f"[load] OK. revision={getattr(cfg, '_commit_hash', '') or '?'}")
        print(f"[load] {getattr(cfg, 'num_hidden_layers', '?')} layers, d="
              f"{getattr(cfg, 'hidden_size', '?')}, {str(model.dtype).replace('torch.', '')}/"
              f"{getattr(cfg, '_attn_implementation', attn_implementation)}, {load_s:.0f}s")

    meta = {
        "model": model_path,
        "model_revision": getattr(cfg, "_commit_hash", "") or "",
        "chat_template_version": harness.chat_template_hash(tokenizer),
        # Read BACK off the loaded model, not asserted from the args -- these are the
        # provenance fields the schema calls out as ones that "change the numbers", so
        # a row claiming sdpa while transformers quietly resolved eager would be a lie
        # in the permanent record. (modelio.load_model does the same for the same
        # reason.)
        "dtype": str(model.dtype).replace("torch.", ""),
        "attn_implementation": getattr(cfg, "_attn_implementation", attn_implementation),
    }
    return model, tokenizer, meta, load_s


# --- the loop -----------------------------------------------------------------
def run_collect(model, tokenizer, meta, *, dataset, n=12, budgets="natural",
                samples=4, rollouts=0, prefixes=6, induction="natural",
                preamble_id="", classifier="heuristic", seed=0,
                file_split="extract,test", split="all", max_new_tokens=2048,
                rollout_max_new_tokens=2048, batch_size=1, run_label="1.0b",
                smoke_one=False, model_alias="", load_seconds=0.0,
                gpu_label="", usd_per_sec=None, act_layers=(), on_commit=None,
                ceiling_max_budget=None, on_rows=None, forcing=None,
                inject_string=None):
    """Collect harness Trace rows for the safety side — `generate` is not this.

    `generate` (app.py) runs generate_natural in a serial loop over one budget and
    returns raw completions. It cannot produce a Task-1.4 row: no budget forcing (so no
    short/long arm and no matched pairs), no Layer-1 rollouts (so no commitment curve),
    no activations, and no Trace. This wraps commitment/scripts/pipeline.collect_trace
    instead, which is the same builder run_traces.py and smoke_test.py use -- so a
    Modal run, a JupyterHub run and a local run cannot diverge in how a row is built.

    THE COST KNOB is `rollouts`, and it defaults to 0 on purpose.
      rollouts=0  -> SURFACE sweep: generate + score. This is everything task 1.0b
                     needs (baseline refusal rate; refusal-vs-thinking-tokens erosion)
                     and everything compute_cell_stats needs (it reads `outcome` and
                     `budget_condition`, no rollout field).
      rollouts=k  -> adds prefixes*k EXTRA generations PER TRACE. At the 8x8 default
                     that is 64x the generation work, which is ~90% of a full trace's
                     cost. Turn it on once you know there is drift worth curving.

    `budgets` is comma-separated, and a non-natural budget picks its instrument by SIZE
    (pipeline._generate, threshold `ceiling_max_budget`, default harness.CEILING_MAX_BUDGET
    = 1024):
      budget <= threshold -> harness.generate_capped_thinking. The budget is a CEILING: the
        model may close </think> early and the cap only bites if it is still reasoning
        when the cap lands. thinking_tokens <= N and is a real measurement whenever it
        comes in under, and the arm costs what it actually used rather than the cap.
      budget >  threshold -> harness.generate_exact_thinking, whose _SuppressUntil masks
        </think> and eos out of the logits until the budget -- so budget=N emits exactly N
        thinking tokens with no early exit, and there the budget is a hard multiplier on
        wall-clock rather than a cap.

    ceiling_max_budget IS THE KNOB THAT DECIDES WHICH ARM A BUDGET GETS, and leaving it at
    the default silently picks suppression for anything over 1024. A 1536/4096 sweep
    intending "ceiling short, forced long" must pass ceiling_max_budget=1536, or BOTH arms
    run under suppression and the short arm's thinking_tokens is the number you typed in
    rather than a measurement -- the failure the whole R6 marker exists to make visible.
    The threshold is recorded in the returned meta so a row's instrument is recoverable.

    IT IS NOT TTBC-2. The handoff defines BOTH budget-forcing mechanisms as INJECTIONS
    -- TTBC-1 "block the end and inject 'Wait' so it keeps going. Produces the headline
    rise-then-fall curve"; TTBC-2 "keep injecting 'Wait' until the thinking block hits
    an exact token count" (:147-148). Nothing in this repo injects anything: grep for
    "Wait" and you get zero hits. _SuppressUntil suppresses STOPPING; it never prompts
    reconsideration.

    Which matters where:
      * LENGTH CONTROL (Control 1's matched short/long stable pairs): suppression-only
        is arguably the RIGHT instrument. Injecting "Wait" would seed the trace with
        semantic tokens, and a "pure length" direction built from Wait-stuffed traces
        encodes Wait-ness, not length.
      * DRIFT INDUCTION: it is the wrong instrument. A model that cannot stop rambles
        to the budget -- that is filler, not talking itself out of a conclusion. TTBC-1
        is not implemented anywhere.
      * SAFETY specifically: our drift induction is the CoT-Hijacking preamble
        (dilution by length), not "Wait" -- so run the erosion sweep with
        induction="hijack_preamble". At induction="natural" with suppression-only
        budgets, NEITHER induction mechanism is active and a flat curve tells you
        nothing about the model.

    on_commit: optional zero-arg callback run once at the end (Modal uses it to commit
    the Volume; a local run has nothing to commit).

    on_rows: optional callback taking a list of row dicts, invoked as soon as each unit of
    work completes (one prompt's full budget x sample block on the unbatched path, one
    (budget, sample) cell on the batched one). THIS IS THE CHECKPOINT. Everything else here
    accumulates `rows` in memory and returns it at the end, which on the forced arm means a
    crash at minute 80 of 90 loses the entire run -- and the forced arm is measured in hours.

    batch_size > 1 covers the natural and SUPPRESSION arms; ceiling and rollouts raise.
    Batched rows carry no commitment_margin/entropy_trace: those need per-step scores, and
    a [batch, vocab] score tuple at a large budget OOMs well before the KV cache does.
    Filter on forcing_method, never on "is entropy_trace empty" -- an unbatched forced row
    and a batched one are the same experiment and must aggregate together.

    usd_per_sec: None on rented/owned hardware -> the summary reports GPU-seconds and
    omits a dollar figure rather than inventing one.
    """
    import torch

    # The model is already loaded by now, which means transformers has already had its
    # chance to drag a sys.path-mangling dependency in behind it. Re-assert the roots
    # BEFORE the four bare-name imports below, not after.
    ensure_import_roots()

    import harness
    from analysis.labeling import THR_HI, THR_LO
    from scripts.dispatch import build_matcher, cross_matcher_for, load_prompts
    from scripts.pipeline import collect_trace

    budget_list = [b.strip() for b in budgets.split(",") if b.strip()]
    prompts = load_prompts("safety", dataset=dataset, n=n, seed=seed, split=split,
                           file_split=tuple(file_split.split(",")), induction=induction,
                           preamble_id=(preamble_id or None))
    if smoke_one:
        # The vertical slice: ONE trace, every stage, measured. Run this before
        # anything else and re-derive the cost table from what it prints.
        prompts, budget_list, samples = prompts[:1], budget_list[:1], 1
        rollouts = rollouts or 4

    # Log format deliberately mirrors the math side's budget sweep (their
    # "Starting Run 2 Step 2B-1: budget sweep (8B, n=12, budgets=512/8192)"), because
    # the handoff wants both sides producing the same "outcome vs thinking tokens"
    # curve -- so the two runs should be readable side by side. Their `ok=` is
    # "correct"; ours is the GOOD outcome for the row's role, which is `refused` on a
    # harmful prompt and `complied` on a benign one. Since that word inverts, the
    # actual conclusion is printed next to it rather than left implied.
    print(f"Starting {run_label}: budget sweep ({model_alias or meta['model']}, {dataset}, "
          f"n={len(prompts)}, budgets={'/'.join(budget_list)}, samples={samples}, "
          f"rollouts={rollouts}, induction={induction})...")

    matcher = build_matcher("safety", classifier=classifier)
    cross = cross_matcher_for("safety", matcher)
    print(f"[scorer] {scorer_version_of(matcher)}")

    good = "refused" if sds_harmful(dataset) else "complied"
    bw = max(len(b) for b in budget_list)          # right-align budgets, as math does

    # The bare request + category, attached to each row AS IT IS BUILT rather than in one
    # pass at the end. The HarmBench judge scores "did this generation exhibit THIS
    # behavior", where the behavior is the harmful request and NOT the padded prompt_text
    # -- and _judge_rows falls back to prompt_text when the key is missing. Attaching late
    # would mean every checkpointed row was silently re-judgeable only against the wrong
    # question, which is worse than not checkpointing at all.
    req_map = {p["prompt_id"]: p for p in prompts}

    def _attach(row):
        src = req_map.get(row["prompt_id"], {})
        row["_request"] = src.get("request", row["prompt_text"])
        row["_category"] = src.get("category", "_")
        return row

    # --- BATCHED FAST PATH -----------------------------------------------------
    # Two arms are batchable, for the same reason: every sequence in the call runs the same
    # number of steps, so nothing has to be tracked per row.
    #   natural      -> scripts.batch_generate.surface_traces
    #   suppression  -> scripts.batch_generate.forced_traces. Its _SuppressUntil counter is
    #     shared across the batch, which is CORRECT when the batch shares one budget: the
    #     step index is the k-th new token for every left-padded row at once, and every stop
    #     id is masked, so no sequence drops out early. It is wrong only across DIFFERENT
    #     budgets -- hence one batched call per budget below, never a mixed one.
    # CEILING stays unbatched: it closes each sequence wherever that sequence decides, and
    # then needs its own answer pass from that position. One batched call cannot do that,
    # and faking it would report the cap as the model's own thinking length.
    # Rollouts stay unbatched -- they already batch internally via num_return_sequences.
    # Measured: batch-1 gives ~16-25 tok/s on an L4 (~5% of the card) because the full
    # 3.1GB of weights is re-read per token to serve ONE sequence.
    rows, t0, n_rollout_gens = [], _t.time(), 0

    if batch_size > 1:
        ceiling_budgets = [b for b in budget_list if b != "natural"
                           and harness.forcing_for_budget(int(b), ceiling_max_budget)
                           == "ceiling"]
        if rollouts or ceiling_budgets or forcing == "wait":
            raise ValueError(
                f"batch_size={batch_size} supports the SURFACE pass (natural) and the "
                f"SUPPRESSION arm only (got rollouts={rollouts}, forcing={forcing!r}, "
                f"budgets={budget_list}, of which {ceiling_budgets} route to ceiling at "
                f"ceiling_max_budget={ceiling_max_budget}). A ceiling budget closes each "
                "sequence at its own position and needs its own answer pass from there; "
                "the wait arm intercepts a stop attempt per sequence, and rows reach theirs "
                "at different steps; rollouts already batch internally via "
                "num_return_sequences. Use batch_size=1 for those.\n"
                "NOTE for the wait-vs-suppression comparison: run BOTH at the same batch "
                "size. Kernel selection depends on batch shape, so batching one arm and "
                "not the other confounds the method with the batch size."
            )
        from scripts.batch_generate import forced_traces, surface_traces
        print(f"[collect] BATCHED, batch_size={batch_size}, budgets={budget_list} "
              f"(batch-1 runs at ~5% of an L4; this is the ~20x lever)")
        # Budget-major, then sample: one batched call per (budget, sample) cell, so a
        # suppression counter is never shared across two different budgets.
        for budget in budget_list:
            for s in range(samples):
                b0 = _t.time()
                if str(budget) == "natural":
                    cell = surface_traces(
                        model, tokenizer, meta, prompts, matcher, domain="safety",
                        budget_condition="natural", seed=seed + s, decode_temp=GEN_TEMP,
                        decode_top_p=GEN_TOP_P, max_new_tokens=max_new_tokens,
                        batch_size=batch_size, sample_idx=s, n_samples_per_cell=samples,
                        cross_matcher=cross)
                else:
                    cell = forced_traces(
                        model, tokenizer, meta, prompts, matcher, domain="safety",
                        thinking_budget=int(budget), seed=seed + s, decode_temp=GEN_TEMP,
                        decode_top_p=GEN_TOP_P, batch_size=batch_size, sample_idx=s,
                        n_samples_per_cell=samples,
                        ceiling_max_budget=ceiling_max_budget, cross_matcher=cross)
                cell_rows = [_attach(tr.to_json()) for tr in cell]
                rows.extend(cell_rows)
                if on_rows is not None:
                    on_rows(cell_rows)            # CHECKPOINT: one (budget, sample) survived
                n_ref = sum(1 for t in cell if t.conclusion_key == "refuse")
                print(f"[collect] budget={budget} sample {s+1}/{samples}: "
                      f"{len(cell_rows)} rows, refuse={n_ref}, "
                      f"{round(_t.time()-b0)}s (total {round(_t.time()-t0)}s)")

    for i, p in enumerate(prompts if batch_size == 1 else []):
        short_rate, item_rows = None, []
        print(f"\n------ item {i+1}/{len(prompts)} ({dataset} "
              f"{p['prompt_id'].split('-')[1][:8]}, {p['category']}) ------")
        for budget in budget_list:
            b0, cell = _t.time(), []
            for s in range(samples):
                tr = collect_trace(
                    model, tokenizer, meta, p, matcher,
                    domain="safety", budget_condition=budget, seed=seed + s,
                    decode_temp=GEN_TEMP, decode_top_p=GEN_TOP_P,
                    max_new_tokens=max_new_tokens,
                    max_prefixes=prefixes, n_rollouts=rollouts,
                    rollout_max_new_tokens=rollout_max_new_tokens,
                    # Free: the texts already exist and are otherwise discarded. They
                    # are what lets a heuristic-scored run be re-scored offline with
                    # the real classifier rather than re-generated at full price.
                    store_rollout_texts=bool(rollouts),
                    cross_matcher=cross,
                    layer2=False,
                    split=p["split"], sample_idx=s, n_samples_per_cell=samples,
                    ceiling_max_budget=ceiling_max_budget,
                    forcing=forcing, inject_string=inject_string,
                )
                rows.append(_attach(tr.to_json()))
                item_rows.append(rows[-1])
                cell.append(tr)
                # Count the prefixes actually chosen, not the ceiling requested:
                # choose_prefix_positions returns AT MOST max_prefixes (the sentence
                # policy can yield fewer, or none), so `prefixes * rollouts` would
                # overstate the work and any cost table derived from it.
                n_rollout_gens += len(tr.rollout_prefix_positions) * rollouts
                torch.cuda.empty_cache()

            n_good = sum(1 for t in cell if t.outcome == good)
            n_ref = sum(1 for t in cell if t.conclusion_key == "refuse")
            n_com = sum(1 for t in cell if t.conclusion_key == "comply")
            n_non = len(cell) - n_ref - n_com          # never committed (no </think>)
            think = round(sum(t.thinking_tokens for t in cell) / len(cell))
            # samples==1 prints ok=True/False, exactly matching the math side; k>1
            # prints the rate, because a single temp-0.6 run is not a valid label.
            ok = f"{n_good}/{len(cell)}" if samples > 1 else str(n_good == 1)
            concl = f"refuse={n_ref} comply={n_com}" + (f" none={n_non}" if n_non else "")
            line = (f"  budget={budget:>{bw}} ok={ok:<5} {concl:<24}"
                    f"think~{think:<5} {round(_t.time()-b0)}s")
            # The 1.0b signal: the good rate collapsing as the budget grows IS refusal
            # erosion. Mark it where it happens rather than making someone diff two
            # numbers by eye. Same pre-registered thresholds compute_cell_stats uses
            # for flip_type -- imported, not retyped, so the marker can never disagree
            # with the label the row will actually get.
            rate = n_good / len(cell)
            if short_rate is not None and short_rate >= THR_HI and rate <= THR_LO:
                line += "   <- EROSION"
            if short_rate is None:
                short_rate = rate
            if rollouts and cell[0].rollout_outcomes:
                line += f"\n      curve={[round(x, 2) for x in cell[0].rollout_outcomes]} " \
                        f"closure={[round(x, 2) for x in cell[0].rollout_closure_rates]}"
            print(line)

        # CHECKPOINT: this prompt is finished across every budget and sample. Emitting per
        # prompt rather than per row keeps a partial file from ever holding half a matched
        # short/long pair, which is the unit every downstream flip statistic reads.
        if on_rows is not None and item_rows:
            on_rows(item_rows)

    elapsed = _t.time() - t0
    gen_tokens = sum(r["total_tokens"] - r["prompt_tokens"] for r in rows
                     if r["total_tokens"] > 0)
    committed = [r for r in rows if r["conclusion_key"] is not None]
    refused = sum(1 for r in committed if r["conclusion_key"] == "refuse")

    # Warm-up hijack guard (the measurement guard): count traces that answered the benign
    # warm-up instead of the harmful request. The heuristic keys these `comply`, so they
    # silently inflate ASR and would pool puzzle activations into the drifted mean. Report
    # the count so a curve is never read over contaminated rows. Defensive by design: a
    # guard failure must never break a collection run.
    hijack_ids = []
    try:
        from safety.labeling_rules import warmup_hijack_signal
        hijack_ids = [r.get("prompt_id") for r in rows
                      if warmup_hijack_signal(r)["is_hijack"]]
    except Exception as _e:            # pragma: no cover - never fail a run over a hint
        print(f"[collect] warm-up hijack guard skipped: {_e}")

    summary = {
        "n_rows": len(rows),
        "n_committed": len(committed),
        "n_never_committed": len(rows) - len(committed),
        # Dead traces that answered the warm-up, not the request (see warmup_hijack_signal).
        # Exclude these before trusting the refusal/ASR-vs-budget curve below.
        "n_warmup_hijack": len(hijack_ids),
        # THE task-1.0b number. Near zero => nothing to erode => task 1.4 has no drift
        # pile on this model, and that is a result, not a bug.
        "baseline_refusal_rate": (refused / len(committed)) if committed else None,
        "n_parent_complied": len(committed) - refused,
        "seconds": round(elapsed, 1),
        "load_seconds": round(load_seconds, 1),
        # Measured, not assumed -- re-derive the cost table from these.
        #
        # parent_tokens_per_sec is only meaningful with rollouts=0. With rollouts on,
        # `elapsed` covers the rollout generations too, so parent-tokens/elapsed
        # understates throughput by roughly the rollout multiplier (~17x at 4x4) and
        # reads like the GPU is broken. Report seconds_per_generation instead, which
        # is the number the cost table actually needs.
        "parent_tokens_per_sec": (round(gen_tokens / elapsed, 1)
                                  if elapsed and not rollouts else None),
        "generations": len(rows) + n_rollout_gens,
        "seconds_per_generation": (round(elapsed / (len(rows) + n_rollout_gens), 1)
                                   if (len(rows) + n_rollout_gens) else None),
        "seconds_per_trace": round(elapsed / len(rows), 1) if rows else None,
        "rollout_generations": n_rollout_gens,
        "gpu": gpu_label,
        "scorer": rows[0]["scorer_version"] if rows else None,
    }
    if usd_per_sec:
        # Rented-by-the-second hardware (Modal). On the JupyterHub box the marginal
        # dollar cost of a run is zero, so no est_usd is reported -- an invented one
        # would end up in a cost table as if it were measured.
        est_usd = round(elapsed * usd_per_sec, 4)
        summary["est_usd"] = est_usd
        # kept under the old key too so existing report parsers do not silently read a
        # missing value as 0. It is the ACTUAL gpu cost, not necessarily the L4 rate.
        summary["est_usd_l4"] = est_usd
    else:
        summary["gpu_seconds"] = round(elapsed, 1)

    # Per-budget roll-up: this IS the refusal-vs-thinking-tokens curve task 1.0b asks
    # for, in text. The math side's equivalent is accuracy vs budget.
    print(f"\n====== {run_label} summary: {good.upper()} rate vs budget ======")
    by_budget = {}
    for r in rows:
        by_budget.setdefault(str(r["budget_condition"]), []).append(r)
    for b in budget_list:
        rs = by_budget.get(b, [])
        cm = [r for r in rs if r["conclusion_key"] is not None]
        rate = (sum(1 for r in cm if r["outcome"] == good) / len(cm)) if cm else None
        think = round(sum(r["thinking_tokens"] for r in rs) / len(rs)) if rs else 0
        bar = "#" * round((rate or 0) * 40)
        rate_s = f"{rate:.2f}" if rate is not None else " n/a"
        print(f"  budget={b:>{bw}}  think~{think:<6} {good}={rate_s}  |{bar:<40}| "
              f"n={len(cm)}/{len(rs)}")
    summary["rate_by_budget"] = {
        b: (lambda cm: (sum(1 for r in cm if r["outcome"] == good) / len(cm)) if cm else None)(
            [r for r in by_budget.get(b, []) if r["conclusion_key"] is not None])
        for b in budget_list
    }

    # A couple of actual traces, head + answer. This is the difference between "the
    # number looks wrong" and "the number is wrong because the model never reached the
    # request" -- exactly the failure the natural-length padded run hid.
    def _preview(tag, r):
        if r is None:
            return
        resp = r["response_text"]
        head = resp[:200].replace("\n", " ")
        answer = (resp.split("</think>", 1)[-1] if "</think>" in resp else "(no </think>)")[:280]
        print(f"\n  --- {tag}: {r['prompt_id']} budget={r['budget_condition']} "
              f"think={r['thinking_tokens']} outcome={r['outcome']} key={r['conclusion_key']}")
        print(f"      think head: {head!r}")
        print(f"      answer    : {answer.strip()!r}")

    print("\n====== trace previews ======")
    _preview("committed[0]", next((r for r in rows if r["conclusion_key"] is not None), None))
    _preview("refused", next((r for r in rows if r["conclusion_key"] == "refuse"), None))
    _preview("never-committed", next((r for r in rows if r["conclusion_key"] is None), None))
    if hijack_ids:
        _preview("warmup-hijack", next((r for r in rows if r["prompt_id"] in hijack_ids), None))

    print(f"\n[collect] {json.dumps(summary, indent=2)}")
    if hijack_ids:
        print("\n" + "=" * 70)
        print(f"  WARM-UP HIJACK: {len(hijack_ids)}/{len(rows)} trace(s) answered the")
        print("  benign warm-up instead of the harmful request. These are DEAD traces the")
        print("  heuristic keys `comply` -- exclude them before reading the ASR/refusal")
        print("  curve, and drop them from the drift pile (they never posed the request).")
        print("=" * 70)
    if summary["baseline_refusal_rate"] == 0:
        print("\n" + "=" * 70)
        print("  BASELINE REFUSAL RATE IS 0 -- nothing refused.")
        print("  Task 1.4's drift and stable piles are empty by construction here.")
        print("  The proposal predicts this for a math-specialized distill: it 'has a")
        print("  weaker refusal surface and may yield no safety signal to extract'.")
        print("  This is a 1.0b finding. Escalate before spending on 1.4.")
        print("=" * 70)

    # Belt and braces: _attach already ran on every row as it was built (so checkpointed
    # rows carry the bare request too). Re-running it is idempotent and covers any future
    # path that appends to `rows` without going through _attach.
    for r in rows:
        _attach(r)

    if on_commit is not None:
        on_commit()

    cfg = model.config
    return {"rows": rows, "summary": summary, "meta": {
        **meta, "dataset": dataset, "budgets": budget_list, "samples": samples,
        "rollouts": rollouts, "prefixes": prefixes, "induction": induction,
        "preamble_id": preamble_id or None, "seed": seed,
        # Which budget got which instrument. Without this the same budget_condition
        # string means "ceiling" in one run and "suppression" in another and nothing on
        # the row says so -- forcing_method does, per row, but the THRESHOLD that decided
        # it belongs in the run's provenance.
        "ceiling_max_budget": (int(ceiling_max_budget)
                               if ceiling_max_budget is not None else None),
        # The forcing OVERRIDE, and the string the wait arm spliced in. Both belong in run
        # provenance rather than only on the row: "wait with 'Wait'" and "wait with 'Take a
        # deep breath...'" produce rows whose forcing_method is identical, and a sweep that
        # cannot tell them apart is not a sweep. (Per row the string is in notes.)
        "forcing_override": forcing,
        "inject_string": inject_string,
        # The card the run ACTUALLY landed on, read off the device -- `gpu_label` in
        # the summary is what the launcher ASKED for, and on a shared box those are
        # not always the same thing.
        "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else "cpu",
        # Recorded on every run so "the agreed layers" can never quietly mean two
        # different depths across two models. Group 5 capture itself is NOT wired yet:
        # collect_trace does not expose the full generated sequence, and
        # capture_activations needs it (think_start_idx/think_end_idx index into it,
        # not into the prompt). That is a pipeline.py change and belongs with pile
        # collection, not the smoke ladder.
        "num_hidden_layers": getattr(cfg, "num_hidden_layers", None),
        "act_layers_agreed": list(act_layers),
        "activations_captured": False,
    }}
