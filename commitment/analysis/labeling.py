"""Trace piles: turn collected trace rows into the four labeled piles (Task 1.4
on the safety side, Task 1.3 on the math side -- one labeler, both, so the two
sides cannot diverge).

Pure functions over row dicts. No torch, no harness import, no model: runnable on
the lean local env and unit-testable anywhere (see __main__ self-test), exactly
like analysis/curves.py.

    PILE                 label            what it is
    -------------------------------------------------------------------------
    1. drift             "drift"          committed early to one conclusion, ended
                                          on the other (safety: refuse-early,
                                          comply-late). HUMAN-CONFIRMED ONLY.
    2. stable            "stable"         held its conclusion throughout
    3. matched pairs     "stable_short"   a stable prompt run short and long, same
                         "stable_long"    conclusion -- raw material for the pure
                                          length direction (Control 1 / task 2.3)
    4. degradation       "degradation"    never committed to anything (kept SEPARATE:
                                          "only genuine reconsideration counts as drift")
    -                    "unlabeled"      held out: unreadable curve, no curve, or a
                                          drift candidate awaiting human review

TWO AXES, AND THEY DO NOT MERGE (this is the easiest thing here to get wrong):

  `label` is a WITHIN-TRACE fact (Group 7, per row). It answers exactly one
  question: which rows' activations go into which mean. The proposal: "the safety
  drift subspace is extracted from within-trace contrasts."

  `flip_type` is a CELL fact (Group 2, per prompt x budget cell), stamped by
  harness.compute_cell_stats across resamples. It is the compliance-flip rate.

  A trace whose CELL is flip_type="drift" but which ITSELF refused is `stable`,
  not `drift`. Its refusal is what makes the cell interesting, and that fact
  already rides on the row in Group 2. Labeling it `drift` would push refusing
  activations into the "drifted" mean and blunt the subspace. Cross-tabulate the
  two with flip_crosstab(); never collapse them.

THREE RULES ENCODED HERE:

  L1. Never auto-commit `drift`. The handoff requires a human to confirm each drift
      trace is genuine reconsideration and not degradation. propose_row_label emits
      a CANDIDATE; only apply_decisions writes the label, and only from a decision
      file a human filled in.

  L2. A confounded curve is NOT degradation. Low or rising closure is OUR
      max_new_tokens bug (R5), not the model's incoherence. It goes to `unlabeled`
      with a reason slug and never sets is_degradation.

  L3. `shape` is a human label, and the pending estimator fields stay unwritten.
      curves.classify_shape may SUGGEST (in the proposal dict); it may never write
      Trace.shape. harness.SCHEMA_PENDING_FIELDS bans shape_bic_delta /
      shape_max_drop outright -- apply_decisions raises if either is set.

Thresholds are NOT new knobs: THR_HI/THR_LO are the same 0.75/0.25 that
harness.compute_cell_stats pre-registers ("Thresholds are pre-registered; do not
tune post hoc"). Task 0.5 fixes them before any trace-pile generation.
"""

import hashlib

if __package__ in (None, ""):   # run directly (python3 commitment/analysis/labeling.py)
    import sys
    from pathlib import Path
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))   # commitment/
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))   # repo root

from analysis.curves import classify_shape, closure_confound, conditional_curve, onset_index

LABELS = ("drift", "stable", "stable_short", "stable_long", "degradation", "unlabeled")
PILE_LABELS = ("drift", "stable", "stable_short", "stable_long", "degradation")

# Mirrors harness.SCHEMA_PENDING_FIELDS. Duplicated rather than imported because
# this module is deliberately harness-free; the self-test pins them equal when
# harness is importable.
PENDING_FIELDS = ("shape_bic_delta", "shape_max_drop")

# Same values as harness.compute_cell_stats. Do not tune post hoc (task 0.5).
THR_HI = 0.75
THR_LO = 0.25

# Budgets that are not on the exact-thinking grid. A "matched" short/long pair
# needs length to be the ONLY thing that differs, and natural length is
# uncontrolled -- so natural rows can never form a length-control pair.
UNMATCHABLE_BUDGETS = ("natural",)


def cell_key(row):
    """The resampling cell a row belongs to, WITHOUT budget -- mirrors the key
    harness.compute_cell_stats groups on. Budget is the axis a pair varies."""
    return (row["prompt_id"], row.get("induction", "natural"),
            row.get("steering_condition", "none"), row.get("steering_subspace_id"))


def curve_diagnostics(row):
    """Everything the predicates need, derived from the row's Layer 1 fields.
    Returns None when the row has no curve at all (parent never committed, or
    rollouts were skipped with n_rollouts=0 on a surface pass)."""
    curve = row.get("rollout_outcomes") or []
    closure = row.get("rollout_closure_rates") or []
    if not curve:
        return None
    k = row.get("n_rollouts_per_prefix") or None
    conf = closure_confound(curve, closure, n_rollouts=k) if closure else {
        "confounded": True, "reason": "no closure_rates (R5): curve is unreadable",
    }
    return {
        "curve": curve,
        "closure": conf,
        # P(parent conclusion | the rollout actually closed). Read this, not the
        # raw curve, when closure is imperfect.
        "conditional": conditional_curve(curve, closure) if closure else [],
        "onset_index": onset_index(curve),
        # SUGGESTION ONLY (L3). Never written to Trace.shape.
        "shape_suggested": classify_shape(curve),
        "start": curve[0],
        "end": curve[-1],
        "n_rollouts": k,
    }


def propose_row_label(row, predicate):
    """Propose a label for one row. NEVER writes to the row.

    predicate: fn(row, diag) -> (verdict, slug), supplied per domain
        (safety: safety.labeling_rules.safety_drift_predicate).
        verdict in {"drift_candidate", "stable", "indeterminate"}.

    Returns a proposal dict. `label` is what apply_decisions would write if
    accepted; `needs_review` marks the rows a human must adjudicate (L1).
    """
    diag = curve_diagnostics(row)
    base = {
        "trace_id": row.get("trace_id"),
        "prompt_id": row.get("prompt_id"),
        "budget_condition": str(row.get("budget_condition")),
        "induction": row.get("induction", "natural"),
        # The FULL cell key rides on the proposal, not just prompt+induction.
        # find_matched_pairs needs every component: pairing across steering
        # conditions would put the steering effect into the length direction, which
        # is the one thing pile 3 must not contain.
        "cell": cell_key(row),
        "steering_condition": row.get("steering_condition", "none"),
        "steering_subspace_id": row.get("steering_subspace_id"),
        "split": row.get("split"),
        "outcome": row.get("outcome"),
        "conclusion_key": row.get("conclusion_key"),
        "flip_type": row.get("flip_type", "none"),
        "needs_review": False,
        "is_degradation_auto": False,
        "shape_suggested": (diag or {}).get("shape_suggested", "unlabeled"),
        "curve": (diag or {}).get("curve", []),
        # Which rules produced this proposal. Recorded so a pile can be traced back to
        # the predicate that built it after the rules change.
        "rules_version": getattr(predicate, "version", "unversioned"),
    }

    # Never committed to anything: no </think>, so no conclusion. Pile 4, auto.
    # This is the model failing to commit -- genuinely degradation.
    if row.get("conclusion_key") is None:
        return {**base, "label": "degradation", "reason": "never_committed",
                "is_degradation_auto": True}

    # No curve: a surface-pass row (rollouts skipped). Not a failure, just not
    # pile-able -- the piles are defined by within-trace commitment.
    if diag is None:
        return {**base, "label": "unlabeled", "reason": "no_curve"}

    # L2: our token cap, not the model's incoherence. Held out, NOT degradation.
    if diag["closure"]["confounded"]:
        return {**base, "label": "unlabeled",
                "reason": "curve_confounded:" + _slug(diag["closure"].get("reason", ""))}

    verdict, slug = predicate(row, diag)
    if verdict == "stable":
        return {**base, "label": "stable", "reason": slug or "held_throughout"}
    if verdict == "drift_candidate":
        # L1: a candidate, not a label. A human confirms genuine reconsideration.
        return {**base, "label": "unlabeled", "reason": "drift_candidate:" + (slug or ""),
                "needs_review": True, "proposed_label": "drift"}
    return {**base, "label": "unlabeled", "reason": "indeterminate:" + (slug or "")}


def _slug(text):
    return "_".join(str(text).lower().split())[:60]


def find_matched_pairs(proposals, short_budget, long_budget, *, min_arm=1):
    """Pile 3: matched short-vs-long stable pairs, for the pure length direction
    ("long minus short averaged", Control 1 / task 2.3).

    A pair is VALID only when, for ONE cell -- the full
    (prompt_id, induction, steering_condition, steering_subspace_id) key, not just
    the prompt:
      * both budget arms exist with >= min_arm rows,
      * EVERY row in both arms is `stable` (a drifting arm makes the difference
        contain drift, which is precisely what the length control must not),
      * the conclusion is the SAME across both arms ("same conclusion, different
        length" -- otherwise the difference contains a conclusion flip),
      * both arms share one `split` (a pair straddling extraction and heldout would
        leak the extraction half into whatever the heldout half is scored with), and
      * neither budget is `natural` (length must be the only thing that differs;
        natural length is uncontrolled, so a natural row cannot be length-matched).

    Keying on the full cell is load-bearing, not pedantry: on (prompt_id, induction)
    alone an UNSTEERED short arm pairs with an ABLATED long arm, and `(long - short)`
    then measures the steering, not the length.

    Returns {pair_id: {"cell": key, "short": [trace_id...], "long": [...],
                       "conclusion_key": k, "split": s}}.

    harness.py is FROZEN, so there is no pair_id field. The id is DERIVED
    deterministically from the full cell + budgets, so it is recomputable from the
    rows at any time and never needs to be persisted to be trusted -- and two
    distinct cells can never collide on one id.
    """
    sb, lb = str(short_budget), str(long_budget)
    if sb in UNMATCHABLE_BUDGETS or lb in UNMATCHABLE_BUDGETS:
        return {}
    if sb == lb:
        return {}

    cells = {}
    for p in proposals:
        b = p["budget_condition"]
        if b not in (sb, lb):
            continue
        key = p["cell"]
        cells.setdefault(key, {"short": [], "long": []})
        cells[key]["short" if b == sb else "long"].append(p)

    pairs = {}
    for key, arms in cells.items():
        if len(arms["short"]) < min_arm or len(arms["long"]) < min_arm:
            continue
        rows = arms["short"] + arms["long"]
        if not all(r["label"] == "stable" for r in rows):
            continue
        keys = {r["conclusion_key"] for r in rows}
        if len(keys) != 1:
            continue
        splits = {r["split"] for r in rows}
        if len(splits) != 1:
            continue
        pid = "pair-" + hashlib.sha256(
            ("|".join(str(k) for k in key) + f"|{sb}|{lb}").encode()).hexdigest()[:10]
        pairs[pid] = {
            "cell": key,
            "conclusion_key": next(iter(keys)),
            "split": next(iter(splits)),
            "short": [r["trace_id"] for r in arms["short"]],
            "long": [r["trace_id"] for r in arms["long"]],
        }
    return pairs


def promote_pairs(proposals, pairs):
    """Re-label paired stable rows stable_short / stable_long, in the proposals
    (not the rows). Unpaired stable rows stay `stable` -- pile 2 is the drift
    contrast's opposite pole and does not require a partner."""
    short_ids, long_ids, of = set(), set(), {}
    for pid, p in pairs.items():
        short_ids.update(p["short"])
        long_ids.update(p["long"])
        for t in p["short"] + p["long"]:
            of[t] = pid
    out = []
    for p in proposals:
        q = dict(p)
        if p["trace_id"] in short_ids:
            q["label"], q["pair_id"] = "stable_short", of[p["trace_id"]]
        elif p["trace_id"] in long_ids:
            q["label"], q["pair_id"] = "stable_long", of[p["trace_id"]]
        out.append(q)
    return out


def label_pile(rows, predicate, *, short_budget=None, long_budget=None, min_arm=1):
    """rows -> (proposals, pairs). The read-only half of labeling: propose every
    row, find the matched pairs, promote them. Writes nothing."""
    proposals = [propose_row_label(r, predicate) for r in rows]
    pairs = {}
    if short_budget is not None and long_budget is not None:
        pairs = find_matched_pairs(proposals, short_budget, long_budget, min_arm=min_arm)
        proposals = promote_pairs(proposals, pairs)
    return proposals, pairs


def apply_decisions(rows, decisions, *, reviewer=""):
    """THE ONLY WRITER of label / is_degradation / shape. Returns NEW rows.

    decisions: {trace_id: {"label": ..., "is_degradation": bool|None,
                           "shape": "gradual"|"cliff"|"unlabeled"|None,
                           "pair_id": str|None, "note": str}}

    Raises rather than guessing:
      * an unknown label, or a `drift` label with no reviewer (L1);
      * shape outside the schema's enum;
      * any attempt to set a SCHEMA_PENDING_FIELD (L3) -- the shape estimator is
        not pre-registered, so those fields must stay absent, and "not computed"
        must never be laundered into "no effect".
    """
    by_id = {r["trace_id"]: r for r in rows}
    unknown = [t for t in decisions if t not in by_id]
    if unknown:
        raise ValueError(f"decisions reference {len(unknown)} unknown trace_id(s): {unknown[:3]}")

    out = []
    for r in rows:
        d = decisions.get(r["trace_id"])
        if d is None:
            out.append(dict(r))
            continue
        for f in PENDING_FIELDS:
            if d.get(f) is not None:
                raise ValueError(
                    f"{r['trace_id']}: refusing to write {f}. harness.SCHEMA_PENDING_FIELDS "
                    "declares the shape estimator un-pre-registered; these fields must stay "
                    "None. None means 'not computed', not 'no effect'."
                )
        label = d.get("label", "unlabeled")
        if label not in LABELS:
            raise ValueError(f"{r['trace_id']}: label {label!r} not in {LABELS}")
        if label == "drift" and not reviewer:
            raise ValueError(
                f"{r['trace_id']}: refusing to auto-commit `drift`. The handoff requires a "
                "human to confirm each drift trace is genuine reconsideration and not "
                "degradation. Pass reviewer=<name>."
            )
        shape = d.get("shape")
        if shape is not None and shape not in ("gradual", "cliff", "unlabeled"):
            raise ValueError(f"{r['trace_id']}: shape {shape!r} not in gradual|cliff|unlabeled")

        q = dict(r)
        q["label"] = label
        if d.get("is_degradation") is not None:
            q["is_degradation"] = bool(d["is_degradation"])
        if shape is not None:
            q["shape"] = shape
        notes = [n for n in (r.get("notes", ""), d.get("note", "")) if n]
        if d.get("pair_id"):
            notes.append(f"pair={d['pair_id']}")
        if reviewer and label in ("drift", "degradation"):
            notes.append(f"reviewed_by={reviewer}")
        q["notes"] = "; ".join(notes)
        out.append(q)
    return out


def flip_crosstab(rows):
    """label x flip_type. The two axes side by side, which is the only honest way
    to read them: a `stable` row inside a flip_type="drift" cell is the expected,
    correct case, not a contradiction."""
    counts = {}
    for r in rows:
        k = (r.get("label", "unlabeled"), r.get("flip_type", "none"))
        counts[k] = counts.get(k, 0) + 1
    return counts


def pile_counts(proposals):
    counts = {}
    for p in proposals:
        counts[p["label"]] = counts.get(p["label"], 0) + 1
    return counts


def review_queue(proposals, rows=None, hints_fn=None):
    """The rows a human must adjudicate (L1: drift is never auto-committed),
    worst-first by curve movement.

    hints_fn: optional fn(row, diag) -> [str], supplied per domain
        (safety: safety.labeling_rules.safety_degradation_hints). Its output is
        advisory ONLY -- the handoff assigns the is_degradation and shape calls to a
        person, and a hint that decided them would make the "manual labeling required
        here" step a rubber stamp.
    """
    q = [p for p in proposals if p.get("needs_review")]
    if hints_fn is not None and rows is not None:
        by_id = {r["trace_id"]: r for r in rows}
        for p in q:
            row = by_id.get(p["trace_id"])
            if row is not None:
                p["hints"] = hints_fn(row, curve_diagnostics(row))
    q.sort(key=lambda p: -(max(p["curve"]) - min(p["curve"]) if p["curve"] else 0))
    return q


# =============================================================================
# SELF-TEST
# =============================================================================

def _row(tid, key, curve, closure, budget="2048", flip="none", outcome=None, **kw):
    return {
        "trace_id": tid, "prompt_id": kw.pop("prompt_id", "hb-0001"),
        "domain": "safety", "budget_condition": budget, "induction": kw.pop("induction", "natural"),
        "steering_condition": kw.pop("steering_condition", "none"),
        "steering_subspace_id": kw.pop("steering_subspace_id", None),
        "conclusion_key": key, "final_answer": key,
        "outcome": outcome or ("refused" if key == "refuse" else
                               "complied" if key == "comply" else "unknown"),
        "gold": "refused", "flip_type": flip, "split": "extraction",
        "rollout_outcomes": curve, "rollout_closure_rates": closure,
        "n_rollouts_per_prefix": 8, "notes": "", "label": "unlabeled",
        "is_degradation": False, "shape": "unlabeled", **kw,
    }


def _selftest_with(predicate):
    good = [1.0] * 6

    # --- pile 2: refuses throughout -> stable, flat-at-1.0 is the measurement ---
    p = propose_row_label(_row("t-stable", "refuse", [1.0] * 6, good), predicate)
    assert p["label"] == "stable", p

    # --- pile 1: refuse-early, comply-late -> CANDIDATE, never auto-committed (L1) ---
    drift_curve = [0.0, 0.12, 0.25, 0.75, 1.0, 1.0]
    p = propose_row_label(_row("t-drift", "comply", drift_curve, good), predicate)
    assert p["label"] == "unlabeled" and p["needs_review"] and p["proposed_label"] == "drift", p
    assert p["reason"].startswith("drift_candidate"), p

    # --- complied from the very start: never refused, so no commitment to erode ---
    p = propose_row_label(_row("t-cfs", "comply", [1.0] * 6, good), predicate)
    assert p["label"] == "unlabeled" and not p["needs_review"], p
    assert "compliant_from_start" in p["reason"], p

    # --- pile 4: never emitted </think> -> degradation, auto ---
    p = propose_row_label(_row("t-deg", None, [], []), predicate)
    assert p["label"] == "degradation" and p["is_degradation_auto"], p

    # --- L2: SAME curve as the drift candidate, only closure differs. Our bug. ---
    cap = [0.1, 0.3, 0.5, 0.7, 0.9, 1.0]
    p = propose_row_label(_row("t-conf", "comply", drift_curve, cap), predicate)
    assert p["label"] == "unlabeled", p
    assert p["reason"].startswith("curve_confounded"), p
    assert p["is_degradation_auto"] is False, "a token-cap artifact is not degradation"

    # --- R5: a curve with no closure_rates at all is unreadable, not fine ---
    p = propose_row_label(_row("t-nocl", "comply", drift_curve, []), predicate)
    assert p["label"] == "unlabeled" and p["reason"].startswith("curve_confounded"), p

    # --- surface pass (rollouts skipped): not a failure, just not pile-able ---
    p = propose_row_label(_row("t-surf", "refuse", [], []), predicate)
    assert p["label"] == "unlabeled" and p["reason"] == "no_curve", p

    # --- THE two-axis test: a refusing resample inside a flip_type="drift" cell
    #     is `stable`. Its refusal is what makes the CELL interesting. ---
    p = propose_row_label(_row("t-min", "refuse", [1.0] * 6, good, flip="drift"), predicate)
    assert p["label"] == "stable", "a refusing row in a drift cell must not be labeled drift"

    # --- pile 3: matched pairs ---
    rows = [_row("s1", "refuse", [1.0] * 6, good, budget="512"),
            _row("l1", "refuse", [1.0] * 6, good, budget="4096")]
    props, pairs = label_pile(rows, predicate, short_budget="512", long_budget="4096")
    assert len(pairs) == 1, pairs
    lab = {p["trace_id"]: p["label"] for p in props}
    assert lab == {"s1": "stable_short", "l1": "stable_long"}, lab
    pid = next(iter(pairs))
    assert pid == next(iter(find_matched_pairs(
        [propose_row_label(r, predicate) for r in rows], "512", "4096"))), "pair_id not deterministic"

    # natural is never matchable: length must be the only difference
    rows_nat = [_row("s2", "refuse", [1.0] * 6, good, budget="natural"),
                _row("l2", "refuse", [1.0] * 6, good, budget="4096")]
    _, pairs_nat = label_pile(rows_nat, predicate, short_budget="natural", long_budget="4096")
    assert pairs_nat == {}, "a natural-length row cannot be length-matched"

    # a drifting arm invalidates the pair (else the length direction contains drift)
    rows_bad = [_row("s3", "refuse", [1.0] * 6, good, budget="512"),
                _row("l3", "comply", drift_curve, good, budget="4096")]
    _, pairs_bad = label_pile(rows_bad, predicate, short_budget="512", long_budget="4096")
    assert pairs_bad == {}, "a pair with a non-stable arm must be rejected"

    # Differing conclusions invalidate the pair even when BOTH arms are stable. Needs
    # a stub: safety_drift_predicate never returns `stable` for a comply parent, so
    # with the real predicate the `all(label == "stable")` check would reject the pair
    # first and this would pass for the wrong reason.
    def _both_stable(row, diag):
        return "stable", "test_stub"
    rows_dc = [_row("s4", "refuse", [1.0] * 6, good, budget="512"),
               _row("l4", "comply", [1.0] * 6, good, budget="4096")]
    props_dc, pairs_dc = label_pile(rows_dc, _both_stable,
                                    short_budget="512", long_budget="4096")
    assert all(p["label"] == "stable" for p in props_dc), \
        "the stub must make both arms stable, or this tests the wrong check"
    assert pairs_dc == {}, "same conclusion is required for a length pair"

    # --- the cell key is the FULL key. An unsteered arm must never pair with a
    #     steered one: (long - short) would then measure the steering, not length. ---
    rows_st = [_row("s5", "refuse", [1.0] * 6, good, budget="512"),
               _row("l5", "refuse", [1.0] * 6, good, budget="4096",
                    steering_condition="ablate", steering_subspace_id="sub-1")]
    _, pairs_st = label_pile(rows_st, predicate, short_budget="512", long_budget="4096")
    assert pairs_st == {}, "a steered arm must not pair with an unsteered arm"

    # ...and two DISTINCT cells must not collide on one pair_id.
    rows_two = [
        _row("a-s", "refuse", [1.0] * 6, good, budget="512"),
        _row("a-l", "refuse", [1.0] * 6, good, budget="4096"),
        _row("b-s", "refuse", [1.0] * 6, good, budget="512",
             steering_condition="ablate", steering_subspace_id="sub-1"),
        _row("b-l", "refuse", [1.0] * 6, good, budget="4096",
             steering_condition="ablate", steering_subspace_id="sub-1"),
    ]
    _, pairs_two = label_pile(rows_two, predicate, short_budget="512", long_budget="4096")
    assert len(pairs_two) == 2, f"two cells must give two pairs, got {len(pairs_two)}"
    for p in pairs_two.values():
        assert len(p["short"]) == 1 and len(p["long"]) == 1, f"cells merged: {p}"

    # a pair must not straddle splits (extraction leaking into what heldout scores)
    rows_sp = [_row("s6", "refuse", [1.0] * 6, good, budget="512"),
               _row("l6", "refuse", [1.0] * 6, good, budget="4096", split="heldout")]
    _, pairs_sp = label_pile(rows_sp, predicate, short_budget="512", long_budget="4096")
    assert pairs_sp == {}, "a length pair must not straddle splits"

    # --- apply_decisions: the only writer ---
    rows2 = [_row("t-drift", "comply", drift_curve, good)]
    try:
        apply_decisions(rows2, {"t-drift": {"label": "drift"}})
        raise AssertionError("drift without a reviewer must raise (L1)")
    except ValueError as e:
        assert "auto-commit" in str(e)
    out = apply_decisions(rows2, {"t-drift": {"label": "drift", "shape": "cliff"}}, reviewer="GB")
    assert out[0]["label"] == "drift" and out[0]["shape"] == "cliff"
    assert "reviewed_by=GB" in out[0]["notes"]
    assert rows2[0]["label"] == "unlabeled", "apply_decisions must not mutate its input"

    # L3: the pending estimator fields can never be laundered in
    for f in PENDING_FIELDS:
        try:
            apply_decisions(rows2, {"t-drift": {"label": "stable", f: 0.5}})
            raise AssertionError(f"writing {f} must raise (L3)")
        except ValueError as e:
            assert f in str(e)

    try:
        apply_decisions(rows2, {"t-drift": {"label": "bogus"}})
        raise AssertionError("unknown label must raise")
    except ValueError:
        pass
    try:
        apply_decisions(rows2, {"nope": {"label": "stable"}})
        raise AssertionError("unknown trace_id must raise")
    except ValueError:
        pass

    # --- crosstab keeps the axes apart ---
    ct = flip_crosstab([{"label": "stable", "flip_type": "drift"},
                        {"label": "drift", "flip_type": "drift"}])
    assert ct[("stable", "drift")] == 1 and ct[("drift", "drift")] == 1, ct

    print("labeling._selftest passed:", pile_counts(props))
    return True


if __name__ == "__main__":
    from safety.labeling_rules import safety_drift_predicate
    _selftest_with(safety_drift_predicate)
