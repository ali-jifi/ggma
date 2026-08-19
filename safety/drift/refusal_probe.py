"""Per-token refusal-intention probe -- the instrument that separates drift from preemption.

Every "did a refusal form?" claim in this project has rested on COT_REFUSAL, a regex that can only
see refusal TEXT. It cannot see an intention the model holds but never writes down, and it fires on
noted-then-overridden concerns, so it is a superset in one direction and blind in the other. That is
why the drift-versus-dilution question has stayed open for the whole program.

Following Refusal Cliff (arXiv:2510.06036), which traces refusal intention by linear probing across
token positions, this scores EVERY token of a trace by projecting its residual stream onto the
refusal direction (Arditi diff-in-means, reused from run_ablation.extract_direction). The result is
a curve, not a label, and the SHAPE of that curve is the discriminator:

    intention never rises            -> PREEMPTION   (no commitment ever formed; nothing to erode)
    intention rises then collapses   -> CLIFF/DRIFT  (a commitment formed and then gave way)
    intention rises and holds        -> HELD         (refusal maintained to the answer)

Calibration matters and is not optional: raw projections have no natural zero. Thresholds are set
from a reference set -- benign prompts (intention should stay low) versus harmful prompts the model
actually refuses (intention should go high) -- so "rises" means "rises relative to what this model
does when it genuinely refuses", not relative to an arbitrary constant.

The pure curve-shape logic lives in classify_curve() and is unit-tested without a GPU.
"""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
for _p in (ROOT, ROOT / "safety", ROOT / "safety" / "drift"):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

PREEMPTION, CLIFF, HELD, FLAT_HIGH = "preemption", "cliff", "held", "held_high"


# ------------------------------- pure logic (tested) -------------------------------
def summarize_curve(scores, start=0):
    """Shape features of a refusal-intention curve over generated tokens.

    `start` lets the caller ignore prompt positions. Returns peak, where the peak sits (as a
    fraction of the generated span), the value at the end, and the drop from peak to end -- the
    'cliff depth'.
    """
    seg = list(scores[start:]) if start else list(scores)
    if not seg:
        return {"n": 0, "peak": 0.0, "peak_frac": None, "final": 0.0, "drop": 0.0, "mean": 0.0}
    peak = max(seg)
    pi = seg.index(peak)
    tail = seg[-max(1, len(seg) // 20):]          # last 5% = the pre-answer window
    final = sum(tail) / len(tail)
    return {"n": len(seg), "peak": peak, "peak_frac": round(pi / max(1, len(seg) - 1), 4),
            "final": final, "drop": peak - final, "mean": sum(seg) / len(seg)}


def classify_curve(summary, form_thresh, drop_thresh):
    """Turn curve shape into a mechanism label.

    form_thresh -- intention must exceed this for a commitment to count as FORMED. Calibrate from
                   traces where the model actually refuses.
    drop_thresh -- peak-to-end fall that counts as a collapse. Calibrate from held refusals.

    Note the asymmetry: PREEMPTION is a claim that nothing formed, so it is only returned when the
    peak stays below threshold. A trace that forms and holds is HELD, never 'no drift by default'.
    """
    if summary["n"] == 0:
        return "empty"
    if summary["peak"] < form_thresh:
        return PREEMPTION
    return CLIFF if summary["drop"] >= drop_thresh else HELD


def calibrate(refused_summaries, benign_summaries, quantile=0.25):
    """Thresholds from reference curves rather than guesswork.

    form_thresh: a low quantile of the peaks seen on genuinely-refused traces, floored above what
    benign traces reach, so 'formed' means 'looks like this model refusing'.
    drop_thresh: a high quantile of the drops seen on refused-and-held traces -- a collapse has to
    exceed the ordinary wobble of a refusal that never gave way.
    """
    def q(xs, p):
        if not xs:
            return 0.0
        s = sorted(xs)
        i = min(len(s) - 1, max(0, int(p * (len(s) - 1))))
        return s[i]

    rp = [s["peak"] for s in refused_summaries]
    bp = [s["peak"] for s in benign_summaries]
    form = max(q(rp, quantile), q(bp, 0.9)) if rp else q(bp, 0.9)
    drop = q([s["drop"] for s in refused_summaries], 0.9) if rp else 0.0
    return {"form_thresh": form, "drop_thresh": drop,
            "n_refused": len(rp), "n_benign": len(bp),
            "refused_peak_median": q(rp, 0.5), "benign_peak_median": q(bp, 0.5),
            "separation": q(rp, 0.5) - q(bp, 0.5)}


# ------------------------------- GPU side -------------------------------
def _probe_scores(model, tok, seq_ids, direction, layer):
    """Projection of each position's residual stream onto the refusal direction, at `layer`."""
    import torch
    import steering_primitives as sp
    layers = sp.get_decoder_layers(model)
    dhat = (direction / (direction.norm() + 1e-8)).to(next(model.parameters()).device).float()
    store = {}

    def hook(mod, args, output):
        h = output[0] if isinstance(output, tuple) else output
        store["h"] = h.detach()[0].float()          # (seq, hidden)
        return output

    hd = layers[layer].register_forward_hook(hook)
    try:
        with torch.no_grad():
            # Only the hook's hidden states are wanted. Left to itself the model materializes
            # logits for EVERY position (seq x 152k vocab), ~1.2GB at 3k tokens -- irrelevant on an
            # A100, decisive on a 16GB card. Ask for one position; older transformers used a
            # different kwarg name, and oldest support neither.
            for kw in ({"logits_to_keep": 1}, {"num_logits_to_keep": 1}, {}):
                try:
                    model(input_ids=seq_ids, **kw)
                    break
                except TypeError:
                    continue
    finally:
        hd.remove()
    return (store["h"] @ dhat).tolist()


def trace_curve(model, tok, prompt_ids, gen_ids, direction, layer):
    """Refusal-intention curve over the GENERATED tokens of one trace.

    Scores the full sequence in a single forward pass (teacher-forced over already-sampled tokens),
    then reports only the generated span -- prompt positions are not part of the trace's own
    trajectory and would otherwise dominate the peak.
    """
    import torch
    seq = torch.cat([prompt_ids, gen_ids.unsqueeze(0)], dim=1) if gen_ids.dim() == 1 else \
        torch.cat([prompt_ids, gen_ids], dim=1)
    scores = _probe_scores(model, tok, seq, direction, layer)
    return scores[prompt_ids.shape[1]:]
