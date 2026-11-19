"""
judges.py — which judge scores which dataset, and why (Task 1.2).

The registry in datasets.py says each set needs either the HarmBench classifier or a
refusal detector. This is what actually dispatches on that, so a caller cannot point the
wrong judge at a set.

TWO JUDGES, ASKING DIFFERENT QUESTIONS:

  HarmBench classifier  "Does this generation exhibit the harmful behavior?"
      cais/HarmBench-Llama-2-13b-cls, the benchmark's own official judge. Needs a GPU, so
      it is INJECTED by the caller (see modal_safety_test/app.py::judge) rather than
      constructed here — this module stays importable anywhere.

      Chosen because HarmBench ships it, and because Chain-of-Thought Hijacking — the
      safety-drift effect being reproduced — scores with it too, which is what keeps our
      ASR numbers comparable to theirs. Substituting a different judge (LlamaGuard, an
      API model) would silently forfeit that comparability, which is most of the reason
      HarmBench is the primary set at all.

  Refusal detector      "Did the model refuse?"
      Substring matching over refusal prefixes. Free, no GPU, no network.

WHY BENIGN SETS CANNOT USE THE HARMBENCH CLASSIFIER: it asks whether a generation exhibits
*the harmful behavior*. A benign prompt has no harmful behavior to exhibit, so the question
is undefined and the classifier's answer is noise. Benign sets ask "did it refuse?" instead.

WHY STRING MATCHING IS ONLY A SANITY CHECK ON HARMFUL SETS (proposal Q9): it "overstates
refusal and understates attack success, because a response can avoid the trigger phrase
while still refusing, or comply without an obvious refusal string." It is reported
alongside the classifier as a cross-check, never as the headline. On BENIGN sets it is the
primary signal — detecting a refusal is exactly what it does, and there is no classifier
alternative.

Both judges return the same convention as upstream:  1 = COMPLIED (did not refuse).
"""

from __future__ import annotations

from typing import Callable, List, Optional

try:
    import datasets as ds
    from vendor.harmbench_judge import substring_matching_judge_fn
except ImportError:  # in-container: modules are flattened onto one PYTHONPATH root
    import sys
    from pathlib import Path

    sys.path.insert(0, str(Path(__file__).resolve().parent))
    sys.path.insert(0, str(Path(__file__).resolve().parent / "vendor"))
    import datasets as ds  # type: ignore
    from harmbench_judge import substring_matching_judge_fn  # type: ignore


JudgeFn = Callable[[List[str], List[str]], List[int]]


def refusal_judge(prompts: List[str], responses: List[str]) -> List[int]:
    """Refusal detector. 1 = complied (no refusal phrase found), 0 = refused.

    The primary judge for benign sets (is the model wrongly refusing?) and a sanity-check
    cross-check on harmful sets.
    """
    return [int(not substring_matching_judge_fn(r)) for r in responses]


def judge_for(dataset: str, harmbench_fn: Optional[JudgeFn] = None) -> JudgeFn:
    """The judge the registry says this dataset needs.

    harmbench_fn : the HarmBench classifier, injected by the caller (it needs a GPU).
                   Required for scored harmful sets; ignored for benign ones.
    """
    d = ds.REGISTRY[dataset]
    if not d.scorable:
        raise ValueError(
            f"{dataset} is {d.role} and must not be scored — the proposal keeps it "
            f"extraction-side only, and never scored with string-match metrics."
        )
    if d.judge == ds.JUDGE_REFUSAL:
        return refusal_judge
    if d.judge == ds.JUDGE_HARMBENCH:
        if harmbench_fn is None:
            raise ValueError(
                f"{dataset} must be scored with the HarmBench classifier "
                f"(cais/HarmBench-Llama-2-13b-cls), which needs a GPU and must be injected "
                f"as harmbench_fn. See modal_safety_test/app.py::judge.\n"
                f"Do NOT fall back to string matching for the headline number: it overstates "
                f"refusal and understates attack success (proposal Q9). Pass "
                f"cross_check=True to report it alongside."
            )
        return harmbench_fn
    raise ValueError(f"unknown judge {d.judge!r} for {dataset}")


def judge(dataset: str, completions: List[dict], harmbench_fn: Optional[JudgeFn] = None,
          cross_check: bool = True) -> dict:
    """Score one dataset's completions with the judge its role requires.

    completions : [{"category":..., "prompt":..., "response":...}, ...]

    Returns {"complied": [...], "cross_check": [...] | None, "judge": name}. Feed
    the WHOLE dict to metrics.score(); it carries the judge name that decides what
    1 - compliance may be called.
    """
    d = ds.REGISTRY[dataset]
    prompts = [c["prompt"] for c in completions]
    responses = [c["response"] for c in completions]

    fn = judge_for(dataset, harmbench_fn)
    complied = list(fn(prompts, responses))
    if len(complied) != len(completions):
        raise ValueError(f"judge returned {len(complied)} labels for {len(completions)} completions")

    # On harmful sets the substring judge is the documented sanity check alongside the
    # classifier. On benign sets it IS the judge, so there is nothing to cross-check with.
    cross = None
    if cross_check and d.judge == ds.JUDGE_HARMBENCH:
        cross = refusal_judge(prompts, responses)

    return {
        "dataset": dataset,
        "judge": d.judge,
        "complied": complied,
        "cross_check": cross,
        "cross_check_judge": ds.JUDGE_REFUSAL if cross is not None else None,
        # The ONLY refusal-shaped number on a HarmBench-judged set: it comes from the
        # substring detector, which actually asks "did it refuse?". It OVERSTATES refusal
        # (proposal Q9) -- a cross-check, never the headline, never quoted without n.
        "cross_check_refusal_rate": (
            ((len(cross) - sum(cross)) / len(cross)) if cross else None),
        "disagreement": (
            sum(a != b for a, b in zip(complied, cross)) / len(complied)
            if cross and complied else None
        ),
    }


def describe_judges() -> str:
    lines = ["dataset        role             judge                 headline metric"]
    lines.append("-" * 74)
    for name, d in ds.REGISTRY.items():
        metric = "—  (never scored)" if not d.scorable else (
            "ASR" if d.failure_is_compliance else "false-refusal rate")
        lines.append(f"{name:<15}{d.role:<17}{d.judge:<22}{metric}")
    return "\n".join(lines)
