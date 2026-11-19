"""ConclusionMatcher interface (the pluggable point of the Two Drifts harness).

Contract (pre-registered):
    match(completion: str) -> Optional[str]
Returns a *conclusion key* -- a normalized, comparable label for the conclusion
the completion reached -- or None if the completion never committed to one
(e.g. a rollout that ran out of tokens mid-think and never emitted </think>).

  math   : normalized numeric string ("42", "3.5") or None
  safety : "refuse" | "comply" | None

harness.rollout_commitment_curve wants Callable[[str], bool] ("did this rollout
reach the parent trace's final conclusion?"); to_match_fn() adapts a matcher +
parent key into that. Swapping math -> safety is a matcher swap only (R4):
same truncation, same sampling, same curve computation.

This module deliberately does not import harness.py (torch-free, so matchers
are unit-testable on machines without a GPU stack).
"""

from typing import Callable, Optional, Protocol, runtime_checkable


@runtime_checkable
class ConclusionMatcher(Protocol):
    name: str
    version: str

    def match(self, completion: str) -> Optional[str]:
        ...


def scorer_version(matcher: ConclusionMatcher) -> str:
    """String for the Trace.scorer_version field."""
    return f"{matcher.name}-{matcher.version}"


def to_match_fn(matcher: ConclusionMatcher, parent_key: str) -> Callable[[str], bool]:
    """Adapt a matcher into harness.rollout_commitment_curve's
    conclusion_match_fn. parent_key is the conclusion key of the parent
    trace's final answer (e.g. matcher.match(parent_response_text))."""
    if parent_key is None:
        raise ValueError("parent trace has no conclusion; nothing to match rollouts against")

    def fn(completion: str) -> bool:
        return matcher.match(completion) == parent_key

    return fn


def agreement_rate(matcher_a: ConclusionMatcher, matcher_b: ConclusionMatcher,
                   completions) -> Optional[float]:
    """Fraction of completions on which two matchers return the same key.
    Used on the safety side to record matcher_agreement_rate (classifier vs.
    heuristic cross-check). None if completions is empty."""
    completions = list(completions)
    if not completions:
        return None
    hits = sum(1 for c in completions if matcher_a.match(c) == matcher_b.match(c))
    return hits / len(completions)
