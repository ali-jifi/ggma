"""Shared data-prep helpers: deterministic split assignment and subsetting.

Split is assigned at data-prep time from a hash of the prompt_id, so it is
stable across runs and machines and can never be re-tagged (R3). Fractions are
pre-registered here; do not tune.
"""

import hashlib
import random

SPLIT_FRACS = (("extraction", 0.50), ("selection", 0.25), ("heldout", 0.25))


def assign_split(prompt_id: str) -> str:
    """Deterministic extraction/selection/heldout assignment from prompt_id."""
    h = int(hashlib.sha256(str(prompt_id).encode()).hexdigest()[:12], 16)
    u = h / 16 ** 12
    acc = 0.0
    for name, frac in SPLIT_FRACS:
        acc += frac
        if u < acc:
            return name
    return SPLIT_FRACS[-1][0]


def deterministic_subset(items, n, seed=0):
    """Seeded sample of n items (order shuffled), stable across runs."""
    if n is None or n >= len(items):
        items = list(items)
    else:
        items = random.Random(seed).sample(list(items), n)
    return items
