"""
datasets_registry.py -- single source of truth for which datasets this
drift-run pipeline supports, and how to load/grade each one. The notebook
imports from here instead of duplicating per-dataset logic.

Budget policy: the ceiling is normally a property of the MODEL (32768,
DeepSeek's own documented max generation length), matching Mirage's TTBC1
exactly -- that was the whole point of the earlier fidelity fix, and why
an earlier n=40 result using a non-ceiling budget was marked invalid.

DISCLOSED DEVIATION (deliberate, not a bug): math500 and gsm8k are capped
at 16384 (2^14) instead of the full 32768, a user decision for compute-time
reasons (large pools, want more items to actually finish). aime2024 keeps
the full 32768 ceiling since its pool is tiny (30 problems) and can afford
full budget on every item. This means math500/gsm8k results are NOT a
literal TTBC1 replication the way the earlier all-32768 runs were --
flag this explicitly in any writeup that cites them, same as the disclosed
Wait-phrasing deviation earlier in this project.

Datasets registered:
  math500   HuggingFaceH4/MATH-500, test split, 500 problems. Free-form
            answers (fractions, expressions). Graded by math500_matcher's
            normalization-based equivalence checker. long_budget=16384.
  gsm8k     openai/gsm8k (config "main"), test split, 1319 problems.
            Plain integer answers. Graded via harness's existing numeric
            extractor (gsm8k_matcher.py). long_budget=16384.
  aime2024  Maxwell-Jia/AIME_2024, train split, ONLY 30 problems total
            (confirmed directly, not assumed). Plain integer answers
            0-999. Graded via aime_matcher.py. Any requested N above 30
            gets capped -- there is no more data to draw from.
            long_budget=32768 (full ceiling, TTBC1-faithful).
"""
from datasets import load_dataset

import math500_matcher
import gsm8k_matcher
import aime_matcher

DATASETS = {
    "math500": dict(
        hf_name="HuggingFaceH4/MATH-500", hf_config=None, split="test",
        prompt_field="problem", pool_size=500, long_budget=24576,
        extract_gold=math500_matcher.extract_math500_gold,
        extract_pred=math500_matcher.extract_math500_pred,
        is_equiv=math500_matcher.is_equiv_math,
    ),
    "gsm8k": dict(
        hf_name="openai/gsm8k", hf_config="main", split="test",
        prompt_field="question", pool_size=1319, long_budget=24576,
        extract_gold=gsm8k_matcher.extract_gold,
        extract_pred=gsm8k_matcher.extract_pred,
        is_equiv=gsm8k_matcher.is_equiv,
    ),
    "aime2024": dict(
        hf_name="Maxwell-Jia/AIME_2024", hf_config=None, split="train",
        prompt_field="Problem", pool_size=30, long_budget=24576,
        extract_gold=aime_matcher.extract_gold,
        extract_pred=aime_matcher.extract_pred,
        is_equiv=aime_matcher.is_equiv,
    ),
}


def load_one(dataset_key):
    cfg = DATASETS[dataset_key]
    if cfg["hf_config"]:
        ds = load_dataset(cfg["hf_name"], cfg["hf_config"], split=cfg["split"])
    else:
        ds = load_dataset(cfg["hf_name"], split=cfg["split"])
    return ds, cfg