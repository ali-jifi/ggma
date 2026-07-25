"""MATH-500 and AIME loading + prompt prep for the math side.

Same contract as load_gsm8k.load_gsm8k -- produces prompt dicts:
    {prompt_id, dataset, question, prompt_text, gold, split}

Both sets reuse the pre-registered GSM8K prompt template (the \\boxed
instruction the numeric matcher keys on) and the GSM8K numeric matcher.
Notes per set:

  math500  HuggingFaceH4/MATH-500 (500 problems, the OpenAI "Let's Verify Step
           by Step" subset). Answers may be LaTeX ("\\frac{1}{2}"), not just
           numbers -- use matchers.math_latex_matcher.Math500Matcher, whose
           shared normalize_math_answer() is also the gold_key here, so golds
           and predictions cannot drift. run_traces dispatches this
           automatically for --math-dataset math500.
  aime     AI-MO/aimo-validation-aime (90 problems, AIME 2022-2024). Answers
           are integers 0-999, so the numeric matcher applies cleanly. Small
           set by design -- it is the hard/long-reasoning knob, expect low
           accuracy and long traces (raise budgets and rollout caps).

Why these two: GSM8K barely makes a strong model wobble, so its calibration
data is all high-commitment (see the 14B calibration run: every surviving
point mapped to >= 0.92). Harder sets produce genuine low-commitment prefixes,
which is what the bottom of the isotonic curve needs.
"""

from pathlib import Path

from data.common import assign_split, deterministic_subset
from data.load_gsm8k import PROMPT_TEMPLATE
from matchers.math_latex_matcher import normalize_math_answer as gold_key

# name -> (hf repo, hf config, hf split, question field, answer field)
MATH_HF_SETS = {
    "math500": ("HuggingFaceH4/MATH-500", None, "test", "problem", "answer"),
    "aime": ("AI-MO/aimo-validation-aime", None, "train", "problem", "answer"),
}


def load_math_hf(name, n=None, seed=0, split_filter=None):
    """Load, template, and split-tag a HF math set ("math500" | "aime").

    Mirrors load_gsm8k: split is assigned from prompt_id (R3), subsetting to n
    happens after split filtering, and the seed picks which prompts you get.
    """
    if name not in MATH_HF_SETS:
        raise KeyError(f"unknown math set {name!r}; known: {sorted(MATH_HF_SETS)}")
    repo, config, hf_split, qfield, afield = MATH_HF_SETS[name]

    from datasets import load_dataset
    ds = load_dataset(repo, config, split=hf_split) if config else load_dataset(repo, split=hf_split)

    prompts = []
    for i, row in enumerate(ds):
        pid = f"{name}-{hf_split}-{i:05d}"
        question = row[qfield]
        prompts.append({
            "prompt_id": pid,
            "dataset": name,
            "question": question,
            "prompt_text": PROMPT_TEMPLATE.format(question=question),
            "gold": gold_key(row[afield]),
            "split": assign_split(pid),
        })
    if split_filter:
        prompts = [p for p in prompts if p["split"] == split_filter]
    return deterministic_subset(prompts, n, seed)


if __name__ == "__main__":
    import sys
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    for name in MATH_HF_SETS:
        ps = load_math_hf(name, n=3)
        print(f"--- {name} ---")
        for p in ps:
            print(p["prompt_id"], p["split"], "gold =", p["gold"])
