"""GSM8K loading + prompt prep for the math side.

Produces prompt dicts consumed by scripts/:
    {prompt_id, dataset, question, prompt_text, gold, split}
gold is already normalized (matchers.gsm8k_matcher.gold_key).

Source: HuggingFace datasets ("openai/gsm8k", "main") by default, or a local
JSONL with {"question", "answer"} rows via local_path (for offline machines).
"""

import json
from pathlib import Path

from data.common import assign_split, deterministic_subset
from matchers.gsm8k_matcher import gold_key

# Pre-registered prompt template (R1-adjacent: the \boxed instruction is what
# the numeric matcher's first extraction pattern keys on).
PROMPT_TEMPLATE = (
    "{question}\n\n"
    "Please reason step by step, and put your final answer within \\boxed{{}}."
)


def _iter_hf(hf_split):
    from datasets import load_dataset
    ds = load_dataset("openai/gsm8k", "main", split=hf_split)
    for i, row in enumerate(ds):
        yield i, row["question"], row["answer"]


def _iter_local(path):
    with open(path, encoding="utf-8") as f:
        for i, line in enumerate(f):
            row = json.loads(line)
            yield i, row["question"], row["answer"]


def load_gsm8k(hf_split="test", n=None, seed=0, local_path=None,
               split_filter=None):
    """Load, template, and split-tag GSM8K prompts.

    split_filter: None for all, or "extraction"/"selection"/"heldout" to keep
    only that split (subsetting to n happens after filtering, so n is the
    number of prompts you actually run).
    """
    rows = _iter_local(local_path) if local_path else _iter_hf(hf_split)
    prompts = []
    for i, question, answer in rows:
        pid = f"gsm8k-{hf_split}-{i:05d}"
        prompts.append({
            "prompt_id": pid,
            "dataset": "gsm8k",
            "question": question,
            "prompt_text": PROMPT_TEMPLATE.format(question=question),
            "gold": gold_key(answer),
            "split": assign_split(pid),
        })
    if split_filter:
        prompts = [p for p in prompts if p["split"] == split_filter]
    return deterministic_subset(prompts, n, seed)


if __name__ == "__main__":
    import sys
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    ps = load_gsm8k(n=3)
    for p in ps:
        print(p["prompt_id"], p["split"], "gold =", p["gold"])
