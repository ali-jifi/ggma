"""Batched generation must equal unbatched generation. GPU test -- skipped without one.

WHY THIS TEST IS THE WHOLE POINT. Batching a decoder-only model needs LEFT padding and
a correct attention mask. Get it wrong and the model attends to pad tokens: the output
is subtly worse, nothing raises, and every downstream number is quietly wrong. There is
no way to eyeball that. So we pin it: batched-GREEDY must equal unbatched-GREEDY, token
for token, per prompt, at several batch sizes.

GREEDY, not sampled: at temp 0.6 the batch-1 and batch-N paths consume the RNG
differently, so they draw different (equally valid) samples. Only greedy admits an exact
comparison. If greedy matches at batch 1, 2 and 8, the padding and mask are right, and
the sampled path shares that code.

Run:  python3 safety/tests/test_batch_equivalence.py --model <hf_id>
      (needs torch + transformers + a GPU; NOT part of the torch-free CI suite)
"""

import argparse
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT / "commitment"))

# Short, benign, and of DELIBERATELY different lengths -- equal-length prompts would
# pad identically and the test would pass with the padding broken.
PROMPTS = [
    "What is 2+2?",
    "Name three primary colours and explain briefly why they are called primary.",
    "Hi.",
    "Write one sentence about the sea.",
    "Explain, in a short paragraph, what a compiler does and why a program needs one "
    "before it can run on a machine.",
    "List two fruits.",
]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B")
    ap.add_argument("--max-new-tokens", type=int, default=48)
    ap.add_argument("--batch-sizes", type=int, nargs="+", default=[1, 2, 8])
    args = ap.parse_args()

    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    import harness as H
    from scripts.batch_generate import generate_natural_batch

    if not torch.cuda.is_available():
        print("[batch-eq] no GPU; skipping")
        return 0

    tok = AutoTokenizer.from_pretrained(args.model)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        args.model, torch_dtype=torch.bfloat16, device_map="cuda").eval()
    model.requires_grad_(False)

    # Reference: the frozen engine, one prompt at a time, greedy.
    print(f"[batch-eq] reference: harness.generate_natural x{len(PROMPTS)}, greedy")
    ref = []
    for p in PROMPTS:
        ids = H.build_input_ids(model, tok, p)
        out = model.generate(ids, max_new_tokens=args.max_new_tokens, do_sample=False,
                             pad_token_id=H.get_pad_id(model, tok))
        # Same cut rule as the batched arm (union of stop ids + pad).
        new = H.cut_at_first_stop(
            out[0, ids.shape[1]:],
            set(H.get_eos_ids(model, tok)) | {H.get_pad_id(model, tok)})
        ref.append(tok.decode(new, skip_special_tokens=True))

    bad = 0
    for bs in args.batch_sizes:
        rows = generate_natural_batch(model, tok, PROMPTS,
                                      max_new_tokens=args.max_new_tokens,
                                      batch_size=bs, greedy=True)
        assert len(rows) == len(PROMPTS), f"batch_size={bs}: got {len(rows)} rows"
        assert [r.prompt_index for r in rows] == list(range(len(PROMPTS))), \
            f"batch_size={bs}: rows came back out of order -- batching reordered prompts"
        mismatch = [i for i, (r, e) in enumerate(zip(rows, ref)) if r.text != e]
        if mismatch:
            bad += 1
            print(f"[batch-eq] FAIL batch_size={bs}: {len(mismatch)}/{len(PROMPTS)} differ")
            for i in mismatch[:2]:
                print(f"    prompt {i}: {PROMPTS[i][:40]!r}")
                print(f"      unbatched: {ref[i][:90]!r}")
                print(f"      batched  : {rows[i].text[:90]!r}")
        else:
            print(f"[batch-eq] ok   batch_size={bs}: all {len(PROMPTS)} identical")

        # prompt_tokens must be the REAL length, not the padded one -- if padding leaks
        # into the count, every total_tokens on every row is inflated.
        for i, r in enumerate(rows):
            real = H.build_input_ids(model, tok, PROMPTS[i]).shape[1]
            if r.prompt_tokens != real:
                bad += 1
                print(f"[batch-eq] FAIL batch_size={bs}: prompt {i} prompt_tokens="
                      f"{r.prompt_tokens}, expected {real} (padding leaked into the count)")
                break

    if bad:
        print("\n[batch-eq] FAIL: batched != unbatched. Left padding or the attention "
              "mask is wrong; do NOT collect with this.")
        return 1
    print("\n[batch-eq] PASS: batching is equivalent to the frozen engine.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
