# stage0_smoke.py
# Purpose: prove the toolkit works on ONE prompt before relying on it at scale.
# Checks three things: (a) generation + commitment metric, (b) activation hooks,
# (c) the </think> token id resolves. Run this inside your Modal function where
# `model` and `tokenizer` (DeepSeek-R1-Distill-Llama-8B) are already loaded.

from harness import (generate_natural, capture_activations,
                     find_think_indices, pool_thinking, build_input_ids,
                     get_think_end_id)


def stage0_smoke(model, tokenizer):
    # --- pre-check: the single most environment-fragile piece ---
    tid = get_think_end_id(tokenizer)
    print("</think> token id:", tid, "(must be a real int, not None)")

    prompt = "What is 17 times 24? Show your reasoning."

    # (a) generation + commitment metric
    text, think_tokens, margins, entropies, full_ids = generate_natural(
        model, tokenizer, prompt, max_new_tokens=1024, seed=0)
    print("thinking tokens:", think_tokens)
    print("num margin values:", len(margins), "(should ~= total generated tokens)")
    print("margin range:", round(min(margins), 3), "to", round(max(margins), 3),
          "(should sit within 0..1)")
    print("first 5 margins:", [round(m, 3) for m in margins[:5]])
    print("answer tail:", repr(text[-120:]))

    # (b) activation capture at a few layers
    prompt_len = build_input_ids(model, tokenizer, prompt).shape[1]
    ts, te = find_think_indices(tokenizer, full_ids, prompt_len)
    print("think span (start,end):", ts, te)
    acts = capture_activations(model, full_ids, layers=[8, 12, 16, 20, 24])
    pooled = pool_thinking(acts[16], ts, te)
    print("layer 16 activations shape:", tuple(acts[16].shape),
          "(should be (seq_len, 4096))")
    print("pooled vector shape:", tuple(pooled.shape), "(should be (4096,))")


# call inside your Modal function:
#     stage0_smoke(model, tokenizer)