"""Wait-injection budget forcing (s1 / TTBC-2). New behavior, iterable tree.

WHAT IT IS. s1's budget forcing (Muennighoff et al., 2501.19393), used verbatim by
2506.04210 as TTBC-2: whenever the model tries to close the think block, suppress the
delimiter, append the literal string "Wait", and resume decoding -- repeating until the
cumulative thinking-token count reaches exactly the budget, at which point </think> is
allowed through. The injected tokens COUNT toward the budget; they are part of the
thinking trace, not scaffolding around it.

WHY IT IS NOT THE SUPPRESSION ARM. `harness.generate_exact_thinking` blocks stopping and
lets the model ramble to the budget. Measured on Qwen3-4B-Thinking, that produces filler:
duplicated-12-word-shingle rate 0.174 at budget 1536 but 0.671 at 4096, 28/30 traces over
0.5 (safety/clip_core.py:6-7). s1's own Table 4 measures the difference on math -- 2x
extension with NO string gives 50.0/93.0/55.1 (AIME24/MATH500/GPQA) and is WORSE than not
extending at all (50.0/93.0/57.6), while every injected string recovers to 59.6 on GPQA.
The string's causal job is breaking the wrap-up loop, and suppression alone does not do it.

WHICH STRING. Table 4 also settles the between-string question, in the other direction:
"Alternatively" 50.0/92.2/59.6, "Hmm" 50.0/93.0/59.6, "Wait" 53.3/93.0/59.6 -- a perfect
three-way GPQA tie, four MATH500 items, one AIME problem. So `inject_string` is a
parameter with "Wait" as the default (s1's own choice, and the only one with a safety
prior), NOT because the choice is load-bearing but because pinning it on the row is how a
later ablation stays interpretable.

THE TRIGGER IS A REAL EMISSION, NOT A GUESS. The model must actually emit </think> for an
injection to fire. The tempting single-call implementation -- mask </think> and inject
whenever it would have been the argmax -- is cheaper and WRONG under sampling: at temp 0.6
the argmax is often not what would have been drawn, so it fires on stops that would never
have happened and misses ones that would. That silently changes the intervention while
still producing exactly `thinking_budget` tokens, i.e. it looks correct in every summary.
So this generates in segments and pays for the re-prefill.

UNBATCHED, deliberately. Sequences reach their stop attempts at different steps, so a
batched version needs per-row segment state and per-row un-padding -- real work, and the
head-to-head against suppression must run at ONE batch size anyway (kernel selection
depends on batch shape, so batching one arm and not the other confounds method with batch
size). Batch this only after the comparison is done.
"""

import json

import torch

import harness as H

DEFAULT_INJECT_STRING = "Wait"

# A model that emits </think> as its very first token every time would inject forever if
# the injected string were empty; it is not, so each injection makes progress. This bound
# exists to catch the pathological case LOUDLY rather than to shape the experiment: at a
# 16384 budget and a plausible 200-token gap between stop attempts, ~80 injections is
# already extreme, and 4x that is unambiguously broken.
MAX_INJECTIONS_PER_TRACE = 512


def encode_injection(tokenizer, inject_string):
    """Token ids for the injected string, with the leading space a mid-trace continuation
    needs. add_special_tokens=False -- a BOS spliced into the middle of a think block is a
    different intervention from the one this module documents."""
    return tokenizer.encode(inject_string if inject_string.startswith((" ", "\n"))
                            else " " + inject_string, add_special_tokens=False)


@torch.no_grad()
def generate_wait_injected(model, tokenizer, prompt, thinking_budget, *,
                           inject_string=DEFAULT_INJECT_STRING, max_answer_tokens=600,
                           temperature=0.6, top_p=0.95, seed=0):
    """One trace, thinking forced to EXACTLY `thinking_budget` tokens by injection.

    Returns (text, think_tokens, margins, entropies, full_ids, injection_record) where
    injection_record is {"n_injections", "spans", "inject_string", "inject_token_ids"} and
    `spans` are [start, end) offsets of the injected tokens WITHIN the think block.

    The spans are the point of recording anything here. Phase 2 pools over the think
    window; the injected tokens are experimenter-authored text sitting inside it, and
    masking them out later is impossible unless their positions were captured at
    generation time. (Masking the positions does not remove their influence on every
    model-written token after them -- see harness.build_input_ids on why a prefill is
    refused for the same reason -- but it is the difference between a control that can be
    attempted and one that cannot.)
    """
    if int(thinking_budget) <= 0:
        raise ValueError(f"thinking_budget must be positive, got {thinking_budget!r}")
    torch.manual_seed(seed)
    think_end_id = H.get_think_end_id(tokenizer)
    eos_ids = H.get_eos_ids(model, tokenizer)
    pad_id = H.get_pad_id(model, tokenizer)
    inject_ids = encode_injection(tokenizer, inject_string)
    if not inject_ids:
        raise ValueError(f"inject_string {inject_string!r} encodes to no tokens")

    ids = H.build_input_ids(model, tokenizer, prompt)
    prompt_len = ids.shape[1]
    cur = ids
    think_len = 0                 # thinking tokens accumulated so far
    spans, n_inj = [], 0
    margins, entropies = [], []

    while think_len < thinking_budget:
        remaining = thinking_budget - think_len
        # Suppress only the model's OWN stop ids for the whole segment -- </think> is
        # deliberately left available, because catching a real emission of it is the
        # trigger. A raw eos here would end the trace without ever closing the block.
        supp = H._SuppressUntil(eos_ids, until=remaining)
        out = model.generate(
            cur, max_new_tokens=remaining, do_sample=True,
            temperature=temperature, top_p=top_p,
            logits_processor=[supp],
            stopping_criteria=[H._StopOnThinkEnd(think_end_id)],
            pad_token_id=pad_id,
            return_dict_in_generate=True, output_scores=True,
        )
        seg = out.sequences[0, cur.shape[1]:]
        m, e = H._scores_to_margin_entropy(out.scores)
        closed_early = len(seg) and int(seg[-1]) == think_end_id
        if closed_early:
            seg = seg[:-1]                     # SUPPRESS the delimiter: drop it entirely
            m, e = m[:len(seg)], e[:len(seg)]
        margins.extend(m)
        entropies.extend(e)
        think_len += int(seg.shape[0])
        cur = torch.cat([cur, seg.unsqueeze(0)], dim=1)

        if not closed_early:
            break                              # ran out the budget mid-thought; done
        if think_len >= thinking_budget:
            break                              # closed exactly at the budget; nothing to add

        # APPEND the string and resume. Clamped so an injection can never overshoot the
        # budget -- the arm's whole contract is an exact thinking-token count.
        room = thinking_budget - think_len
        add = inject_ids[:room]
        spans.append([think_len, think_len + len(add)])
        n_inj += 1
        if n_inj > MAX_INJECTIONS_PER_TRACE:
            raise RuntimeError(
                f"wait injection ran away: {n_inj} injections at budget "
                f"{thinking_budget} (think_len={think_len}). The model is closing "
                f"</think> almost immediately every time; inspect the prompt before "
                f"trusting anything from this configuration.")
        cur = torch.cat(
            [cur, torch.tensor([add], device=cur.device, dtype=cur.dtype)], dim=1)
        think_len += len(add)
        # The injected tokens were not sampled, so they have no margin/entropy. Pad with
        # None rather than skipping: the per-step lists must stay aligned to think
        # positions or `spans` indexes into the wrong place.
        margins.extend([None] * len(add))
        entropies.extend([None] * len(add))

    produced = cur.shape[1] - prompt_len
    if produced != thinking_budget:
        raise RuntimeError(
            f"wait injection produced {produced} thinking tokens, asked "
            f"{thinking_budget}. Do NOT trust thinking_tokens on this row.")

    seq1 = torch.cat(
        [cur, torch.tensor([[think_end_id]], device=cur.device, dtype=cur.dtype)], dim=1)
    out2 = model.generate(
        seq1, max_new_tokens=max_answer_tokens, do_sample=True,
        temperature=temperature, top_p=top_p, pad_token_id=pad_id,
        return_dict_in_generate=True, output_scores=True,
    )
    m2, e2 = H._scores_to_margin_entropy(out2.scores)
    full = out2.sequences
    new_tokens = full[0][prompt_len:]
    text = tokenizer.decode(new_tokens, skip_special_tokens=True)
    record = {
        "n_injections": n_inj,
        "spans": spans,
        "inject_string": inject_string,
        "inject_token_ids": inject_ids,
    }
    return (text, H._count_thinking_tokens(tokenizer, new_tokens),
            list(margins) + list(m2), list(entropies) + list(e2), full, record)


def injection_note(record):
    """The injection record as a `Trace.notes` string.

    Deliberately NOT a new schema field. harness.py and the Trace schema are FROZEN (R4:
    a change on one side must land on both), and this arm should be able to produce data
    before a schema negotiation completes. `notes` is already free-form, and a reader that
    wants the spans can json.loads them back. If the arm survives contact with a real run,
    promote it to a real field WITH team sign-off rather than leaving it here.
    """
    return json.dumps({"wait_injection": record}, separators=(",", ":"))
