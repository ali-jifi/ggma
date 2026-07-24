"""
wait_injection.py -- induction mechanisms for the math short/long budget
conditions. Lives outside harness.py deliberately: harness.py's own header
says the engine + Trace schema are frozen, everything else (new induction
methods, matchers, scripts) is iterable and plugs in through the documented
primitives (get_think_end_id, build_input_ids, _scores_to_margin_entropy,
_count_thinking_tokens). This file does that; it does not modify harness.py.

Two functions, one per budget condition:

  generate_capped_no_injection  -- the SHORT condition. No Wait, ever. The
    budget is a CEILING only: the model tries to conclude on its own, and we
    only force a close if it's still reasoning when the ceiling hits. This
    matches s1's (Muennighoff et al., 2501.19393) case (I) budget forcing
    exactly: "if the model generates more thinking tokens than a desired
    limit, we forcefully end the thinking process." It does NOT suppress an
    early natural close the way harness.py's generate_exact_thinking does;
    that suppression-for-the-full-duration behavior is what caused the
    repetition-padding failure mode on the long side, and there is no reason
    to import that risk into the short side too.

  generate_stop_triggered_wait  -- the LONG condition. This is s1's case
    (II): "suppress the generation of the end-of-thinking token delimiter
    and instead append 'Wait'" -- but only AT THE MOMENT the model tries to
    stop, not at fixed intervals. Implemented here by giving model.generate
    a real stopping condition (eos_token_id=[think_end_id, eos_id]) so it
    halts exactly when the model tries to conclude, rather than by blocking
    logits the whole time. Capped at max_injections per trace (default 4);
    once the cap is spent, a further stop attempt is allowed to close for
    real rather than being intercepted again.

Both return the same tuple shape the Stage 1 scripts already expect, so
they're drop-in for save_trace_row's commitment_margin / entropy_trace
fields: (text, think_tokens, margins, entropies, full_ids, ...extra).
"""

import torch
import torch.nn.functional as F

from harness import get_think_end_id, build_input_ids, _count_thinking_tokens

WAIT_TEXT = "Wait, let me double-check that."


def _scores_to_margin_entropy(scores_list):
    margins, entropies = [], []
    for step_logits in scores_list:
        probs = F.softmax(step_logits[0].float(), dim=-1)
        top2 = torch.topk(probs, 2).values
        margins.append((top2[0] - top2[1]).item())
        ent = -(probs * torch.log(probs + 1e-12)).sum().item()
        entropies.append(ent)
    return margins, entropies


@torch.no_grad()
def generate_capped_no_injection(model, tokenizer, prompt, budget,
                                 max_answer_tokens=900, temperature=0.6,
                                 top_p=0.95, seed=0):
    """SHORT condition. Budget is a ceiling, not a forced exact length.
    Returns (text, think_tokens, margins, entropies, full_ids, closed_naturally).
    closed_naturally=False means the ceiling was hit before the model tried
    to conclude on its own -- worth keeping on the trace row so we can tell
    "the model's real quick answer" apart from "we cut it off mid-thought."
    """
    torch.manual_seed(seed)
    think_end_id = get_think_end_id(tokenizer)
    eos_id = tokenizer.eos_token_id
    ids = build_input_ids(model, tokenizer, prompt)
    prompt_len = ids.shape[1]

    out1 = model.generate(
        ids, max_new_tokens=budget,
        do_sample=True, temperature=temperature, top_p=top_p,
        eos_token_id=[think_end_id, eos_id], pad_token_id=eos_id,
        return_dict_in_generate=True, output_scores=True,
    )
    seq = out1.sequences
    last_token = seq[0, -1].item()
    closed_naturally = (last_token == think_end_id)

    if not closed_naturally:
        # Still reasoning when the ceiling hit (the common case for a short
        # budget on a nontrivial problem), or it emitted raw EOS without
        # ever closing </think> (rare, treat the same way): force the close.
        if last_token == eos_id:
            seq = seq[:, :-1]
        seq = torch.cat(
            [seq, torch.tensor([[think_end_id]], device=seq.device)], dim=1)

    out2 = model.generate(
        seq, max_new_tokens=max_answer_tokens,
        do_sample=True, temperature=temperature, top_p=top_p,
        pad_token_id=eos_id,
        return_dict_in_generate=True, output_scores=True,
    )
    full = out2.sequences
    new_tokens = full[0][prompt_len:]
    text = tokenizer.decode(new_tokens, skip_special_tokens=True)
    margins, entropies = _scores_to_margin_entropy(list(out1.scores) + list(out2.scores))
    think_tokens = _count_thinking_tokens(tokenizer, new_tokens)
    return text, think_tokens, margins, entropies, full, closed_naturally


@torch.no_grad()
def generate_stop_triggered_wait(model, tokenizer, prompt, long_budget,
                                 max_injections=4, max_answer_tokens=900,
                                 temperature=0.6, top_p=0.95, seed=0):
    """LONG condition. Only intercepts a GENUINE stop attempt; never
    interrupts the model mid-thought. Returns (text, think_tokens, margins,
    entropies, full_ids, n_injections, injection_offsets).
    """
    torch.manual_seed(seed)
    think_end_id = get_think_end_id(tokenizer)
    eos_id = tokenizer.eos_token_id
    wait_ids = tokenizer.encode(WAIT_TEXT, add_special_tokens=False)

    ids = build_input_ids(model, tokenizer, prompt)
    prompt_len = ids.shape[1]

    seq = ids
    think_tokens_used = 0
    n_injections = 0
    injection_offsets = []
    all_scores = []

    while True:
        remaining = long_budget - think_tokens_used
        if remaining <= 0:
            seq = torch.cat(
                [seq, torch.tensor([[think_end_id]], device=seq.device)], dim=1)
            break

        out = model.generate(
            seq, max_new_tokens=remaining,
            do_sample=True, temperature=temperature, top_p=top_p,
            eos_token_id=[think_end_id, eos_id], pad_token_id=eos_id,
            return_dict_in_generate=True, output_scores=True,
        )
        all_scores.extend(out.scores)
        new_len = out.sequences.shape[1] - seq.shape[1]
        think_tokens_used += new_len
        seq = out.sequences
        last_token = seq[0, -1].item()

        if last_token not in (think_end_id, eos_id):
            # Ran out of its allotted budget without trying to stop -- still
            # actively working. Since `remaining` was already the FULL
            # remaining budget, this means the overall ceiling is now
            # exhausted; force the close. This is the only case that forces
            # a close on a still-reasoning trace, and it only happens at the
            # true long-budget ceiling, never mid-thought.
            seq = torch.cat(
                [seq, torch.tensor([[think_end_id]], device=seq.device)], dim=1)
            break

        # genuine stop attempt (or a raw EOS, handled the same way)
        if n_injections >= max_injections:
            if last_token != think_end_id:
                seq = torch.cat(
                    [seq[:, :-1], torch.tensor([[think_end_id]], device=seq.device)], dim=1)
            break  # cap spent: let it close for real

        # inject: strip the stop token just emitted, append "Wait...", continue
        n_injections += 1
        injection_offsets.append(think_tokens_used - 1)
        seq = seq[:, :-1]
        wait_tensor = torch.tensor([wait_ids], device=seq.device)
        seq = torch.cat([seq, wait_tensor], dim=1)
        think_tokens_used += len(wait_ids)

    out2 = model.generate(
        seq, max_new_tokens=max_answer_tokens,
        do_sample=True, temperature=temperature, top_p=top_p,
        pad_token_id=eos_id,
        return_dict_in_generate=True, output_scores=True,
    )
    all_scores.extend(out2.scores)
    full = out2.sequences
    new_tokens = full[0][prompt_len:]
    text = tokenizer.decode(new_tokens, skip_special_tokens=True)
    margins, entropies = _scores_to_margin_entropy(all_scores)
    think_tokens = _count_thinking_tokens(tokenizer, new_tokens)
    return text, think_tokens, margins, entropies, full, n_injections, injection_offsets