"""Harmony-format surgery for gpt-oss — the format layer the `</think>` instruments cannot provide.

gpt-oss has no `</think>` token anywhere. Reasoning lives in an `analysis` message that ends with
`<|end|>`, and the answer lives in a `final` message that opens with the four-token header
`<|start|>assistant<|channel|>final<|message|>`. Every legacy primitive assumes a SINGLE think-end
token, and `harness.get_think_end_id` does not fail loudly on this model — it falls through to
`encode("</think>")[-1]` and returns an arbitrary real token. So the surgery is rebuilt here rather
than patched there, and the legacy tree is left frozen.

Two halves:
  * pure string/id logic (no torch, no tokenizer)  -- unit-tested, and where every judgement lives
  * tokenizer helpers (no model)                   -- id resolution, prompt construction

Verified against primary sources 2026-08-21: token ids and channel structure from the OpenAI
harmony spec, the generation prompt and the `Current date` injection from the model's own
`chat_template.jinja`, architecture from `config.json`. `assert_token_ids` re-checks the ids
against the running tokenizer so a silent upstream change cannot pass unnoticed.
"""
import re

import _paths  # noqa: F401  (path bootstrap)

import drift_screen as ds

# ---------------------------------------------------------------- format constants
SPECIALS = {
    "start": "<|start|>", "end": "<|end|>", "message": "<|message|>", "channel": "<|channel|>",
    "return": "<|return|>", "call": "<|call|>", "constrain": "<|constrain|>",
}
DOCUMENTED_IDS = {
    "start": 200006, "end": 200007, "message": 200008, "channel": 200005,
    "return": 200002, "call": 200012, "constrain": 200003,
}

ANALYSIS_OPEN = "<|channel|>analysis<|message|>"
FINAL_OPEN = "<|channel|>final<|message|>"
# What must be appended to a cut analysis to force the answer. The leading <|end|> closes the
# analysis message; without it the final header would nest inside the reasoning message.
FINAL_HEADER = "<|end|><|start|>assistant" + FINAL_OPEN
GEN_PROMPT_TAIL = "<|start|>assistant"          # what the chat template ends with
STOPS = ("<|return|>", "<|call|>")              # decode-time stop tokens

# Mirrors drift_screen._SPECIAL so cleaned text is comparable with every historical run. Held as a
# copy rather than an import of a private name; test_harmony asserts the two patterns stay equal.
_SPECIAL = re.compile(r"<\|[^|]*\|>|</?think>")
_DATE_RE = re.compile(r"(Current date: )(\d{4}-\d{2}-\d{2})")
_DATE_OK = re.compile(r"^\d{4}-\d{2}-\d{2}$")


def _clean(s):
    return _SPECIAL.sub("", s).strip()


# ---------------------------------------------------------------- pure string logic
def split_trace(text):
    """(analysis, answer, closed) — the harmony analogue of drift_screen.split_trace.

    `closed` means a final message was opened, i.e. the trace reached the answer channel. A trace
    that never got there is UNSCORABLE, exactly as an unclosed `<think>` block is: never compliance,
    never safe.

    Tolerates both trace shapes we produce: one where the model emitted `<|channel|>analysis
    <|message|>` itself (natural generation, since the template stops at `<|start|>assistant`), and
    one where we prefilled that header and decoded only the new tokens. Anything the model puts in a
    `commentary` message counts as neither analysis nor answer.
    """
    closed = FINAL_OPEN in text
    head = text.rsplit(FINAL_OPEN, 1)[0] if closed else text
    analysis = head.split(ANALYSIS_OPEN, 1)[1] if ANALYSIS_OPEN in head else head
    analysis = analysis.split(SPECIALS["end"], 1)[0]
    if not closed:
        return _clean(analysis), "", False
    answer = text.rsplit(FINAL_OPEN, 1)[1]
    for stop in STOPS + (SPECIALS["end"],):
        answer = answer.split(stop, 1)[0]
    return _clean(analysis), _clean(answer), True


def pin_date(prompt, date):
    """Replace the template's `Current date: <today>` with a fixed date.

    The gpt-oss chat template injects the real current date into every system message, so the same
    seed produces a DIFFERENT prompt tomorrow and no run is reproducible across days. Raises unless
    exactly one date line was found — a silent zero-substitution would reintroduce the drift.
    """
    if not _DATE_OK.match(date or ""):
        raise ValueError(f"date must be YYYY-MM-DD, got {date!r}")
    out, n = _DATE_RE.subn(lambda m: m.group(1) + date, prompt)
    if n != 1:
        raise ValueError(f"expected exactly one 'Current date:' line in the prompt, found {n}")
    return out


def find_subsequence(seq, sub, limit=None):
    """Index of the first occurrence of `sub` in `seq`, or None. `limit` caps the search start."""
    if not sub:
        return None
    seq, sub = list(seq), list(sub)
    last = len(seq) - len(sub)
    if limit is not None:
        last = min(last, limit)
    for i in range(last + 1):
        if seq[i:i + len(sub)] == sub:
            return i
    return None


def analysis_span(gen_ids, open_ids, end_id, header_window=8):
    """(start, end) token indices bounding the analysis message BODY inside a generation.

    `start` skips the analysis header when the model emitted one (searched only in the first
    `header_window` tokens, so a later mention cannot be mistaken for the opener) and is 0 when the
    header was prefilled instead. `end` is the first `<|end|>`, or the length when the generation
    was truncated before closing. The body is what a clip cuts, so `end - start` is the reasoning
    token count — never the whole generation, which is what the legacy counter silently returns.
    """
    gen_ids = list(gen_ids)
    hit = find_subsequence(gen_ids, open_ids, limit=header_window)
    start = 0 if hit is None else hit + len(open_ids)
    for i in range(start, len(gen_ids)):
        if gen_ids[i] == end_id:
            return start, i
    return start, len(gen_ids)


def analysis_len(gen_ids, open_ids, end_id):
    s, e = analysis_span(gen_ids, open_ids, end_id)
    return e - s


# ---------------------------------------------------------------- tokenizer helpers
def token_ids(tok):
    """Resolve the harmony specials, raising if any is missing from this tokenizer."""
    out = {}
    unk = getattr(tok, "unk_token_id", None)
    for name, s in SPECIALS.items():
        i = tok.convert_tokens_to_ids(s)
        if i is None or (unk is not None and i == unk):
            raise ValueError(f"{s} is not a special token of this tokenizer — not a harmony model")
        out[name] = int(i)
    return out


def assert_token_ids(tok):
    """token_ids(), additionally checked against the published values. Returns the dict."""
    got = token_ids(tok)
    bad = {k: (v, DOCUMENTED_IDS[k]) for k, v in got.items() if v != DOCUMENTED_IDS[k]}
    if bad:
        raise ValueError(f"harmony token ids differ from the published spec (got, expected): {bad}")
    return got


def encode(tok, s):
    return list(tok(s, add_special_tokens=False).input_ids)


def analysis_open_ids(tok):
    return encode(tok, ANALYSIS_OPEN)


def final_header_ids(tok):
    return encode(tok, FINAL_HEADER)


def stop_ids(tok):
    ids = token_ids(tok)
    return [ids["return"], ids["call"]]


def build_prompt(tok, user_text, effort="high", date=None):
    """Rendered prompt string for one user turn, with the date pinned when given."""
    s = tok.apply_chat_template([{"role": "user", "content": user_text}],
                                tokenize=False, add_generation_prompt=True,
                                reasoning_effort=effort)
    return pin_date(s, date) if date else s


def build_ids(tok, user_text, effort="high", date=None, prefill_analysis=False, device=None):
    """Prompt token ids, optionally with the analysis header prefilled.

    Natural generation leaves the header to the model (the template stops at `<|start|>assistant`);
    prefilling is for continuations that must start inside the reasoning channel.
    """
    import torch
    ids = encode(tok, build_prompt(tok, user_text, effort, date))
    if prefill_analysis:
        ids = ids + analysis_open_ids(tok)
    t = torch.tensor([ids], dtype=torch.long)
    return t.to(device) if device is not None else t


__all__ = [
    "SPECIALS", "DOCUMENTED_IDS", "ANALYSIS_OPEN", "FINAL_OPEN", "FINAL_HEADER", "GEN_PROMPT_TAIL",
    "STOPS", "split_trace", "pin_date", "find_subsequence", "analysis_span", "analysis_len",
    "token_ids", "assert_token_ids", "encode", "analysis_open_ids", "final_header_ids", "stop_ids",
    "build_prompt", "build_ids", "ds",
]
