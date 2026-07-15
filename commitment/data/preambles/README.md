# Hijack preambles

One `.txt` file per preamble. The filename stem is the `preamble_id` recorded
on every safety trace (`Trace.preamble_id`) and used to build
`hijack_preamble` prompts.

The preamble text is prepended to the bare HarmBench request:

    <preamble text>

    <harmful request>

Per Rule **R6**, the anchor span is the harmful request **only** -- the
preamble is deliberately excluded from `anchor_text`, so
`harness.find_prompt_anchor_span` locates just the request inside the full
sequence. Add real preamble texts here; keep this README out of the way (only
`*.txt` files are loaded).
