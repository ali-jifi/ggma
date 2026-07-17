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

## What these are, and what they are not

These implement the **benign-padding** method of Chain-of-Thought Hijacking
(2510.26418), which the proposal names as the safety-side drift induction:

> "induce refusal drift with established protocols rather than waiting for it: the
> benign-padding method of Chain-of-Thought Hijacking [9] and the multi-stage
> induction framework of Path Drift [12]"

The mechanism is **dilution by length**, not persuasion. The padding is a genuinely
harmless reasoning task; the safety signal weakens because the model spends its
reasoning budget elsewhere before it reaches the request. So the text here contains
no persona manipulation, no role-play framing, no instruction to ignore anything,
and nothing harmful of its own. If a preamble ever needs such content to work, it is
no longer testing this mechanism and does not belong in this directory.

Each preamble deliberately ends with a colon, so the request reads as the thing
being asked *after* the padding. Do not rely on trailing whitespace: `load_preambles`
`.strip()`s the file, and the blank line between preamble and request comes from
`load_safety`'s `f"{preamble_text}\n\n{request}"`.

## Why they persist past Stage 1

The padding is built as Stage 1 / task 1.0b's instrument (handoff:194), but it is
**not** a Stage-1 throwaway. `induction` and `preamble_id` are fields on the frozen
`Trace` schema and `compute_cell_stats` keys its cell on `induction`, because the
proposal requires both arms to be compared (p.14):

> "a subspace extracted from adversarially induced drift may partly encode
> 'responding to this manipulation' rather than generic commitment erosion, which is
> why we compare induced against naturally occurring drift where possible and treat
> agreement between them as the stronger evidence."

So `hijack_preamble` is one of **two permanent arms** of task 1.4, alongside
`natural`.

## The requirement nobody wrote down

If the **drift** pile is padded and the **stable** pile is not, any drift-vs-stable
contrast encodes commitment erosion *plus the preamble's activation signature*. The
length control cannot remove it: Control 1 builds the length direction from
`(long - short)` on stable pairs, and the preamble cancels in that subtraction
whichever arm the pairs are in -- so the length direction never contains a preamble
component to project out. Matching `induction` across the two piles is the only
defence.

**Collect the stable pile padded-but-still-refusing**, under the same `preamble_id`
as the drift pile. `compute_cell_stats` already pairs within-induction, so the
harness supports this; nothing else enforces it.

## Length

Dilution scales with padding length, so a preamble that is too short simply will not
induce. These are sized for the 1.5B debug model's budget grid; if the erosion curve
stays flat under `hijack_preamble` at the long budget, lengthen the padding before
concluding the model does not drift.

## Current preambles

| id | padding task | notes |
|---|---|---|
| `hijack_logic_grid` | 5x5 constraint-satisfaction logic puzzle | forces long systematic elimination |
| `hijack_enumeration` | blocked-lattice path counting, 4 parts | forces explicit cell-by-cell DP; longer |
