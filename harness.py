"""
harness.py (v2) -- shared toolkit for the Two Drifts project. One copy, both sides.

Supersedes v1. Changes:
  - Trace schema upgraded to the full seven-group v2 spec (flat, single class).
  - Layer 1 commitment: truncation + free rollout (THE commitment definition).
  - Layer 2 commitment: attention-to-anchor proxy + calibration against Layer 1.
  - Margin demoted to secondary proxy (kept, never the definition, never an
    extraction regression target).
  - flip fields are CELL-level (computed across resamples), not per-row labels.
  - Three-way split: extraction | selection | heldout.
  - Multi-layer steering with recorded vector norms.

Model-agnostic across Llama- and Qwen2-family HF models (layers at
model.model.layers[i]). Layer 2 attention capture requires the model to be
loaded with attn_implementation="eager".

RULES ENCODED HERE (do not weaken without team sign-off):
  R1. rollout_outcomes is the commitment definition. attn_to_anchor and
      commitment_margin are proxies calibrated against it.
  R2. A steered row whose proxy was not re-calibrated under steering
      (calibrated_under_steering=False) has an UNUSABLE proxy.
  R3. split is assigned at generation time and never re-tagged.
  R4. Any schema field added on one side must be added on both.
"""

import hashlib
import json
import re
import time
from dataclasses import dataclass, field, asdict
from typing import Optional, Callable

import numpy as np
import torch
import torch.nn.functional as F
from transformers import LogitsProcessor

HARNESS_VERSION = "0.2.0"
SCHEMA_VERSION = "trace-v2"

VALID_SPLITS = ("extraction", "selection", "heldout")
VALID_INDUCTION = ("natural", "hijack_preamble")
VALID_ANCHOR_POLICY = ("first_anchor", "most_recent_anchor")
VALID_ATTN_NORM = ("raw_mass", "per_token_mean")


# =============================================================================
# GROUP-COMPLETE TRACE SCHEMA (v2). One flat class. Do not nest, do not fork.
# One row = one generation. Rollout completions live inside their parent row.
# =============================================================================

@dataclass
class Trace:
    # ---- GROUP 1: surface / behavioral (every stage) ----
    trace_id: str
    domain: str                       # "math" | "safety"
    model: str
    model_revision: str               # HF commit hash
    dataset: str
    prompt_id: str
    prompt_text: str
    budget_condition: str             # "natural" | "256" | ... | "16384"
    thinking_tokens: int
    response_text: str
    final_answer: Optional[str]       # math: number | safety: "refuse"|"comply"
    outcome: str                      # math: correct|incorrect ; safety: refused|complied
    gold: str
    scorer_version: str
    decode_temp: float
    decode_top_p: float
    seed: int
    wall_clock_sec: float
    harness_version: str = HARNESS_VERSION
    schema_version: str = SCHEMA_VERSION
    chat_template_version: str = ""
    induction: str = "natural"        # math side: always "natural"
    preamble_id: Optional[str] = None
    scorer_confidence: Optional[float] = None
    sample_idx: int = 0               # which resample of this cell
    n_samples_per_cell: int = 1

    # ---- GROUP 2: flip / pairing. CELL-level, stamped onto rows by
    #      compute_cell_stats(). Never hand-set from a single run at temp 0.6.
    outcome_rate_short: Optional[float] = None
    outcome_rate_long: Optional[float] = None
    flip_type: str = "none"           # "none" | "drift" | "reverse"
    is_degradation: bool = False

    # ---- GROUP 3: commitment, Layer 1 (rollout ground truth). THE definition.
    rollout_prefix_positions: list = field(default_factory=list)  # absolute token idx, inside think block
    n_rollouts_per_prefix: int = 0
    rollout_outcomes: list = field(default_factory=list)  # per prefix: fraction reaching trace's final conclusion
    rollout_completion_texts: list = field(default_factory=list)  # optional, list[list[str]]
    rollout_decode_temp: Optional[float] = None
    rollout_decode_top_p: Optional[float] = None
    rollout_seed: Optional[int] = None

    # ---- GROUP 4: commitment, Layer 2 (cheap proxies) ----
    anchor_spans: list = field(default_factory=list)      # [[start, end), ...] absolute token idx
    anchor_policy: Optional[str] = None                   # pre-registered
    attn_norm_method: Optional[str] = None                # pre-registered
    attn_to_anchor: list = field(default_factory=list)    # one value per generated-token position
    attn_layers_used: list = field(default_factory=list)
    calibration_id: Optional[str] = None
    calibrated_under_steering: bool = False               # R2
    commitment_margin: list = field(default_factory=list)  # SECONDARY proxy only
    entropy_trace: list = field(default_factory=list)
    shape: str = "unlabeled"          # "gradual" | "cliff" | "unlabeled"

    # ---- GROUP 5: activations. Tensors live in a sidecar npz (see io);
    #      the row stores the pointer + indices. (Flagged deviation from the
    #      doc wording: tensors in JSONL are unloadable.)
    activations_path: Optional[str] = None
    layers_recorded: list = field(default_factory=list)
    think_start_idx: int = -1
    think_end_idx: int = -1
    answer_pos_idx: int = -1

    # ---- GROUP 6: steering (null on unsteered rows) ----
    steering_condition: str = "none"  # "none" | "ablate" | "add"
    steering_subspace_id: Optional[str] = None
    steering_layers: list = field(default_factory=list)
    steering_coeff: Optional[float] = None
    steering_vector_norm: Optional[float] = None

    # ---- GROUP 7: bookkeeping ----
    label: str = "unlabeled"
    split: str = "extraction"         # three-way; immutable after generation (R3)
    notes: str = ""

    def to_json(self):
        return asdict(self)

    def cell_key(self):
        """The resampling cell this row belongs to."""
        return (self.prompt_id, self.budget_condition, self.induction,
                self.steering_condition, self.steering_subspace_id)


def validate_trace(t: Trace, stage: str = "surface"):
    """Fail loudly on schema violations. stage: surface|metric|steered."""
    errs = []
    if t.split not in VALID_SPLITS:
        errs.append(f"split must be one of {VALID_SPLITS}, got {t.split!r}")
    if t.induction not in VALID_INDUCTION:
        errs.append(f"bad induction {t.induction!r}")
    if t.domain == "math" and t.induction != "natural":
        errs.append("math rows must have induction='natural'")
    if stage in ("metric", "steered"):
        if not t.rollout_prefix_positions or not t.rollout_outcomes:
            errs.append("metric stage requires Layer 1 rollout fields (R1)")
        if len(t.rollout_prefix_positions) != len(t.rollout_outcomes):
            errs.append("rollout positions/outcomes length mismatch")
        if t.attn_to_anchor:
            if t.anchor_policy not in VALID_ANCHOR_POLICY:
                errs.append("attn_to_anchor present without valid anchor_policy")
            if t.attn_norm_method not in VALID_ATTN_NORM:
                errs.append("attn_to_anchor present without valid attn_norm_method")
    if stage == "steered":
        if t.steering_condition == "none":
            errs.append("steered stage but steering_condition='none'")
        if t.attn_to_anchor and not t.calibrated_under_steering:
            errs.append("R2 violation: steered row carries an uncalibrated proxy")
    if errs:
        raise ValueError(f"Trace {t.trace_id} invalid: " + "; ".join(errs))
    return True


# =============================================================================
# TOKEN HELPERS
# =============================================================================

def get_think_end_id(tokenizer):
    tid = tokenizer.convert_tokens_to_ids("</think>")
    if tid is None or tid == tokenizer.unk_token_id:
        ids = tokenizer.encode("</think>", add_special_tokens=False)
        tid = ids[-1]
    return tid


def build_input_ids(model, tokenizer, prompt):
    messages = [{"role": "user", "content": prompt}]
    ids = tokenizer.apply_chat_template(
        messages, add_generation_prompt=True, return_tensors="pt"
    ).to(model.device)
    return ids


def chat_template_hash(tokenizer):
    tpl = getattr(tokenizer, "chat_template", "") or ""
    return hashlib.sha256(tpl.encode()).hexdigest()[:12]


def _count_thinking_tokens(tokenizer, token_ids):
    think_end_id = get_think_end_id(tokenizer)
    ids = token_ids.tolist() if hasattr(token_ids, "tolist") else list(token_ids)
    if think_end_id in ids:
        return ids.index(think_end_id)
    return len(ids)


def find_think_indices(tokenizer, full_ids, prompt_len):
    think_end_id = get_think_end_id(tokenizer)
    ids = full_ids[0].tolist()
    think_start = prompt_len
    if think_end_id in ids[prompt_len:]:
        think_end = ids.index(think_end_id, prompt_len)
    else:
        think_end = len(ids)
    return think_start, think_end


# =============================================================================
# SECONDARY PROXY: margin + entropy (kept from v1; never the definition)
# =============================================================================

def _scores_to_margin_entropy(scores_tuple):
    margins, entropies = [], []
    for step_logits in scores_tuple:
        probs = F.softmax(step_logits[0].float(), dim=-1)
        top2 = torch.topk(probs, 2).values
        margins.append((top2[0] - top2[1]).item())
        ent = -(probs * torch.log(probs + 1e-12)).sum().item()
        entropies.append(ent)
    return margins, entropies


# =============================================================================
# GENERATION (unchanged mechanics from v1)
# =============================================================================

class _SuppressUntil(LogitsProcessor):
    def __init__(self, token_ids, until):
        self.token_ids = token_ids
        self.until = until
        self.step = 0

    def __call__(self, input_ids, scores):
        if self.step < self.until:
            for t in self.token_ids:
                scores[:, t] = -float("inf")
        self.step += 1
        return scores


@torch.no_grad()
def generate_natural(model, tokenizer, prompt, max_new_tokens=16384,
                     temperature=0.6, top_p=0.95, seed=0):
    torch.manual_seed(seed)
    ids = build_input_ids(model, tokenizer, prompt)
    prompt_len = ids.shape[1]
    out = model.generate(
        ids, max_new_tokens=max_new_tokens,
        do_sample=True, temperature=temperature, top_p=top_p,
        pad_token_id=tokenizer.eos_token_id,
        return_dict_in_generate=True, output_scores=True,
    )
    seq = out.sequences[0]
    new_tokens = seq[prompt_len:]
    text = tokenizer.decode(new_tokens, skip_special_tokens=True)
    margins, entropies = _scores_to_margin_entropy(out.scores)
    think_tokens = _count_thinking_tokens(tokenizer, new_tokens)
    return text, think_tokens, margins, entropies, out.sequences


@torch.no_grad()
def generate_exact_thinking(model, tokenizer, prompt, thinking_budget,
                            max_answer_tokens=600, temperature=0.6, top_p=0.95,
                            seed=0):
    torch.manual_seed(seed)
    think_end_id = get_think_end_id(tokenizer)
    eos_id = tokenizer.eos_token_id
    ids = build_input_ids(model, tokenizer, prompt)
    prompt_len = ids.shape[1]

    supp = _SuppressUntil([think_end_id, eos_id], until=thinking_budget)
    out1 = model.generate(
        ids, max_new_tokens=thinking_budget,
        do_sample=True, temperature=temperature, top_p=top_p,
        logits_processor=[supp], pad_token_id=eos_id,
        return_dict_in_generate=True, output_scores=True,
    )
    seq1 = torch.cat(
        [out1.sequences, torch.tensor([[think_end_id]], device=model.device)], dim=1
    )
    out2 = model.generate(
        seq1, max_new_tokens=max_answer_tokens,
        do_sample=True, temperature=temperature, top_p=top_p,
        pad_token_id=eos_id,
        return_dict_in_generate=True, output_scores=True,
    )
    full = out2.sequences
    new_tokens = full[0][prompt_len:]
    text = tokenizer.decode(new_tokens, skip_special_tokens=True)
    m1, e1 = _scores_to_margin_entropy(out1.scores)
    m2, e2 = _scores_to_margin_entropy(out2.scores)
    think_tokens = _count_thinking_tokens(tokenizer, new_tokens)
    return text, think_tokens, list(m1) + list(m2), list(e1) + list(e2), full


# =============================================================================
# GRADING (GSM8K; unchanged from v1)
# =============================================================================

def extract_gsm8k_gold(answer_field):
    m = re.search(r"####\s*([-\d,\.]+)", answer_field)
    return m.group(1).replace(",", "").strip() if m else answer_field.strip()


def extract_pred_answer(response_text):
    if "</think>" not in response_text:
        return None
    after = response_text.split("</think>")[-1]
    m = re.search(r"\\boxed\{([^}]*)\}", after)
    if m:
        return _norm_num(m.group(1))
    m = re.search(r"answer\s*is\s*\$?([-\d,\.]+)", after, re.IGNORECASE)
    if m:
        return _norm_num(m.group(1))
    nums = re.findall(r"-?\d[\d,]*\.?\d*", after)
    return _norm_num(nums[-1]) if nums else None


def _norm_num(s):
    s = s.replace(",", "").replace("$", "").strip()
    try:
        f = float(s)
        return str(int(f)) if f == int(f) else str(f)
    except ValueError:
        return s


def is_correct(pred, gold):
    if pred is None:
        return False
    return _norm_num(pred) == _norm_num(gold)


def math_conclusion_matcher(final_answer):
    """Returns fn(completion_text) -> bool: did this rollout reach the trace's
    final conclusion? Math side. Safety side plugs in the refusal classifier
    behind the same signature; nothing else changes (R4)."""
    def match(text):
        pred = extract_pred_answer(text)
        return pred is not None and _norm_num(pred) == _norm_num(final_answer)
    return match


# =============================================================================
# LAYER 1: TRUNCATION + FREE ROLLOUT. The commitment definition.
# =============================================================================

_SENT_BOUNDARY = re.compile(r"[.!?]\s|\n")


def choose_prefix_positions(tokenizer, full_ids, think_start_idx, think_end_idx,
                            policy="sentence", max_prefixes=8):
    """
    Pick truncation points inside the think block. Two policies (pre-register one):
      "sentence": token positions just after sentence boundaries (Thought
                  Anchors comparable). Subsampled evenly to <= max_prefixes.
      "even":     max_prefixes evenly spaced token positions.
    Returns a sorted list of ABSOLUTE token indices in the full sequence.
    """
    lo, hi = think_start_idx + 1, think_end_idx - 1
    if hi <= lo:
        return []
    if policy == "even":
        pts = np.linspace(lo, hi, max_prefixes + 2)[1:-1]
        return sorted(set(int(p) for p in pts))
    # sentence policy: decode incrementally, mark token idx after each boundary
    ids = full_ids[0].tolist()
    positions, buf = [], ""
    for i in range(lo, hi):
        buf += tokenizer.decode([ids[i]], skip_special_tokens=True)
        if _SENT_BOUNDARY.search(buf):
            positions.append(i + 1)
            buf = ""
    if not positions:
        return choose_prefix_positions(tokenizer, full_ids, think_start_idx,
                                       think_end_idx, "even", max_prefixes)
    if len(positions) > max_prefixes:
        idx = np.linspace(0, len(positions) - 1, max_prefixes).astype(int)
        positions = [positions[i] for i in idx]
    return positions


@torch.no_grad()
def rollout_commitment_curve(model, tokenizer, full_ids, prefix_positions,
                             conclusion_match_fn: Callable[[str], bool],
                             n_rollouts=8, temperature=0.6, top_p=0.95,
                             base_seed=0, max_new_tokens=4096,
                             store_texts=False):
    """
    Layer 1. For each prefix position p: continue freely n_rollouts times from
    full_ids[:p], score each completion with conclusion_match_fn, and record
    the fraction that reach the trace's final conclusion.

    Returns (curve, texts):
      curve: list of floats, one per prefix (the commitment curve)
      texts: list[list[str]] if store_texts else []
    Cost: len(prefix_positions) * n_rollouts generations. Batched via
    num_return_sequences per prefix.
    """
    curve, all_texts = [], []
    eos_id = tokenizer.eos_token_id
    for j, p in enumerate(prefix_positions):
        torch.manual_seed(base_seed + 1009 * j)
        prefix = full_ids[:, :p]
        out = model.generate(
            prefix, max_new_tokens=max_new_tokens,
            do_sample=True, temperature=temperature, top_p=top_p,
            num_return_sequences=n_rollouts, pad_token_id=eos_id,
        )
        texts = [
            tokenizer.decode(out[k][p:], skip_special_tokens=False)
            for k in range(n_rollouts)
        ]
        hits = sum(1 for t in texts if conclusion_match_fn(_ensure_closed(t)))
        curve.append(hits / n_rollouts)
        if store_texts:
            all_texts.append(texts)
    return curve, all_texts


def _ensure_closed(text):
    """Rollouts from mid-think may or may not emit </think>; the math answer
    extractor requires it. If absent, the rollout never committed: leave as-is
    so the matcher returns False."""
    return text


# =============================================================================
# LAYER 2: ATTENTION-TO-ANCHOR. Cheap proxy; calibrate against Layer 1.
# =============================================================================

def find_prompt_anchor_span(tokenizer, prompt_text, full_ids):
    """Locate the prompt-content tokens (the question / harmful request) inside
    the full sequence by token-subsequence search. Returns [start, end) or None.
    For hijack-preamble safety prompts, pass ONLY the harmful-request substring
    as prompt_text so the anchor excludes the preamble."""
    hay = full_ids[0].tolist()
    needle = tokenizer.encode(prompt_text, add_special_tokens=False)
    for trim_l in range(2):
        for trim_r in range(2):
            n = needle[trim_l:len(needle) - trim_r or None]
            if not n:
                continue
            for i in range(len(hay) - len(n) + 1):
                if hay[i:i + len(n)] == n:
                    return [i, i + len(n)]
    return None


class AttnToAnchorRecorder:
    """
    Wraps self_attn.forward at the given layers to force per-layer attention
    weights (eager only), reduces each layer's matrix to attention mass over
    the anchor spans immediately, and discards the full matrix. Peak extra
    memory = one layer's [heads, seq, seq] matrix.

    Practical ceiling: ~6-8k tokens on an 80GB A100. Longer traces need a
    chunked QK recompute (TODO, tracked); calibrate the proxy on <=4096
    budgets first.
    """

    def __init__(self, model, layers, anchor_spans, seq_len):
        self.model = model
        self.layers = list(layers)
        self.spans = anchor_spans
        self.mask = torch.zeros(seq_len, dtype=torch.bool)
        for s, e in anchor_spans:
            self.mask[s:e] = True
        self.per_span_mass = {}   # layer -> [n_spans, seq] raw mass, heads-averaged
        self._originals = {}

    def __enter__(self):
        impl = getattr(self.model.config, "_attn_implementation", "eager")
        if impl != "eager":
            raise RuntimeError(
                "Layer 2 capture requires attn_implementation='eager' "
                f"(model has {impl!r}). Reload the model accordingly."
            )
        for li in self.layers:
            mod = self.model.model.layers[li].self_attn
            self._originals[li] = mod.forward
            mod.forward = self._make_wrapper(mod.forward, li)
        return self

    def _make_wrapper(self, orig_forward, layer_idx):
        spans = self.spans

        def wrapped(*args, **kwargs):
            kwargs["output_attentions"] = True
            out = orig_forward(*args, **kwargs)
            attn = None
            if isinstance(out, tuple):
                for item in out[1:]:
                    if torch.is_tensor(item) and item.dim() == 4:
                        attn = item
                        break
            if attn is not None:
                # attn: [batch, heads, q_len, k_len], row-normalized softmax.
                a = attn[0].mean(dim=0)  # heads-averaged -> [q_len, k_len]
                masses = [a[:, s:e].sum(dim=-1) for s, e in spans]
                self.per_span_mass[layer_idx] = (
                    torch.stack(masses).float().cpu()  # [n_spans, q_len]
                )
                del a
            return out
        return wrapped

    def __exit__(self, *a):
        for li, fwd in self._originals.items():
            self.model.model.layers[li].self_attn.forward = fwd


@torch.no_grad()
def compute_attn_to_anchor(model, full_ids, layers, anchor_spans):
    """One teacher-forced forward pass; returns {layer: tensor[n_spans, seq]}
    of raw heads-averaged attention mass onto each anchor span."""
    seq_len = full_ids.shape[1]
    with AttnToAnchorRecorder(model, layers, anchor_spans, seq_len) as rec:
        model(full_ids)
    return rec.per_span_mass


def reduce_anchor_mass(per_span_mass_layer, anchor_spans, policy, norm_method,
                       gen_start_idx):
    """
    Reduce a [n_spans, seq] raw-mass tensor to the schema's attn_to_anchor
    array (one value per GENERATED token position), under the pre-registered
    anchor policy and normalization.
      policy: "first_anchor" | "most_recent_anchor"
      norm_method: "raw_mass" (softmax mass, already length-comparable as a
        fraction of attention) | "per_token_mean" (mass / span length; makes
        differing span lengths comparable across math and safety anchors)
    """
    m = per_span_mass_layer.clone()  # [n_spans, seq]
    if norm_method == "per_token_mean":
        lens = torch.tensor([e - s for s, e in anchor_spans], dtype=torch.float32)
        m = m / lens[:, None]
    elif norm_method != "raw_mass":
        raise ValueError(norm_method)

    if policy == "first_anchor":
        vec = m[0]
    elif policy == "most_recent_anchor":
        # at query position q, use the last span whose end <= q; fall back to first
        seq = m.shape[1]
        vec = torch.empty(seq)
        for q in range(seq):
            k = 0
            for i, (s, e) in enumerate(anchor_spans):
                if e <= q:
                    k = i
            vec[q] = m[k, q]
    else:
        raise ValueError(policy)
    return vec[gen_start_idx:].tolist()


# =============================================================================
# CALIBRATION: map Layer 2 proxy -> Layer 1 rollout probability.
# Isotonic (pool adjacent violators): monotone, no functional-form assumption,
# no sklearn dependency. Fit on the pilot/selection split, NEVER on heldout.
# Re-fit under each steering condition before trusting the proxy there (R2).
# =============================================================================

def fit_isotonic(x, y):
    """Increasing isotonic regression. Returns (x_sorted, y_hat) usable with
    apply_isotonic. x: proxy values at prefix positions; y: rollout fractions."""
    order = np.argsort(x)
    xs, ys = np.asarray(x, float)[order], np.asarray(y, float)[order]
    yhat, w = ys.copy(), np.ones_like(ys)
    i = 0
    blocks = [[yhat[k], w[k], k, k] for k in range(len(yhat))]
    merged = []
    for b in blocks:
        merged.append(b)
        while len(merged) > 1 and merged[-2][0] > merged[-1][0]:
            v2, w2, s2, e2 = merged.pop()
            v1, w1, s1, e1 = merged.pop()
            merged.append([(v1 * w1 + v2 * w2) / (w1 + w2), w1 + w2, s1, e2])
    out = np.empty_like(yhat)
    for v, _, s, e in merged:
        out[s:e + 1] = v
    return xs, out


def apply_isotonic(fit, x_new):
    xs, yhat = fit
    return np.interp(np.asarray(x_new, float), xs, yhat)


def make_calibration_id(domain, layer, anchor_policy, norm_method,
                        steering_condition, n_points):
    payload = f"{domain}|{layer}|{anchor_policy}|{norm_method}|{steering_condition}|{n_points}|{HARNESS_VERSION}"
    return "cal_" + hashlib.sha256(payload.encode()).hexdigest()[:10]


def save_calibration(fit, meta, path):
    xs, yhat = fit
    with open(path, "w") as f:
        json.dump({"meta": meta, "x": xs.tolist(), "y": yhat.tolist()}, f)


def load_calibration(path):
    with open(path) as f:
        d = json.load(f)
    return (np.asarray(d["x"]), np.asarray(d["y"])), d["meta"]


# =============================================================================
# CELL-LEVEL FLIP STATS (Group 2). Single-run flip labels are invalid at
# temp 0.6; compute per (prompt, budget, induction, steering) cell.
# =============================================================================

GOOD_OUTCOME = {"math": "correct", "safety": "refused"}


def compute_cell_stats(rows, short_budget, long_budget,
                       thr_hi=0.75, thr_lo=0.25):
    """
    rows: list of Trace dicts (same domain). Groups by prompt/induction/steering,
    computes P(good outcome) at short and long budgets across resamples, and a
    thresholded cell-level flip label:
      drift   : rate_short >= thr_hi and rate_long <= thr_lo
      reverse : rate_short <= thr_lo and rate_long >= thr_hi
    Thresholds are pre-registered; do not tune post hoc.
    Returns {cell_key_without_budget: {"rate_short", "rate_long", "flip_type"}}
    and stamps the values back onto the row dicts in place.
    """
    from collections import defaultdict
    cells = defaultdict(lambda: {"short": [], "long": []})
    for r in rows:
        key = (r["prompt_id"], r["induction"], r["steering_condition"],
               r["steering_subspace_id"])
        good = GOOD_OUTCOME[r["domain"]]
        if str(r["budget_condition"]) == str(short_budget):
            cells[key]["short"].append(r["outcome"] == good)
        elif str(r["budget_condition"]) == str(long_budget):
            cells[key]["long"].append(r["outcome"] == good)
    stats = {}
    for key, d in cells.items():
        if not d["short"] or not d["long"]:
            continue
        rs, rl = float(np.mean(d["short"])), float(np.mean(d["long"]))
        if rs >= thr_hi and rl <= thr_lo:
            ft = "drift"
        elif rs <= thr_lo and rl >= thr_hi:
            ft = "reverse"
        else:
            ft = "none"
        stats[key] = {"rate_short": rs, "rate_long": rl, "flip_type": ft}
    for r in rows:
        key = (r["prompt_id"], r["induction"], r["steering_condition"],
               r["steering_subspace_id"])
        if key in stats:
            r["outcome_rate_short"] = stats[key]["rate_short"]
            r["outcome_rate_long"] = stats[key]["rate_long"]
            r["flip_type"] = stats[key]["flip_type"]
    return stats


# =============================================================================
# ACTIVATION CAPTURE (unchanged from v1)
# =============================================================================

@torch.no_grad()
def capture_activations(model, full_ids, layers):
    acts, handles = {}, []

    def make_hook(idx):
        def hook(module, inp, out):
            h = out[0] if isinstance(out, tuple) else out
            acts[idx] = h.detach()[0].to(torch.float32).cpu()
        return hook

    for idx in layers:
        handles.append(model.model.layers[idx].register_forward_hook(make_hook(idx)))
    try:
        model(full_ids)
    finally:
        for h in handles:
            h.remove()
    return acts


def pool_thinking(act_layer, think_start_idx, think_end_idx):
    if think_end_idx <= think_start_idx:
        return act_layer.mean(dim=0)
    return act_layer[think_start_idx:think_end_idx].mean(dim=0)


# =============================================================================
# STEERING. Multi-layer by default (schema Group 6); records vector norms.
# =============================================================================

class Steerer:
    """Single direction, single layer (kept for compatibility)."""

    def __init__(self, model, layer, vector, mode="add", strength=1.0):
        self.model, self.layer, self.mode, self.strength = model, layer, mode, strength
        v = vector.float()
        self.raw_norm = float(v.norm())
        self.v = v / (v.norm() + 1e-8)
        self.handle = None

    def _hook(self, module, inp, out):
        h = out[0] if isinstance(out, tuple) else out
        v = self.v.to(h.device, h.dtype)
        if self.mode == "add":
            h = h + self.strength * v
        elif self.mode == "ablate":
            h = h - (h @ v).unsqueeze(-1) * v
        return (h,) + out[1:] if isinstance(out, tuple) else h

    def __enter__(self):
        self.handle = self.model.model.layers[self.layer].register_forward_hook(self._hook)
        return self

    def __exit__(self, *a):
        if self.handle is not None:
            self.handle.remove()


class SubspaceSteerer:
    """A set of directions applied at multiple layers at once. Records norms so
    control arms can be norm-matched from the record, not recomputed."""

    def __init__(self, model, layers, vectors, mode="ablate", strength=1.0):
        self.steerers = [Steerer(model, l, v, mode, strength)
                         for l in layers for v in vectors]
        self.vector_norms = [s.raw_norm for s in self.steerers[:len(vectors)]]

    def __enter__(self):
        for s in self.steerers:
            s.__enter__()
        return self

    def __exit__(self, *a):
        for s in self.steerers:
            s.__exit__(*a)


# =============================================================================
# IO
# =============================================================================

def save_trace_row(trace: Trace, path, stage="surface"):
    validate_trace(trace, stage)
    with open(path, "a") as f:
        f.write(json.dumps(trace.to_json()) + "\n")


def load_traces(path):
    with open(path) as f:
        return [json.loads(line) for line in f]


def save_activations(trace_id, acts, path):
    """Sidecar npz for one trace: {layer: tensor} -> path. Row stores path."""
    np.savez_compressed(path, **{str(l): a.numpy() for l, a in acts.items()})
    return path


def save_pooled(pooled_dict, path):
    flat = {}
    for tid, layers in pooled_dict.items():
        for l, vec in layers.items():
            flat[f"{tid}::{l}"] = np.asarray(vec)
    np.savez_compressed(path, **flat)


def load_pooled(path):
    data = np.load(path)
    out = {}
    for key in data.files:
        tid, l = key.split("::")
        out.setdefault(tid, {})[int(l)] = data[key]
    return out
