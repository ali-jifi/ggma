# steering_primitives.py — reusable diff-in-means steering

A single-file, model-agnostic extraction of the reusable core of *Refusal in Language Models Is
Mediated by a Single Direction* (Arditi et al., 2024), for porting into the overthinking-vs-safety
work. Depends only on `torch` + a HuggingFace decoder-only model you pass in — **none** of the
`refusal_direction` repo's per-model/pipeline code.

## API
| Function | Purpose |
|---|---|
| `difference_in_means(model, pos_ids, pos_mask, neg_ids, neg_mask, layer, position)` | Extract a direction as mean(concept-present) − mean(concept-absent) of the residual stream. Returns the **raw** mean-diff vector. |
| `mean_activations(model, ids, mask, position)` | Mean residual activation per layer (generic, via `output_hidden_states`). |
| `projection(activation, direction)` | Signed scalar projection onto a direction (for subspace-overlap analysis). `cosine(a, b)` too. |
| `ablate_direction(model, direction)` → `(pre, post)` | Directional **ablation** at all layers/positions (remove the concept). |
| `add_direction(model, direction, layer, coeff)` → `(pre, post)` | **Activation addition** at one layer (`+1` induces, `−1` suppresses). |
| `add_hooks(pre, post)` | Context manager to apply the hooks around a forward/generate. |
| `orthogonalize_weights(model, direction)` | Bake ablation into weights permanently (paper §4; Llama/Gemma/Qwen2 layout). |

## Minimal use
```python
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
import steering_primitives as sp

MODEL = "deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B"    # any HF decoder-only model works
tok = AutoTokenizer.from_pretrained(MODEL); tok.padding_side = "left"
model = AutoModelForCausalLM.from_pretrained(MODEL, torch_dtype=torch.bfloat16).to("cuda")

def enc(instrs):
    texts = [tok.apply_chat_template([{"role":"user","content":x}], tokenize=False, add_generation_prompt=True) for x in instrs]
    e = tok(texts, return_tensors="pt", padding=True, add_special_tokens=False)
    return e.input_ids, e.attention_mask

pos_ids, pos_mask = enc(concept_present_prompts)     # e.g. harmful / "overthinking" prompts
neg_ids, neg_mask = enc(concept_absent_prompts)      # e.g. harmless / "direct" prompts
d = sp.difference_in_means(model, pos_ids, pos_mask, neg_ids, neg_mask, layer=14, position=-1)

pre, post = sp.ablate_direction(model, d)            # or sp.add_direction(model, d, layer=14, coeff=1.0)
with sp.add_hooks(pre, post):
    out = model.generate(**enc(test_prompts))
```

## Notes for the subspace study
- **Extraction is generic** (`output_hidden_states`); interventions auto-detect the block/attn/mlp
  modules for Llama/Gemma/Mistral/Qwen2/GPT-2/NeoX. For two concepts (overthinking, safety) just call
  `difference_in_means` twice and compare with `cosine` / `projection` to study subspace overlap.
- **Scale matters for addition, not ablation.** The direction is the raw mean-diff. `ablate_direction`
  and `projection` normalize internally; `add_direction` uses the raw magnitude (so `coeff=±1` matches
  the paper). Pass `normalize=True` to `difference_in_means` only if you want a unit vector for analysis.
- **`(layer, position)` is model-specific.** `position=-2` matched the paper's Gemma selection
  (second-to-last template token); the modal eval defaults to `layer=14, position=-1` for
  DeepSeek-R1-Distill-Qwen-1.5B (28 layers). Sweep per model for the best choice.

## Used by
`safety/modal_safety_test/` (the cloud eval) imports this module to extract the refusal direction
for the DeepSeek/Qwen reasoning models (`difference_in_means`) and to ablate it during generation
(`ablate_direction`).

## Verified
Checked against a reference implementation on gemma-2b-it:
`[1]` ported direction vs. reference **cosine +0.972**; `[2]` ablation flips **4/4 → 0/4** refusals;
`[3]` activation-addition (+1) turns a harmless completion into a refusal.
(The demo script and its full output are kept in local scratch, outside this repo.)
