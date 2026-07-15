"""
steering_primitives.py — model-agnostic difference-in-means steering.

A minimal, self-contained port of the reusable core of "Refusal in Language Models Is Mediated
by a Single Direction" (Arditi et al., 2024): extract a direction by difference-in-means, then
steer with it via directional ablation (remove) or activation addition (inject). Decoupled from
that repo's per-model classes — works with any HuggingFace decoder-only CausalLM.

Depends only on `torch` (+ a HF model object you pass in). No transformers import here.

Typical use for a subspace study (e.g. overthinking vs. safety):
    from steering_primitives import (
        mean_activations, difference_in_means, projection,
        ablate_direction, add_direction, add_hooks, orthogonalize_weights,
    )

    # 1) a direction per concept, from contrastive prompt sets
    d_safety = difference_in_means(model, pos_ids, pos_mask, neg_ids, neg_mask, layer=10, position=-2)

    # 2) measure how much an activation lies along it (subspace overlap work)
    p = projection(some_activation, d_safety)

    # 3) steer
    pre, post = ablate_direction(model, d_safety)              # remove it everywhere
    with add_hooks(pre, post):
        out = model.generate(**inputs)
"""
from __future__ import annotations
import contextlib
import functools
from typing import Iterable, Sequence

import torch
from torch import Tensor

__all__ = [
    "get_decoder_layers", "get_attn_module", "get_mlp_module", "model_dims",
    "mean_activations", "difference_in_means", "projection", "cosine",
    "add_hooks", "ablate_direction", "add_direction", "orthogonalize_weights",
]

# --------------------------------------------------------------------------------------
# Architecture auto-detection (covers the common decoder-only families)
# --------------------------------------------------------------------------------------
def get_decoder_layers(model) -> torch.nn.ModuleList:
    """Return the list of transformer decoder blocks for common HF architectures."""
    m = model
    for path in ("model.layers", "transformer.h", "gpt_neox.layers", "model.decoder.layers"):
        obj = m
        try:
            for attr in path.split("."):
                obj = getattr(obj, attr)
            return obj
        except AttributeError:
            continue
    raise AttributeError(
        f"Could not locate decoder layers on {type(model).__name__}. "
        "Pass the block list explicitly or extend get_decoder_layers()."
    )

def get_attn_module(layer):
    for name in ("self_attn", "attn", "attention"):
        if hasattr(layer, name):
            return getattr(layer, name)
    raise AttributeError("No attention submodule found on decoder layer.")

def get_mlp_module(layer):
    for name in ("mlp", "feed_forward", "ffn"):
        if hasattr(layer, name):
            return getattr(layer, name)
    raise AttributeError("No MLP submodule found on decoder layer.")

def model_dims(model) -> tuple[int, int]:
    """(num_layers, d_model) from the config."""
    cfg = model.config
    n = getattr(cfg, "num_hidden_layers", None) or getattr(cfg, "n_layer")
    d = getattr(cfg, "hidden_size", None) or getattr(cfg, "n_embd")
    return int(n), int(d)


# --------------------------------------------------------------------------------------
# Direction extraction (generic: uses output_hidden_states, no module hooks needed)
# --------------------------------------------------------------------------------------
@torch.no_grad()
def mean_activations(
    model,
    input_ids: Tensor,
    attention_mask: Tensor,
    position: int = -1,
    batch_size: int = 16,
) -> Tensor:
    """
    Mean residual-stream activation at one token `position`, for every layer.

    Returns a tensor of shape (num_layers + 1, d_model): index 0 is the embedding output
    (= resid_pre of layer 0), index L is the output of decoder layer L-1 (= resid_pre of layer L).
    So a "layer L" difference-in-means direction uses index L.

    Assumes LEFT padding (so `position` counts from the true end of every sequence).
    """
    device = next(model.parameters()).device
    n = input_ids.shape[0]
    acc = None
    for i in range(0, n, batch_size):
        ids = input_ids[i:i + batch_size].to(device)
        mask = attention_mask[i:i + batch_size].to(device)
        hs = model(input_ids=ids, attention_mask=mask, output_hidden_states=True).hidden_states
        # hs: tuple length (num_layers+1), each (batch, seq, d_model)
        batch = torch.stack([h[:, position, :].float().sum(dim=0) for h in hs], dim=0)  # (L+1, d)
        acc = batch if acc is None else acc + batch
    return acc / n


@torch.no_grad()
def difference_in_means(
    model,
    pos_input_ids: Tensor, pos_attention_mask: Tensor,
    neg_input_ids: Tensor, neg_attention_mask: Tensor,
    layer: int,
    position: int = -1,
    batch_size: int = 16,
    normalize: bool = False,
) -> Tensor:
    """
    Difference-in-means direction at (layer, position): mean(pos) - mean(neg) of the residual
    stream. `pos` is the concept-present set (e.g. harmful), `neg` the concept-absent set.

    Returns the RAW mean-difference vector (d_model,) by default — this is what the paper stores
    and what `add_direction` expects (activation addition is scale-dependent; coeff ±1 on the raw
    vector). `ablate_direction` and `projection` re-normalize internally, so the raw vector is the
    right common object for all three. Pass normalize=True only if you specifically want a unit vector.
    """
    mu_pos = mean_activations(model, pos_input_ids, pos_attention_mask, position, batch_size)[layer]
    mu_neg = mean_activations(model, neg_input_ids, neg_attention_mask, position, batch_size)[layer]
    d = mu_pos - mu_neg
    if normalize:
        d = d / (d.norm() + 1e-8)
    return d


def projection(activation: Tensor, direction: Tensor) -> Tensor:
    """Signed scalar projection of `activation` onto unit `direction` (last-dim contraction)."""
    dir_u = direction / (direction.norm(dim=-1, keepdim=True) + 1e-8)
    return (activation.to(dir_u) * dir_u).sum(dim=-1)


def cosine(a: Tensor, b: Tensor) -> float:
    a, b = a.flatten().float(), b.flatten().float()
    return torch.nn.functional.cosine_similarity(a, b, dim=0).item()


# --------------------------------------------------------------------------------------
# Interventions (runtime forward hooks)
# --------------------------------------------------------------------------------------
@contextlib.contextmanager
def add_hooks(pre_hooks: Sequence[tuple], hooks: Sequence[tuple]):
    """Temporarily register (module, fn) forward-pre-hooks and forward-hooks; auto-removes."""
    handles = []
    try:
        for module, fn in pre_hooks:
            handles.append(module.register_forward_pre_hook(fn))
        for module, fn in hooks:
            handles.append(module.register_forward_hook(fn))
        yield
    finally:
        for h in handles:
            h.remove()


def _proj_out(activation: Tensor, direction: Tensor) -> Tensor:
    d = (direction / (direction.norm() + 1e-8)).to(activation)
    return activation - (activation @ d).unsqueeze(-1) * d


def _ablation_pre_hook(direction):
    def fn(module, args):
        x = args[0] if isinstance(args, tuple) else args
        x = _proj_out(x, direction)
        return (x, *args[1:]) if isinstance(args, tuple) else x
    return fn

def _ablation_out_hook(direction):
    def fn(module, args, output):
        x = output[0] if isinstance(output, tuple) else output
        x = _proj_out(x, direction)
        return (x, *output[1:]) if isinstance(output, tuple) else x
    return fn

def _actadd_pre_hook(direction, coeff):
    def fn(module, args):
        x = args[0] if isinstance(args, tuple) else args
        x = x + coeff * direction.to(x)
        return (x, *args[1:]) if isinstance(args, tuple) else x
    return fn


def ablate_direction(model, direction: Tensor) -> tuple[list, list]:
    """
    Directional ablation at ALL layers/positions (paper §2.4): project `direction` out of the
    residual stream at each block input and out of each attention & MLP output. Returns
    (pre_hooks, hooks) to use with `add_hooks`.
    """
    layers = get_decoder_layers(model)
    pre = [(blk, _ablation_pre_hook(direction)) for blk in layers]
    post = [(get_attn_module(blk), _ablation_out_hook(direction)) for blk in layers]
    post += [(get_mlp_module(blk), _ablation_out_hook(direction)) for blk in layers]
    return pre, post


def add_direction(model, direction: Tensor, layer: int, coeff: float = 1.0) -> tuple[list, list]:
    """
    Activation addition (paper §2.4): add `coeff * direction` to the residual stream at the input
    of `layer`. coeff=+1 induces the concept; coeff=-1 suppresses it. Returns (pre_hooks, hooks).
    """
    layers = get_decoder_layers(model)
    return [(layers[layer], _actadd_pre_hook(direction, coeff))], []


# --------------------------------------------------------------------------------------
# Weight orthogonalization (permanent edit; paper §4) — llama-family layout
# --------------------------------------------------------------------------------------
@torch.no_grad()
def orthogonalize_weights(model, direction: Tensor) -> None:
    """
    Bake directional ablation into the weights (no runtime hooks): remove the `direction`
    component from every matrix that writes to the residual stream — token embeddings and each
    layer's attention output projection (o_proj) and MLP down projection (down_proj).

    Implemented for the Llama/Gemma/Mistral/Qwen2 layout (attn.o_proj, mlp.down_proj,
    model.embed_tokens). Raises if those attributes are absent — extend for other layouts.
    """
    d = (direction / (direction.norm() + 1e-8))

    def orth_rows(W):  # W: (out, in) writing to residual of size==len(d) on `out`
        dd = d.to(W)
        return W - torch.outer(dd, dd @ W)

    emb = model.get_input_embeddings()          # (vocab, d_model)
    dd = d.to(emb.weight)
    emb.weight.data = emb.weight.data - torch.outer(emb.weight.data @ dd, dd)

    for blk in get_decoder_layers(model):
        attn, mlp = get_attn_module(blk), get_mlp_module(blk)
        if not (hasattr(attn, "o_proj") and hasattr(mlp, "down_proj")):
            raise AttributeError(
                "orthogonalize_weights expects attn.o_proj and mlp.down_proj (Llama/Gemma layout)."
            )
        attn.o_proj.weight.data = orth_rows(attn.o_proj.weight.data)
        mlp.down_proj.weight.data = orth_rows(mlp.down_proj.weight.data)
