"""Model loading for the drivers. Isolated so torch/transformers import only
here -- the matchers and data loaders stay import-light for offline machines.

Layer 2 attention capture requires attn_implementation='eager' (the harness's
AttnToAnchorRecorder raises otherwise), so that is the default.
"""


def load_model(model_name, dtype="bfloat16", device_map="auto",
               attn_implementation="eager", revision=None):
    """Returns (model, tokenizer, meta) where meta carries the fields the Trace
    schema needs stamped at generation time."""
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    torch_dtype = getattr(torch, dtype) if isinstance(dtype, str) else dtype
    tok = AutoTokenizer.from_pretrained(model_name, revision=revision)
    model = AutoModelForCausalLM.from_pretrained(
        model_name, torch_dtype=torch_dtype, device_map=device_map,
        attn_implementation=attn_implementation, revision=revision,
    )
    model.eval()

    from harness import chat_template_hash
    meta = {
        "model": model_name,
        "model_revision": revision or getattr(model.config, "_commit_hash", "") or "",
        "chat_template_version": chat_template_hash(tok),
        # v0.3: recorded per row -- both change the numbers a run produces.
        # Read back from the loaded model, not the args, so the row reflects
        # what actually got used if transformers silently substitutes.
        "dtype": str(getattr(model, "dtype", torch_dtype)).replace("torch.", ""),
        "attn_implementation": getattr(model.config, "_attn_implementation",
                                       attn_implementation),
    }
    return model, tok, meta
