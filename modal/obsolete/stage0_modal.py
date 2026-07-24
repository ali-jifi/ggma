import modal

MODEL_NAME = "deepseek-ai/DeepSeek-R1-Distill-Llama-8B"
CACHE_DIR = "/root/.cache/huggingface"

app = modal.App("math-drift")

hf_cache = modal.Volume.from_name("hf-cache", create_if_missing=True)

image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install(
        "torch==2.6.0",
        "transformers==4.49.0",
        "accelerate==1.4.0",
        "datasets==3.3.0",
        "huggingface_hub[hf_transfer]==0.29.0",
        "numpy",
    )
    .env({"HF_HUB_ENABLE_HF_TRANSFER": "1", "HF_HOME": CACHE_DIR})
    # This is the new part vs. stage1/stage2: it copies harness.py (and this
    # stage0_smoke.py) from your local folder into the container image, so
    # `from harness import ...` works remotely. Both files must sit in the
    # SAME folder as this script when you run it.
    .add_local_python_source("harness", "stage0_smoke")
)

with image.imports():
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer


@app.cls(
    image=image,
    gpu="A10G",
    volumes={CACHE_DIR: hf_cache},
    timeout=30 * 60,
    scaledown_window=5 * 60,
)
class Model:
    @modal.enter()
    def load(self):
        self.tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
        self.model = AutoModelForCausalLM.from_pretrained(
            MODEL_NAME,
            torch_dtype=torch.bfloat16,
            device_map="cuda",
        )
        self.model.eval()

    @modal.method()
    def smoke(self):
        # Import here (inside the container) since harness/torch only exist remotely.
        from stage0_smoke import stage0_smoke
        stage0_smoke(self.model, self.tokenizer)


@app.local_entrypoint()
def main():
    Model().smoke.remote()
