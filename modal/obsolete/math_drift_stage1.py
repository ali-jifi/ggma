import modal

MODEL_NAME = "deepseek-ai/DeepSeek-R1-Distill-Llama-8B"
CACHE_DIR = "/root/.cache/huggingface"

app = modal.App("math-drift")

# A persistent disk so the ~16GB of weights download once, then are reused every run.
hf_cache = modal.Volume.from_name("hf-cache", create_if_missing=True)

# The cloud container: Python + the libraries we need. Modal builds this once and caches it.
image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install(
        "torch==2.6.0",
        "transformers==4.49.0",
        "accelerate==1.4.0",
        "datasets==3.3.0",
        "huggingface_hub[hf_transfer]==0.29.0",
    )
    .env({"HF_HUB_ENABLE_HF_TRANSFER": "1", "HF_HOME": CACHE_DIR})
)

with image.imports():
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer


@app.cls(
    image=image,
    gpu="A10G",                       # 24GB GPU; an 8B model in bf16 is ~16GB, so this fits
    volumes={CACHE_DIR: hf_cache},    # mount the weight cache
    timeout=30 * 60,                  # allow up to 30 min (first run downloads the model)
    scaledown_window=5 * 60,          # keep the GPU warm 5 min between calls
)
class Model:
    @modal.enter()
    def load(self):
        # Runs ONCE when the container starts, so we load the model a single time.
        self.tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
        self.model = AutoModelForCausalLM.from_pretrained(
            MODEL_NAME,
            torch_dtype=torch.bfloat16,   # half precision, fits the GPU, standard for inference
            device_map="cuda",            # put the whole model on one GPU (clean for hooks later)
        )
        self.model.eval()

    @modal.method()
    def generate(self, question: str, max_new_tokens: int = 2048):
        # DeepSeek-R1 models: no system prompt, temperature 0.6, top_p 0.95 (their recommended settings).
        messages = [{"role": "user", "content": question}]
        input_ids = self.tokenizer.apply_chat_template(
            messages, add_generation_prompt=True, return_tensors="pt"
        ).to("cuda")

        with torch.no_grad():
            output = self.model.generate(
                input_ids,
                max_new_tokens=max_new_tokens,
                do_sample=True,
                temperature=0.6,
                top_p=0.95,
            )

        # Keep only the newly generated tokens (everything after the prompt).
        new_tokens = output[0][input_ids.shape[1]:]
        return self.tokenizer.decode(new_tokens, skip_special_tokens=True)


@app.local_entrypoint()
def main():
    question = (
        "Natalia sold clips to 48 of her friends in April, and then she sold "
        "half as many clips in May. How many clips did she sell altogether "
        "in April and May?"
    )
    model = Model()
    trace = model.generate.remote(question)
    print("=" * 80)
    print(trace)
    print("=" * 80)