import re
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
    )
    .env({"HF_HUB_ENABLE_HF_TRANSFER": "1", "HF_HOME": CACHE_DIR})
)

with image.imports():
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer


# ---------- Answer extraction and grading (pure Python, no GPU needed) ----------

def normalize_number(s: str):
    """Turn a messy answer string into a clean numeric string, or None."""
    if s is None:
        return None
    s = s.strip().replace(",", "").replace("$", "").replace("%", "").rstrip(".")
    m = re.search(r"-?\d+\.?\d*", s)
    return m.group(0) if m else None


def extract_gold_answer(gsm8k_answer: str):
    """GSM8K gold answers end with '#### <number>'. Pull that number out."""
    m = re.search(r"####\s*(.+)", gsm8k_answer)
    return normalize_number(m.group(1)) if m else None


def extract_pred_answer(response: str):
    """Read the model's final numeric answer, trusting \\boxed{} above all.
    If the model never closed </think>, it never truly answered: return None."""
    finished = "</think>" in response

    # Prefer a boxed answer, but only trust it if reasoning actually completed
    # OR the box sits after </think>. A box mid-loop is not a real final answer.
    boxed = re.findall(r"\\boxed\{([^}]*)\}", response)
    if boxed and finished:
        return normalize_number(boxed[-1])

    if not finished:
        return None  # truncated / looping: no real answer, do not scavenge

    # Finished but no box: look only after </think> for explicit answer phrasing.
    final = response.split("</think>")[-1]
    m = re.search(
        r"(?:final answer|answer|total)\s*(?:is|=|:)?\s*\$?(-?[\d,]+\.?\d*)",
        final, re.IGNORECASE,
    )
    if m:
        return normalize_number(m.group(1))

    nums = re.findall(r"-?[\d,]+\.?\d*", final)
    return normalize_number(nums[-1]) if nums else None


def is_correct(pred, gold) -> bool:
    if pred is None or gold is None:
        return False
    try:
        return abs(float(pred) - float(gold)) < 1e-4
    except ValueError:
        return pred.strip() == gold.strip()


# ---------- The model, with a batch evaluation method ----------

@app.cls(
    image=image,
    gpu="A10G",
    volumes={CACHE_DIR: hf_cache},
    timeout=60 * 60,
    scaledown_window=5 * 60,
    memory=32768,                     # reserve 32 GB CPU RAM so the checkpoint loads without an OOM kill
)
class Model:
    @modal.enter()
    def load(self):
        self.tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
        self.model = AutoModelForCausalLM.from_pretrained(
            MODEL_NAME,
            torch_dtype=torch.bfloat16,
            device_map="cuda",
            low_cpu_mem_usage=True,   # stream weights to GPU, avoids the CPU-RAM spike on load
        )
        self.model.eval()

    def _run(self, question: str, max_new_tokens: int) -> str:
        # Ask for a boxed final answer. R1-distill models are trained to comply,
        # which gives us one unambiguous thing to read instead of scavenging prose.
        prompt = (
            question
            + "\n\nPlease reason step by step, and put your final answer within \\boxed{}."
        )
        messages = [{"role": "user", "content": prompt}]
        inputs = self.tokenizer.apply_chat_template(
            messages,
            add_generation_prompt=True,
            return_tensors="pt",
            return_dict=True,          # gives us input_ids AND attention_mask
        ).to("cuda")
        with torch.no_grad():
            output = self.model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=True,
                temperature=0.6,
                top_p=0.95,
                pad_token_id=self.tokenizer.eos_token_id,
            )
        new_tokens = output[0][inputs["input_ids"].shape[1]:]
        return self.tokenizer.decode(new_tokens, skip_special_tokens=True)

    @modal.method()
    def evaluate(self, n_problems: int = 20, max_new_tokens: int = 8192):
        from datasets import load_dataset
        ds = load_dataset("openai/gsm8k", "main", split="test")
        results = []
        for i in range(n_problems):
            q = ds[i]["question"]
            gold = extract_gold_answer(ds[i]["answer"])
            trace = self._run(q, max_new_tokens)
            pred = extract_pred_answer(trace)
            results.append({
                "i": i,
                "gold": gold,
                "pred": pred,
                "correct": is_correct(pred, gold),
                "finished": "</think>" in trace,   # did it finish reasoning in budget?
                "chars": len(trace),
                "trace": trace,
            })
        return results


@app.local_entrypoint()
def main(n: int = 20):
    results = Model().evaluate.remote(n_problems=n)

    correct = sum(r["correct"] for r in results)
    unfinished = sum(not r["finished"] for r in results)
    print("=" * 80)
    print(f"Accuracy: {correct}/{len(results)} = {100 * correct / len(results):.1f}%")
    print(f"Traces that ran out of tokens (no </think>): {unfinished}")
    print("=" * 80)

    for r in results:
        flag = "OK " if r["correct"] else "XX "
        fin = "" if r["finished"] else "  [TRUNCATED]"
        print(f"{flag} #{r['i']:>3}  gold={str(r['gold']):>8}  pred={str(r['pred']):>8}{fin}")

    # Print the first wrong trace in full, so you can see whether it is a
    # real model mistake or an extraction bug.
    for r in results:
        if not r["correct"]:
            print("\n" + "=" * 80)
            print(f"First incorrect case (#{r['i']}): gold={r['gold']}, pred={r['pred']}")
            print("-" * 80)
            print(r["trace"])
            print("=" * 80)
            break