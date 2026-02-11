"""3C: Eagle-3 speculative decoding. Run in vLLM TPU container."""
from vllm import LLM, SamplingParams
import json
import os
import time
from datetime import datetime

NAME = "eagle3"
MODEL = "Qwen/Qwen3-4B"
DRAFT_MODEL = "Tengyunw/qwen3_4b_eagle3"

llm = LLM(
    model=MODEL,
    speculative_config={
        "method": "eagle3",
        "model": DRAFT_MODEL,
        "num_speculative_tokens": 3,
        "draft_tensor_parallel_size": 1,
    },
    max_model_len=512,
    max_num_seqs=1,
    tensor_parallel_size=1,
    async_scheduling=0,
)

prompts = [
    "If 12 notebooks cost $36, what does one notebook cost?",
    "Solve: Find x if 3x + 7 = 40.",
    "Compute the remainder when 2^20 is divided by 7.",
    "How many positive divisors does 360 have?",
]

params = SamplingParams(temperature=0.0, max_tokens=128, ignore_eos=True)

start = time.perf_counter()
outputs = llm.generate(prompts, params)
elapsed = time.perf_counter() - start

total_tokens = sum(len(o.outputs[0].token_ids) for o in outputs)
tok_per_s = total_tokens / elapsed
print(f"Eagle-3: {total_tokens} tokens in {elapsed:.2f}s = {tok_per_s:.1f} tok/s")

# Save results (mount -v $(pwd)/results:/mnt/results to get files on host)
results_dir = os.environ.get("BENCHMARK_RESULTS_DIR", "/mnt/results")
if not os.path.isdir(results_dir):
    results_dir = os.path.join(os.path.dirname(__file__), "..", "results")
os.makedirs(results_dir, exist_ok=True)
ts = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
record = {
    "benchmark": NAME,
    "model": MODEL,
    "draft_model": DRAFT_MODEL,
    "num_speculative_tokens": 3,
    "total_tokens": total_tokens,
    "elapsed_sec": round(elapsed, 3),
    "tok_per_sec": round(tok_per_s, 2),
    "num_prompts": len(prompts),
    "max_tokens": 128,
    "timestamp_utc": datetime.utcnow().isoformat() + "Z",
}
path = os.path.join(results_dir, f"{NAME}_{ts}.json")
with open(path, "w") as f:
    json.dump(record, f, indent=2)
print(f"Saved: {path}")
summary_path = os.path.join(results_dir, "benchmarks.jsonl")
with open(summary_path, "a") as f:
    f.write(json.dumps(record) + "\n")
