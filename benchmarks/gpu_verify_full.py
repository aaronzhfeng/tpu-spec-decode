"""V7: GPU full target model forward pass at K=16 vs K=128.

Validates that Doc 42's isolated-component measurements (FFN matmuls,
attention matmuls tested separately) compose correctly into the actual
end-to-end verification penalty on GPU.

Loads the real Qwen3-4B model (not synthetic weights) and measures full
forward pass latency including FFN, attention, RMSNorm, RoPE, etc.

Usage:
    python benchmarks/gpu_verify_full.py
    python benchmarks/gpu_verify_full.py --context-lengths 256,512,1024,2048
    python benchmarks/gpu_verify_full.py --trials 50 --output-json results/gpu_verify_full.json
"""

import argparse
import json
import time

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


def gpu_sync():
    torch.cuda.synchronize()


def main():
    parser = argparse.ArgumentParser(
        description="V7: GPU full forward pass verification benchmark")
    parser.add_argument("--model", type=str, default="Qwen/Qwen3-4B",
                        help="Target model to benchmark")
    parser.add_argument("--trials", type=int, default=50,
                        help="Number of timed trials per (L, K) pair")
    parser.add_argument("--warmup", type=int, default=10,
                        help="Warmup iterations per (L, K) pair")
    parser.add_argument("--context-lengths", type=str, default="64,256,512,1024",
                        help="Comma-separated context lengths L")
    parser.add_argument("--k-values", type=str, default="16,32,64,128",
                        help="Comma-separated K (query token) values")
    parser.add_argument("--output-json", type=str, default=None)
    args = parser.parse_args()

    if not torch.cuda.is_available():
        print("[ERROR] No CUDA GPU found.")
        return

    device = torch.device("cuda")
    gpu_name = torch.cuda.get_device_name(0)
    context_lengths = [int(x) for x in args.context_lengths.split(",")]
    k_values = [int(x) for x in args.k_values.split(",")]

    print(f"[INFO] GPU: {gpu_name}")
    print(f"[INFO] PyTorch: {torch.__version__}")
    print(f"[INFO] Context lengths: {context_lengths}")
    print(f"[INFO] K values: {k_values}")

    # ── Load model ──────────────────────────────────────────────────────
    print(f"\n[INFO] Loading {args.model}...")
    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        attn_implementation="sdpa",
        dtype=torch.bfloat16,
    ).to(device).eval()

    tokenizer = AutoTokenizer.from_pretrained(args.model)
    print(f"[INFO] Model loaded. GPU memory: "
          f"{torch.cuda.memory_allocated()/1e9:.1f} GB")

    results = {}

    for ctx_len in context_lengths:
        print(f"\n{'='*70}")
        print(f"  CONTEXT LENGTH: L={ctx_len}")
        print(f"{'='*70}")

        results[ctx_len] = {}

        for K in k_values:
            # Build inputs: prefill context of length L, then verify K tokens
            # Use random token IDs (content doesn't affect latency)
            prefill_ids = torch.randint(100, 10000, (1, ctx_len),
                                        device=device)
            verify_ids = torch.randint(100, 10000, (1, K), device=device)

            # Warmup — full prefill + verify cycle
            with torch.no_grad():
                for _ in range(args.warmup):
                    # Prefill: build KV cache
                    prefill_out = model(
                        prefill_ids,
                        use_cache=True,
                    )
                    past_kv = prefill_out.past_key_values

                    # Verify: K new tokens against existing KV cache
                    verify_pos = torch.arange(
                        ctx_len, ctx_len + K, device=device).unsqueeze(0)
                    _ = model(
                        verify_ids,
                        past_key_values=past_kv,
                        position_ids=verify_pos,
                        use_cache=False,
                    )
                    gpu_sync()

            # Timed trials — time ONLY the verify forward pass
            latencies = []
            with torch.no_grad():
                for _ in range(args.trials):
                    # Fresh prefill each trial (consistent KV cache state)
                    prefill_out = model(
                        prefill_ids,
                        use_cache=True,
                    )
                    past_kv = prefill_out.past_key_values
                    gpu_sync()

                    # Time the verify pass
                    verify_pos = torch.arange(
                        ctx_len, ctx_len + K, device=device).unsqueeze(0)

                    gpu_sync()
                    t0 = time.perf_counter()
                    _ = model(
                        verify_ids,
                        past_key_values=past_kv,
                        position_ids=verify_pos,
                        use_cache=False,
                    )
                    gpu_sync()
                    t1 = time.perf_counter()
                    latencies.append((t1 - t0) * 1000)

            mean_ms = sum(latencies) / len(latencies)
            std_ms = (sum((l - mean_ms) ** 2 for l in latencies)
                      / len(latencies)) ** 0.5

            results[ctx_len][K] = {
                "mean_ms": round(mean_ms, 3),
                "std_ms": round(std_ms, 3),
            }

        # Print results for this context length
        base_ms = results[ctx_len][k_values[0]]["mean_ms"]
        print(f"\n  {'K':>5} | {'Verify (ms)':>14} | {'vs K={}'.format(k_values[0]):>10}")
        print("  " + "-" * 40)
        for K in k_values:
            r = results[ctx_len][K]
            ratio = r["mean_ms"] / base_ms if base_ms > 0 else 0
            results[ctx_len][K]["ratio"] = round(ratio, 2)
            print(f"  {K:>5} | {r['mean_ms']:>8.3f} ± {r['std_ms']:.3f} "
                  f"| {ratio:>9.2f}x")

    # ==================================================================
    # Summary
    # ==================================================================
    print(f"\n{'='*70}")
    print("SUMMARY: K=128/K=16 RATIO BY CONTEXT LENGTH")
    print(f"{'='*70}")

    print(f"\n  {'Context':>8} | {'K=16 (ms)':>10} | {'K=128 (ms)':>11} | "
          f"{'GPU ratio':>10} | {'Verdict':>20}")
    print("  " + "-" * 70)

    # GPU isolated attention ratios from Doc 42 for comparison
    gpu_attn_only = {64: 1.04, 256: 1.58, 512: 1.98, 1024: 2.51}

    for ctx in context_lengths:
        if 16 in results[ctx] and 128 in results[ctx]:
            k16 = results[ctx][16]["mean_ms"]
            k128 = results[ctx][128]["mean_ms"]
            ratio = k128 / k16 if k16 > 0 else 0

            if ratio < 1.15:
                verdict = "FLAT"
            elif ratio < 1.5:
                verdict = "MILD SCALING"
            elif ratio < 2.0:
                verdict = "MODERATE SCALING"
            else:
                verdict = "LINEAR SCALING"

            print(f"  {ctx:>8} | {k16:>10.3f} | {k128:>11.3f} | "
                  f"{ratio:>9.2f}x | {verdict:>20}")

    # Comparison with isolated measurements
    print(f"\n{'='*70}")
    print("COMPARISON: FULL FORWARD PASS vs ISOLATED COMPONENTS (Doc 42)")
    print(f"{'='*70}")
    print(f"\n  {'Context':>8} | {'Full FP ratio':>14} | {'Attn-only ratio':>16} | "
          f"{'FFN-only ratio':>15}")
    print("  " + "-" * 65)

    for ctx in context_lengths:
        if 16 in results[ctx] and 128 in results[ctx]:
            full_ratio = results[ctx][128]["mean_ms"] / results[ctx][16]["mean_ms"]
            attn_ratio = gpu_attn_only.get(ctx, "N/A")
            ffn_ratio = 1.09  # From Doc 42

            if isinstance(attn_ratio, float):
                print(f"  {ctx:>8} | {full_ratio:>13.2f}x | {attn_ratio:>15.2f}x | "
                      f"{ffn_ratio:>14.2f}x")
            else:
                print(f"  {ctx:>8} | {full_ratio:>13.2f}x | {'N/A':>15} | "
                      f"{ffn_ratio:>14.2f}x")

    # Save
    if args.output_json:
        import os
        os.makedirs(os.path.dirname(os.path.abspath(args.output_json)),
                    exist_ok=True)
        output = {
            "metadata": {
                "gpu": gpu_name,
                "pytorch_version": torch.__version__,
                "model": args.model,
                "trials": args.trials,
                "warmup": args.warmup,
                "dtype": "bfloat16",
                "attn_implementation": "sdpa",
            },
            "context_lengths": context_lengths,
            "k_values": k_values,
            "results": {
                str(ctx): {
                    str(k): v for k, v in ctx_results.items()
                }
                for ctx, ctx_results in results.items()
            },
        }
        with open(args.output_json, "w") as f:
            json.dump(output, f, indent=2)
        print(f"\n  Results saved to: {args.output_json}")


if __name__ == "__main__":
    main()
