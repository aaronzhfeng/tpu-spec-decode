"""V12: GPU full forward pass scaling across context lengths.

Verifies whether the 1.24x GPU penalty (K=128 vs K=16) is constant
across context lengths, or grows with L. This is the GPU companion
to verify_context_scaling.py (TPU).

The claim (proposal v8): "GPU verification at K=128 costs 1.24x of
K=16, constant across context lengths."

If 1.24x grows at longer L (because attention's quadratic component
becomes larger), the GPU penalty is worse than claimed.

Requirements:
    pip install torch transformers accelerate

Usage:
    python benchmarks/gpu_verify_context_scaling.py
    python benchmarks/gpu_verify_context_scaling.py --trials 30 --output-json results/v12_gpu_context.json
"""

import argparse
import json
import time

import torch


def gpu_sync():
    torch.cuda.synchronize()


def main():
    parser = argparse.ArgumentParser(
        description="V12: GPU verify latency vs context length and K")
    parser.add_argument("--model", type=str, default="Qwen/Qwen3-4B")
    parser.add_argument("--trials", type=int, default=20)
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--k-values", type=str, default="16,64,128",
                        help="K values to test")
    parser.add_argument("--context-lengths", type=str, default="256,512,1024,2048",
                        help="Context lengths to test")
    parser.add_argument("--output-json", type=str, default=None)
    args = parser.parse_args()

    k_values = [int(x) for x in args.k_values.split(",")]
    context_lengths = [int(x) for x in args.context_lengths.split(",")]

    if not torch.cuda.is_available():
        print("[ERROR] No CUDA GPU found.")
        return

    device = torch.device("cuda")
    gpu_name = torch.cuda.get_device_name(0)
    gpu_mem = torch.cuda.get_device_properties(0).total_mem / 1e9
    print(f"[INFO] GPU: {gpu_name} ({gpu_mem:.1f} GB)")
    print(f"[INFO] CUDA: {torch.version.cuda}")
    print(f"[INFO] PyTorch: {torch.__version__}")

    # ── Load model ───────────────────────────────────────────────────────
    print(f"\n[INFO] Loading {args.model}...")
    from transformers import AutoModelForCausalLM

    model = AutoModelForCausalLM.from_pretrained(
        args.model, torch_dtype=torch.bfloat16, device_map="auto",
        trust_remote_code=True)
    model.eval()

    n_layers = model.config.num_hidden_layers
    print(f"[INFO] {n_layers} layers, hidden={model.config.hidden_size}")
    print(f"[INFO] Context lengths: {context_lengths}")
    print(f"[INFO] K values: {k_values}")
    print(f"[INFO] Trials: {args.trials}, Warmup: {args.warmup}")

    results = {
        "metadata": {
            "gpu": gpu_name, "gpu_mem_gb": round(gpu_mem, 1),
            "model": args.model, "n_layers": n_layers,
            "cuda": torch.version.cuda, "pytorch": torch.__version__,
            "trials": args.trials, "warmup": args.warmup,
        },
        "results": {},
    }

    # ── Run sweep ────────────────────────────────────────────────────────
    for L in context_lengths:
        print(f"\n{'='*60}")
        print(f"  CONTEXT LENGTH: {L}")
        print(f"{'='*60}")

        # Check memory feasibility
        estimated_kv_gb = 2 * n_layers * 2 * L * model.config.hidden_size * 2 / 1e9
        print(f"  Estimated KV cache: {estimated_kv_gb:.2f} GB")

        if estimated_kv_gb > gpu_mem * 0.5:
            print(f"  [WARN] KV cache may exceed available memory, skipping L={L}")
            continue

        # Build prefill tokens
        prompt_ids = torch.randint(100, 30000, (1, L), device=device)

        results["results"][str(L)] = {}

        for K in k_values:
            print(f"\n  --- K={K} (context={L}) ---")
            query_ids = torch.randint(100, 30000, (1, K), device=device)

            # Warmup
            for w in range(args.warmup):
                with torch.no_grad():
                    out = model(prompt_ids, use_cache=True)
                    past_kv = out.past_key_values
                    _ = model(query_ids, past_key_values=past_kv, use_cache=True)
                    gpu_sync()

            # Timed trials — measure ONLY the verify forward pass
            latencies = []
            for t in range(args.trials):
                with torch.no_grad():
                    # Fresh prefill each trial
                    out = model(prompt_ids, use_cache=True)
                    past_kv = out.past_key_values
                    gpu_sync()

                    # Time verify
                    t0 = time.perf_counter()
                    _ = model(query_ids, past_key_values=past_kv, use_cache=True)
                    gpu_sync()
                    t1 = time.perf_counter()
                    latencies.append((t1 - t0) * 1000)

            mean_ms = sum(latencies) / len(latencies)
            std_ms = (sum((x - mean_ms)**2 for x in latencies) / len(latencies)) ** 0.5

            results["results"][str(L)][str(K)] = {
                "mean_ms": round(mean_ms, 3),
                "std_ms": round(std_ms, 3),
                "per_token_ms": round(mean_ms / K, 3),
                "latencies": [round(x, 3) for x in latencies],
            }

            print(f"    K={K:>4}: {mean_ms:.2f} ± {std_ms:.2f} ms  "
                  f"({mean_ms/K:.3f} ms/token)")

    # ── Summary tables ───────────────────────────────────────────────────
    print(f"\n{'='*70}")
    print("RAW LATENCIES (ms)")
    print(f"{'='*70}")

    header = f"  {'Context':>8} |"
    for K in k_values:
        header += f" {'K='+str(K):>14} |"
    print(header)
    print("  " + "-" * (10 + 17 * len(k_values)))

    for L in context_lengths:
        if str(L) not in results["results"]:
            continue
        row = f"  {L:>8} |"
        for K in k_values:
            if str(K) in results["results"][str(L)]:
                r = results["results"][str(L)][str(K)]
                row += f" {r['mean_ms']:>8.2f} ± {r['std_ms']:.1f} |"
            else:
                row += f" {'N/A':>14} |"
        print(row)

    # ── K=128/K=16 ratio table (the critical metric) ────────────────────
    if 16 in k_values and 128 in k_values:
        print(f"\n{'='*70}")
        print("K=128 vs K=16 RATIO ACROSS CONTEXT LENGTHS")
        print(f"{'='*70}")
        print(f"  {'Context':>8} | {'K=16 (ms)':>10} | {'K=128 (ms)':>11} | "
              f"{'Ratio':>8} | {'vs 1.24x claim':>16}")
        print("  " + "-" * 65)

        ratios = []

        for L in context_lengths:
            Ls = str(L)
            if Ls not in results["results"]:
                continue
            if "16" not in results["results"][Ls] or "128" not in results["results"][Ls]:
                continue

            k16 = results["results"][Ls]["16"]["mean_ms"]
            k128 = results["results"][Ls]["128"]["mean_ms"]
            ratio = k128 / k16 if k16 > 0 else 0
            ratios.append(ratio)

            diff = ratio - 1.24
            if abs(diff) < 0.1:
                verdict = "CONSISTENT"
            elif diff > 0:
                verdict = f"+{diff:.2f} (WORSE)"
            else:
                verdict = f"{diff:.2f} (BETTER)"

            results["results"][Ls]["ratio_128_16"] = round(ratio, 3)

            print(f"  {L:>8} | {k16:>10.2f} | {k128:>11.2f} | "
                  f"{ratio:>7.2f}x | {verdict:>16}")

        if ratios:
            mean_ratio = sum(ratios) / len(ratios)
            max_ratio = max(ratios)
            min_ratio = min(ratios)
            print(f"\n  Mean ratio:  {mean_ratio:.2f}x")
            print(f"  Range:       {min_ratio:.2f}x – {max_ratio:.2f}x")

            if max_ratio - min_ratio < 0.15:
                print(f"  Verdict:     CONSTANT across context lengths (range < 0.15)")
            else:
                print(f"  Verdict:     VARIES with context length (range = {max_ratio - min_ratio:.2f})")

            results["summary"] = {
                "mean_ratio": round(mean_ratio, 3),
                "min_ratio": round(min_ratio, 3),
                "max_ratio": round(max_ratio, 3),
                "is_constant": max_ratio - min_ratio < 0.15,
            }

    # ── Comparison with TPU ──────────────────────────────────────────────
    tpu_ratios = {256: 1.02, 1024: 1.01, 2048: 0.91, 4096: 0.95}

    print(f"\n{'='*70}")
    print("GPU vs TPU K=128/K=16 RATIO COMPARISON")
    print(f"{'='*70}")
    print(f"  {'Context':>8} | {'GPU ratio':>10} | {'TPU ratio':>10} | {'GPU penalty':>12}")
    print("  " + "-" * 50)

    for L in context_lengths:
        Ls = str(L)
        if Ls not in results["results"] or "ratio_128_16" not in results["results"].get(Ls, {}):
            continue
        gpu_r = results["results"][Ls]["ratio_128_16"]
        tpu_r = tpu_ratios.get(L, None)
        if tpu_r:
            penalty = gpu_r / tpu_r
            print(f"  {L:>8} | {gpu_r:>9.2f}x | {tpu_r:>9.2f}x | {penalty:>10.1f}x worse")

    if args.output_json:
        # Strip raw latencies for cleaner output
        clean = {k: v for k, v in results.items()}
        for Ls in clean.get("results", {}):
            for Ks in clean["results"][Ls]:
                if isinstance(clean["results"][Ls][Ks], dict) and "latencies" in clean["results"][Ls][Ks]:
                    del clean["results"][Ls][Ks]["latencies"]
        with open(args.output_json, "w") as f:
            json.dump(results, f, indent=2)
        print(f"\nResults saved to {args.output_json}")


if __name__ == "__main__":
    main()
