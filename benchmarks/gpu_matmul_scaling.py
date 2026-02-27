"""GPU matmul scaling benchmark — companion to drafter_scaling.py (TPU).

Standalone script with ZERO dependencies beyond PyTorch.
Measures raw matmul latency at K=16,32,64,128 for the same dimensions
used in the TPU benchmark, proving that GPU scales linearly while
TPU is flat (MXU tile amortization).

Model dimensions (Qwen3-4B / DFlash):
  - DFlash:  5 layers,  hidden=2560, intermediate=9728
  - Target: 36 layers, hidden=2560, intermediate=9728
  - Attention: 32 heads, head_dim=80

These are hardcoded to match our TPU measurements exactly.
No model loading required — pure matmul microbenchmark.

Usage:
    pip install torch  # only dependency
    python gpu_matmul_scaling.py
    python gpu_matmul_scaling.py --trials 50 --warmup 10
    python gpu_matmul_scaling.py --output-json results.json
"""

import argparse
import json
import time

import torch


def gpu_sync():
    torch.cuda.synchronize()


def main():
    parser = argparse.ArgumentParser(
        description="GPU matmul scaling benchmark (companion to TPU drafter_scaling.py)")
    parser.add_argument("--trials", type=int, default=20,
                        help="Number of timed trials per configuration")
    parser.add_argument("--warmup", type=int, default=5,
                        help="Number of warmup iterations per configuration")
    parser.add_argument("--output-json", type=str, default=None,
                        help="Optional path to save results as JSON")
    args = parser.parse_args()

    if not torch.cuda.is_available():
        print("[ERROR] No CUDA GPU found. This benchmark requires a GPU.")
        print("        Run on a machine with an NVIDIA GPU (A100, H100, etc.)")
        return

    device = torch.device("cuda")
    gpu_name = torch.cuda.get_device_name(0)
    print(f"[INFO] GPU: {gpu_name}")
    print(f"[INFO] CUDA: {torch.version.cuda}")
    print(f"[INFO] PyTorch: {torch.__version__}")

    # ── Model dimensions (hardcoded to match TPU benchmark exactly) ──────
    # These come from Qwen3-4B (target) and Qwen3-4B-DFlash-b16 (drafter)
    DRAFT_HIDDEN = 2560
    DRAFT_INTERMEDIATE = 9728
    DRAFT_LAYERS = 5

    TARGET_HIDDEN = 2560
    TARGET_INTERMEDIATE = 9728
    TARGET_LAYERS = 36

    NUM_HEADS = 32
    HEAD_DIM = 80  # 2560 / 32

    print(f"\n[INFO] DFlash: {DRAFT_LAYERS} layers, hidden={DRAFT_HIDDEN}, "
          f"intermediate={DRAFT_INTERMEDIATE}")
    print(f"[INFO] Target: {TARGET_LAYERS} layers, hidden={TARGET_HIDDEN}, "
          f"intermediate={TARGET_INTERMEDIATE}")
    print(f"[INFO] Attention: {NUM_HEADS} heads, head_dim={HEAD_DIM}")

    k_values = [16, 32, 64, 128]
    kv_lengths = [64, 256, 512, 1024]
    results = {}

    # ==================================================================
    # Part 1: Raw matmul scaling (DFlash-sized)
    # ==================================================================
    print("\n" + "=" * 70)
    print("PART 1: RAW MATMUL SCALING (DFlash dimensions)")
    print("=" * 70)

    print(f"\n  DFlash FFN matmul: (K, {DRAFT_HIDDEN}) x ({DRAFT_HIDDEN}, "
          f"{DRAFT_INTERMEDIATE})")
    print(f"  {DRAFT_LAYERS} layers, simulating full forward pass FFN cost")

    w_gate = torch.ones(DRAFT_HIDDEN, DRAFT_INTERMEDIATE,
                        dtype=torch.bfloat16, device=device)
    w_up = torch.ones(DRAFT_HIDDEN, DRAFT_INTERMEDIATE,
                      dtype=torch.bfloat16, device=device)
    w_down = torch.ones(DRAFT_INTERMEDIATE, DRAFT_HIDDEN,
                        dtype=torch.bfloat16, device=device)

    print(f"\n  {'K':>5} | {'Per-layer (ms)':>14} | {'Full model (ms)':>16} "
          f"| {'vs K=16':>8}")
    print("  " + "-" * 55)

    base_time = None
    results["part1_dflash"] = []

    for K in k_values:
        x = torch.ones(K, DRAFT_HIDDEN, dtype=torch.bfloat16, device=device)

        # Warmup
        for _ in range(args.warmup):
            g = torch.mm(x, w_gate)
            u = torch.mm(x, w_up)
            h = g * u
            o = torch.mm(h, w_down)
            gpu_sync()

        # Timed trials
        latencies = []
        for _ in range(args.trials):
            gpu_sync()
            t0 = time.perf_counter()
            g = torch.mm(x, w_gate)
            u = torch.mm(x, w_up)
            h = g * u
            o = torch.mm(h, w_down)
            gpu_sync()
            t1 = time.perf_counter()
            latencies.append((t1 - t0) * 1000)

        mean_ms = sum(latencies) / len(latencies)
        std_ms = (sum((l - mean_ms) ** 2 for l in latencies)
                  / len(latencies)) ** 0.5
        full_model_ms = mean_ms * DRAFT_LAYERS
        if base_time is None:
            base_time = full_model_ms
        ratio = full_model_ms / base_time

        results["part1_dflash"].append({
            "K": K, "per_layer_ms": round(mean_ms, 3),
            "std_ms": round(std_ms, 3),
            "full_model_ms": round(full_model_ms, 3),
            "ratio": round(ratio, 2)
        })

        print(f"  {K:>5} | {mean_ms:>10.3f} ± {std_ms:.3f} | "
              f"{full_model_ms:>16.3f} | {ratio:>7.2f}x")

    # ==================================================================
    # Part 2: Raw matmul scaling (Target model dimensions)
    # ==================================================================
    print("\n" + "=" * 70)
    print("PART 2: RAW MATMUL SCALING (Target model dimensions)")
    print("=" * 70)

    print(f"\n  Target FFN matmul: (K, {TARGET_HIDDEN}) x ({TARGET_HIDDEN}, "
          f"{TARGET_INTERMEDIATE})")
    print(f"  {TARGET_LAYERS} layers")

    tw_gate = torch.ones(TARGET_HIDDEN, TARGET_INTERMEDIATE,
                         dtype=torch.bfloat16, device=device)
    tw_up = torch.ones(TARGET_HIDDEN, TARGET_INTERMEDIATE,
                       dtype=torch.bfloat16, device=device)
    tw_down = torch.ones(TARGET_INTERMEDIATE, TARGET_HIDDEN,
                         dtype=torch.bfloat16, device=device)

    print(f"\n  {'K':>5} | {'Per-layer (ms)':>14} | {'Full model (ms)':>16} "
          f"| {'vs K=16':>8}")
    print("  " + "-" * 55)

    base_time = None
    results["part2_target"] = []

    for K in k_values:
        x = torch.ones(K, TARGET_HIDDEN, dtype=torch.bfloat16, device=device)

        for _ in range(args.warmup):
            g = torch.mm(x, tw_gate)
            u = torch.mm(x, tw_up)
            h = g * u
            o = torch.mm(h, tw_down)
            gpu_sync()

        latencies = []
        for _ in range(args.trials):
            gpu_sync()
            t0 = time.perf_counter()
            g = torch.mm(x, tw_gate)
            u = torch.mm(x, tw_up)
            h = g * u
            o = torch.mm(h, tw_down)
            gpu_sync()
            t1 = time.perf_counter()
            latencies.append((t1 - t0) * 1000)

        mean_ms = sum(latencies) / len(latencies)
        std_ms = (sum((l - mean_ms) ** 2 for l in latencies)
                  / len(latencies)) ** 0.5
        full_model_ms = mean_ms * TARGET_LAYERS
        if base_time is None:
            base_time = full_model_ms
        ratio = full_model_ms / base_time

        results["part2_target"].append({
            "K": K, "per_layer_ms": round(mean_ms, 3),
            "std_ms": round(std_ms, 3),
            "full_model_ms": round(full_model_ms, 3),
            "ratio": round(ratio, 2)
        })

        print(f"  {K:>5} | {mean_ms:>10.3f} ± {std_ms:.3f} | "
              f"{full_model_ms:>16.3f} | {ratio:>7.2f}x")

    # ==================================================================
    # Part 3: Attention matmul scaling
    # ==================================================================
    print("\n" + "=" * 70)
    print("PART 3: ATTENTION Q×K^T SCALING (Target model)")
    print("=" * 70)

    print(f"\n  Attention: Q(K, {HEAD_DIM}) × K^T({HEAD_DIM}, KV_len) "
          f"per head ({NUM_HEADS} heads)")

    results["part3_attention"] = {}

    for kv_len in kv_lengths:
        print(f"\n  --- KV length = {kv_len} ---")
        print(f"  {'K_query':>8} | {'Attn (ms)':>10} | {'vs K=16':>8}")
        print("  " + "-" * 35)

        K_key = torch.ones(NUM_HEADS, kv_len, HEAD_DIM,
                           dtype=torch.bfloat16, device=device)
        base_attn = None
        results["part3_attention"][str(kv_len)] = []

        for K in k_values:
            Q = torch.ones(NUM_HEADS, K, HEAD_DIM,
                           dtype=torch.bfloat16, device=device)

            for _ in range(args.warmup):
                scores = torch.bmm(Q, K_key.transpose(1, 2))
                gpu_sync()

            latencies = []
            for _ in range(args.trials):
                gpu_sync()
                t0 = time.perf_counter()
                scores = torch.bmm(Q, K_key.transpose(1, 2))
                gpu_sync()
                t1 = time.perf_counter()
                latencies.append((t1 - t0) * 1000)

            mean_ms = sum(latencies) / len(latencies)
            std_ms = (sum((l - mean_ms) ** 2 for l in latencies)
                      / len(latencies)) ** 0.5
            if base_attn is None:
                base_attn = mean_ms
            ratio = mean_ms / base_attn

            results["part3_attention"][str(kv_len)].append({
                "K": K, "mean_ms": round(mean_ms, 3),
                "std_ms": round(std_ms, 3),
                "ratio": round(ratio, 2)
            })

            print(f"  {K:>8} | {mean_ms:>7.3f} ± {std_ms:.2f} "
                  f"| {ratio:>7.2f}x")

    # ==================================================================
    # Summary
    # ==================================================================
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)

    # Extract K=128 ratios for summary
    dflash_128 = results["part1_dflash"][-1]["ratio"]
    target_128 = results["part2_target"][-1]["ratio"]
    attn_128_256 = results["part3_attention"]["256"][-1]["ratio"]

    print(f"\n  GPU K=128 vs K=16 ratios:")
    print(f"    DFlash FFN:   {dflash_128:.2f}x")
    print(f"    Target FFN:   {target_128:.2f}x")
    print(f"    Attention:    {attn_128_256:.2f}x (KV=256)")

    print(f"\n  TPU K=128 vs K=16 ratios (from Doc 37):")
    print(f"    DFlash FFN:   0.95x")
    print(f"    Target FFN:   0.96x")
    print(f"    Attention:    0.97x (KV=256)")

    if dflash_128 > 1.5:
        print(f"\n  CONFIRMED: GPU scales ~linearly ({dflash_128:.1f}x at K=128)")
        print(f"  while TPU is flat (~1.0x). MXU tile amortization is TPU-specific.")
    else:
        print(f"\n  NOTE: GPU ratio is {dflash_128:.2f}x — check if this GPU has")
        print(f"  unusual tensor core behavior at these small dimensions.")

    # Add metadata
    results["metadata"] = {
        "gpu": gpu_name,
        "cuda_version": torch.version.cuda,
        "pytorch_version": torch.__version__,
        "trials": args.trials,
        "warmup": args.warmup,
        "dtype": "bfloat16",
        "draft_hidden": DRAFT_HIDDEN,
        "draft_intermediate": DRAFT_INTERMEDIATE,
        "draft_layers": DRAFT_LAYERS,
        "target_hidden": TARGET_HIDDEN,
        "target_intermediate": TARGET_INTERMEDIATE,
        "target_layers": TARGET_LAYERS,
        "num_heads": NUM_HEADS,
        "head_dim": HEAD_DIM,
        "tpu_reference": {
            "device": "TPU v4-8",
            "dflash_128_ratio": 0.95,
            "target_128_ratio": 0.96,
            "attention_128_ratio_kv256": 0.97,
            "source": "Doc 37 (benchmarks/drafter_scaling.py)"
        }
    }

    if args.output_json:
        with open(args.output_json, "w") as f:
            json.dump(results, f, indent=2)
        print(f"\n  Results saved to: {args.output_json}")


if __name__ == "__main__":
    main()
