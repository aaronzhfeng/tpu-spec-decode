"""GPU draft forward pass speed benchmark — K=16 vs K=128.

Measures the actual DFlash draft model forward pass latency at different
block sizes (K values). This isolates just the drafting cost, excluding
verification/acceptance logic.

DFlash is a block-parallel drafter: it predicts all K tokens in ONE forward
pass. With the current b16-trained model, generating 128 draft tokens
requires 8 sequential passes. A hypothetical b128 model would do it in 1.

This benchmark measures:
  1. Single forward pass at K=16, 32, 64, 128 (raw compute scaling)
  2. Multi-pass simulation: 8×K=16 vs 1×K=128 (practical cost comparison)

Usage:
    python benchmarks/gpu_draft_speed.py
    python benchmarks/gpu_draft_speed.py --trials 50 --warmup 20
    python benchmarks/gpu_draft_speed.py --output-json results/gpu_draft_speed.json
"""

import argparse
import json
import time

import torch
from transformers import AutoModel


def gpu_sync():
    torch.cuda.synchronize()


def main():
    parser = argparse.ArgumentParser(
        description="GPU DFlash draft forward pass speed benchmark")
    parser.add_argument("--trials", type=int, default=50,
                        help="Number of timed trials per configuration")
    parser.add_argument("--warmup", type=int, default=20,
                        help="Number of warmup iterations")
    parser.add_argument("--output-json", type=str, default=None,
                        help="Optional path to save results as JSON")
    args = parser.parse_args()

    if not torch.cuda.is_available():
        print("[ERROR] No CUDA GPU found.")
        return

    device = torch.device("cuda")
    gpu_name = torch.cuda.get_device_name(0)
    print(f"[INFO] GPU: {gpu_name}")
    print(f"[INFO] PyTorch: {torch.__version__}")

    # ── Load DFlash draft model ─────────────────────────────────────────
    print("\n[INFO] Loading DFlash draft model (z-lab/Qwen3-4B-DFlash-b16)...")
    model = AutoModel.from_pretrained(
        "z-lab/Qwen3-4B-DFlash-b16",
        trust_remote_code=True,
        dtype=torch.bfloat16,
    ).to(device).eval()

    # Model dimensions from config
    hidden_size = 2560
    num_target_layers = 5  # target_layer_ids: [1, 9, 17, 25, 33]
    fc_input_dim = hidden_size * num_target_layers  # 12800

    print(f"[INFO] Model loaded: 5 layers, hidden={hidden_size}")
    print(f"[INFO] fc input dim: {fc_input_dim} (5 target layers × {hidden_size})")

    k_values = [16, 32, 64, 128]
    ctx_len = 64  # context length (target hidden states from prefill)
    results = {}

    # ==================================================================
    # Part 1: Single forward pass at each K
    # ==================================================================
    print("\n" + "=" * 70)
    print("PART 1: SINGLE DRAFT FORWARD PASS AT EACH K")
    print("=" * 70)
    print(f"\n  Measures one model.forward() call with K query positions")
    print(f"  Context length: {ctx_len} (target hidden states)")
    print(f"  {args.warmup} warmup, {args.trials} timed trials\n")

    print(f"  {'K':>5} | {'Forward (ms)':>14} | {'vs K=16':>8}")
    print("  " + "-" * 40)

    base_time = None
    results["single_pass"] = []

    for K in k_values:
        # DFlash attention concatenates target_hidden (ctx) + noise (K) for KV
        # So position_ids must cover ctx_len + K positions
        noise_emb = torch.randn(1, K, hidden_size,
                                dtype=torch.bfloat16, device=device)
        target_hid = torch.randn(1, ctx_len, fc_input_dim,
                                 dtype=torch.bfloat16, device=device)
        pos_ids = torch.arange(ctx_len + K, device=device).unsqueeze(0)

        # Warmup
        with torch.no_grad():
            for _ in range(args.warmup):
                _ = model(
                    position_ids=pos_ids,
                    noise_embedding=noise_emb,
                    target_hidden=target_hid,
                )
                gpu_sync()

        # Timed trials
        latencies = []
        with torch.no_grad():
            for _ in range(args.trials):
                gpu_sync()
                t0 = time.perf_counter()
                _ = model(
                    position_ids=pos_ids,
                    noise_embedding=noise_emb,
                    target_hidden=target_hid,
                )
                gpu_sync()
                t1 = time.perf_counter()
                latencies.append((t1 - t0) * 1000)

        mean_ms = sum(latencies) / len(latencies)
        std_ms = (sum((l - mean_ms) ** 2 for l in latencies)
                  / len(latencies)) ** 0.5
        if base_time is None:
            base_time = mean_ms
        ratio = mean_ms / base_time

        results["single_pass"].append({
            "K": K, "mean_ms": round(mean_ms, 3),
            "std_ms": round(std_ms, 3), "ratio": round(ratio, 2)
        })

        print(f"  {K:>5} | {mean_ms:>8.3f} ± {std_ms:.3f} | {ratio:>7.2f}x")

    # ==================================================================
    # Part 2: Practical comparison — 8×K=16 vs 1×K=128
    # ==================================================================
    print("\n" + "=" * 70)
    print("PART 2: PRACTICAL COMPARISON — 8×K=16 vs 1×K=128")
    print("=" * 70)
    print(f"\n  Current b16 model: 8 sequential passes to get 128 tokens")
    print(f"  Hypothetical b128: 1 pass to get 128 tokens\n")

    # 8 × K=16 (current approach for 128 draft tokens)
    noise_16 = torch.randn(1, 16, hidden_size,
                           dtype=torch.bfloat16, device=device)
    target_16 = torch.randn(1, ctx_len, fc_input_dim,
                            dtype=torch.bfloat16, device=device)
    pos_16 = torch.arange(ctx_len + 16, device=device).unsqueeze(0)

    with torch.no_grad():
        for _ in range(args.warmup):
            for _ in range(8):
                _ = model(position_ids=pos_16,
                          noise_embedding=noise_16,
                          target_hidden=target_16)
            gpu_sync()

    latencies_8x16 = []
    with torch.no_grad():
        for _ in range(args.trials):
            gpu_sync()
            t0 = time.perf_counter()
            for _ in range(8):
                _ = model(position_ids=pos_16,
                          noise_embedding=noise_16,
                          target_hidden=target_16)
            gpu_sync()
            t1 = time.perf_counter()
            latencies_8x16.append((t1 - t0) * 1000)

    mean_8x16 = sum(latencies_8x16) / len(latencies_8x16)
    std_8x16 = (sum((l - mean_8x16) ** 2 for l in latencies_8x16)
                / len(latencies_8x16)) ** 0.5

    # 1 × K=128 (hypothetical b128 model — same weights, different input size)
    noise_128 = torch.randn(1, 128, hidden_size,
                            dtype=torch.bfloat16, device=device)
    target_128 = torch.randn(1, ctx_len, fc_input_dim,
                             dtype=torch.bfloat16, device=device)
    pos_128 = torch.arange(ctx_len + 128, device=device).unsqueeze(0)

    with torch.no_grad():
        for _ in range(args.warmup):
            _ = model(position_ids=pos_128,
                      noise_embedding=noise_128,
                      target_hidden=target_128)
            gpu_sync()

    latencies_1x128 = []
    with torch.no_grad():
        for _ in range(args.trials):
            gpu_sync()
            t0 = time.perf_counter()
            _ = model(position_ids=pos_128,
                      noise_embedding=noise_128,
                      target_hidden=target_128)
            gpu_sync()
            t1 = time.perf_counter()
            latencies_1x128.append((t1 - t0) * 1000)

    mean_1x128 = sum(latencies_1x128) / len(latencies_1x128)
    std_1x128 = (sum((l - mean_1x128) ** 2 for l in latencies_1x128)
                 / len(latencies_1x128)) ** 0.5

    speedup = mean_8x16 / mean_1x128

    results["practical"] = {
        "8x_k16_ms": round(mean_8x16, 3),
        "8x_k16_std": round(std_8x16, 3),
        "1x_k128_ms": round(mean_1x128, 3),
        "1x_k128_std": round(std_1x128, 3),
        "speedup": round(speedup, 2),
    }

    print(f"  {'Method':>20} | {'Time (ms)':>14} | {'Tokens':>7} | {'ms/token':>10}")
    print("  " + "-" * 60)
    print(f"  {'8×K=16 (current)':>20} | {mean_8x16:>8.3f} ± {std_8x16:.3f} "
          f"| {128:>7} | {mean_8x16/128:>10.4f}")
    print(f"  {'1×K=128 (b128)':>20} | {mean_1x128:>8.3f} ± {std_1x128:.3f} "
          f"| {128:>7} | {mean_1x128/128:>10.4f}")
    print(f"\n  Speedup from b128 single-pass: {speedup:.2f}x")

    # ==================================================================
    # Summary
    # ==================================================================
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)

    k16_ms = results["single_pass"][0]["mean_ms"]
    k128_ms = results["single_pass"][-1]["mean_ms"]
    k128_ratio = results["single_pass"][-1]["ratio"]

    print(f"\n  Single forward pass scaling:")
    print(f"    K=16:  {k16_ms:.3f} ms")
    print(f"    K=128: {k128_ms:.3f} ms ({k128_ratio:.2f}x)")

    print(f"\n  To generate 128 draft tokens:")
    print(f"    Current (8×b16): {mean_8x16:.3f} ms")
    print(f"    b128 (1×b128):   {mean_1x128:.3f} ms ({speedup:.2f}x faster)")

    if k128_ratio < 1.5:
        print(f"\n  FINDING: Draft forward pass is memory-bound — K=128 costs")
        print(f"  only {k128_ratio:.2f}x vs K=16. A b128 model would save")
        print(f"  ~{mean_8x16 - mean_1x128:.1f}ms per draft cycle ({speedup:.1f}x faster).")
    else:
        print(f"\n  FINDING: Draft forward pass scales {k128_ratio:.2f}x from K=16→128.")

    results["metadata"] = {
        "gpu": gpu_name,
        "pytorch_version": torch.__version__,
        "model": "z-lab/Qwen3-4B-DFlash-b16",
        "trials": args.trials,
        "warmup": args.warmup,
        "dtype": "bfloat16",
    }

    if args.output_json:
        with open(args.output_json, "w") as f:
            json.dump(results, f, indent=2)
        print(f"\n  Results saved to: {args.output_json}")


if __name__ == "__main__":
    main()
