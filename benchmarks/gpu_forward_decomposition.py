"""V10: GPU forward pass component decomposition.

Profiles a real Qwen3-4B forward pass on GPU, timing FFN, attention,
and other components separately using CUDA events. Measures the actual
component split at K=16 and K=128 to verify whether:

  FFN_fraction × FFN_ratio + attn_fraction × attn_ratio ≈ 1.24x

This resolves the discrepancy between the Doc 44 estimate (83/17 → 1.33x)
and the V7 measured full-pass ratio (1.24x).

Requirements:
    pip install torch transformers accelerate

Usage:
    python benchmarks/gpu_forward_decomposition.py
    python benchmarks/gpu_forward_decomposition.py --trials 30 --output-json results/v10_gpu_decomposition.json
    python benchmarks/gpu_forward_decomposition.py --context-lengths 256,1024
"""

import argparse
import json
import time

import torch
import torch.nn as nn


def gpu_sync():
    torch.cuda.synchronize()


def make_cuda_events():
    return torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)


def main():
    parser = argparse.ArgumentParser(
        description="V10: GPU forward pass component decomposition")
    parser.add_argument("--model", type=str, default="Qwen/Qwen3-4B",
                        help="HuggingFace model name")
    parser.add_argument("--trials", type=int, default=20)
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--k-values", type=str, default="16,128",
                        help="K values to test")
    parser.add_argument("--context-lengths", type=str, default="256,1024",
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
    print(f"[INFO] GPU: {gpu_name}")
    print(f"[INFO] CUDA: {torch.version.cuda}")
    print(f"[INFO] PyTorch: {torch.__version__}")

    # ── Load model ───────────────────────────────────────────────────────
    print(f"\n[INFO] Loading {args.model}...")
    from transformers import AutoModelForCausalLM, AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        args.model, torch_dtype=torch.bfloat16, device_map="auto",
        trust_remote_code=True)
    model.eval()

    config = model.config
    n_layers = config.num_hidden_layers
    print(f"[INFO] Loaded: {n_layers} layers, hidden={config.hidden_size}, "
          f"intermediate={config.intermediate_size}")
    print(f"[INFO] Attention: {config.num_attention_heads} Q heads, "
          f"{config.num_key_value_heads} KV heads, "
          f"head_dim={config.hidden_size // config.num_attention_heads}")

    results = {
        "metadata": {
            "gpu": gpu_name, "model": args.model,
            "cuda": torch.version.cuda, "pytorch": torch.__version__,
            "trials": args.trials, "warmup": args.warmup,
        },
        "experiments": {},
    }

    # ── Hook-based profiling ─────────────────────────────────────────────
    # We'll instrument each decoder layer to time FFN vs attention

    for L in context_lengths:
        for K in k_values:
            exp_key = f"L={L}_K={K}"
            print(f"\n{'='*70}")
            print(f"  EXPERIMENT: context={L}, K={K} query tokens")
            print(f"{'='*70}")

            # Build KV cache by running prefill
            prompt_ids = torch.randint(100, 30000, (1, L), device=device)
            query_ids = torch.randint(100, 30000, (1, K), device=device)

            # Method: time full forward, then time with selective component disable
            # Simpler approach: time the full forward with hooks on each layer

            layer_timings = {i: {"attn": [], "ffn": [], "other": []} for i in range(n_layers)}

            # Register hooks on each decoder layer's submodules
            hooks = []
            timing_data = {}

            def make_attn_pre_hook(layer_idx):
                def hook(module, input):
                    gpu_sync()
                    timing_data[f"attn_start_{layer_idx}"] = time.perf_counter()
                return hook

            def make_attn_post_hook(layer_idx):
                def hook(module, input, output):
                    gpu_sync()
                    timing_data[f"attn_end_{layer_idx}"] = time.perf_counter()
                return hook

            def make_ffn_pre_hook(layer_idx):
                def hook(module, input):
                    gpu_sync()
                    timing_data[f"ffn_start_{layer_idx}"] = time.perf_counter()
                return hook

            def make_ffn_post_hook(layer_idx):
                def hook(module, input, output):
                    gpu_sync()
                    timing_data[f"ffn_end_{layer_idx}"] = time.perf_counter()
                return hook

            # Find and hook the decoder layers
            decoder_layers = None
            if hasattr(model.model, 'layers'):
                decoder_layers = model.model.layers
            elif hasattr(model, 'transformer') and hasattr(model.transformer, 'h'):
                decoder_layers = model.transformer.h

            if decoder_layers is None:
                print("[ERROR] Cannot find decoder layers in model architecture.")
                return

            for i, layer in enumerate(decoder_layers):
                # Hook attention (self_attn)
                if hasattr(layer, 'self_attn'):
                    hooks.append(layer.self_attn.register_forward_pre_hook(make_attn_pre_hook(i)))
                    hooks.append(layer.self_attn.register_forward_hook(make_attn_post_hook(i)))
                # Hook FFN (mlp)
                if hasattr(layer, 'mlp'):
                    hooks.append(layer.mlp.register_forward_pre_hook(make_ffn_pre_hook(i)))
                    hooks.append(layer.mlp.register_forward_hook(make_ffn_post_hook(i)))

            # Warmup: prefill to build KV cache, then verify
            print(f"  Warming up ({args.warmup} iterations)...")
            for w in range(args.warmup):
                with torch.no_grad():
                    # Prefill
                    out = model(prompt_ids, use_cache=True)
                    past_kv = out.past_key_values
                    # Verify (the part we measure)
                    _ = model(query_ids, past_key_values=past_kv, use_cache=True)
                    gpu_sync()

            # Timed trials
            print(f"  Running {args.trials} timed trials...")
            full_latencies = []
            attn_totals = []
            ffn_totals = []

            for t in range(args.trials):
                timing_data.clear()

                with torch.no_grad():
                    # Fresh prefill each trial for consistency
                    out = model(prompt_ids, use_cache=True)
                    past_kv = out.past_key_values
                    gpu_sync()

                    # Time the verify forward pass
                    gpu_sync()
                    t0 = time.perf_counter()
                    _ = model(query_ids, past_key_values=past_kv, use_cache=True)
                    gpu_sync()
                    t1 = time.perf_counter()

                full_ms = (t1 - t0) * 1000
                full_latencies.append(full_ms)

                # Aggregate per-layer timings
                attn_total_ms = 0
                ffn_total_ms = 0
                for i in range(n_layers):
                    attn_key_s = f"attn_start_{i}"
                    attn_key_e = f"attn_end_{i}"
                    ffn_key_s = f"ffn_start_{i}"
                    ffn_key_e = f"ffn_end_{i}"

                    if attn_key_s in timing_data and attn_key_e in timing_data:
                        attn_ms = (timing_data[attn_key_e] - timing_data[attn_key_s]) * 1000
                        attn_total_ms += attn_ms
                        layer_timings[i]["attn"].append(attn_ms)

                    if ffn_key_s in timing_data and ffn_key_e in timing_data:
                        ffn_ms = (timing_data[ffn_key_e] - timing_data[ffn_key_s]) * 1000
                        ffn_total_ms += ffn_ms
                        layer_timings[i]["ffn"].append(ffn_ms)

                attn_totals.append(attn_total_ms)
                ffn_totals.append(ffn_total_ms)

            # Remove hooks
            for h in hooks:
                h.remove()

            # Compute statistics
            full_mean = sum(full_latencies) / len(full_latencies)
            full_std = (sum((x - full_mean)**2 for x in full_latencies) / len(full_latencies)) ** 0.5
            attn_mean = sum(attn_totals) / len(attn_totals)
            ffn_mean = sum(ffn_totals) / len(ffn_totals)
            other_mean = full_mean - attn_mean - ffn_mean

            attn_pct = attn_mean / full_mean * 100 if full_mean > 0 else 0
            ffn_pct = ffn_mean / full_mean * 100 if full_mean > 0 else 0
            other_pct = other_mean / full_mean * 100 if full_mean > 0 else 0

            results["experiments"][exp_key] = {
                "context_length": L, "K": K,
                "full_ms": round(full_mean, 3), "full_std": round(full_std, 3),
                "attn_ms": round(attn_mean, 3), "attn_pct": round(attn_pct, 1),
                "ffn_ms": round(ffn_mean, 3), "ffn_pct": round(ffn_pct, 1),
                "other_ms": round(other_mean, 3), "other_pct": round(other_pct, 1),
            }

            print(f"\n  Results (L={L}, K={K}):")
            print(f"    Full forward:  {full_mean:>8.2f} ± {full_std:.2f} ms")
            print(f"    Attention:     {attn_mean:>8.2f} ms ({attn_pct:.1f}%)")
            print(f"    FFN:           {ffn_mean:>8.2f} ms ({ffn_pct:.1f}%)")
            print(f"    Other:         {other_mean:>8.2f} ms ({other_pct:.1f}%)")

    # ── Summary: K=128 vs K=16 ratios ────────────────────────────────────
    print(f"\n{'='*70}")
    print("SUMMARY: K=128 vs K=16 RATIOS")
    print(f"{'='*70}")

    if 16 in k_values and 128 in k_values:
        print(f"\n  {'Context':>8} | {'Component':<12} | {'K=16 (ms)':>10} | "
              f"{'K=128 (ms)':>11} | {'Ratio':>8} | {'% of full':>10}")
        print("  " + "-" * 70)

        for L in context_lengths:
            k16_key = f"L={L}_K=16"
            k128_key = f"L={L}_K=128"

            if k16_key not in results["experiments"] or k128_key not in results["experiments"]:
                continue

            r16 = results["experiments"][k16_key]
            r128 = results["experiments"][k128_key]

            for comp, field, pct_field in [
                ("Full", "full_ms", None),
                ("Attention", "attn_ms", "attn_pct"),
                ("FFN", "ffn_ms", "ffn_pct"),
                ("Other", "other_ms", "other_pct"),
            ]:
                v16 = r16[field]
                v128 = r128[field]
                ratio = v128 / v16 if v16 > 0 else 0
                pct = f"{r16[pct_field]:.0f}%" if pct_field else ""

                print(f"  {L:>8} | {comp:<12} | {v16:>10.2f} | {v128:>11.2f} | "
                      f"{ratio:>7.2f}x | {pct:>10}")
            print("  " + "-" * 70)

        # Blended ratio check
        print(f"\n  BLENDED RATIO VERIFICATION:")
        for L in context_lengths:
            k16_key = f"L={L}_K=16"
            k128_key = f"L={L}_K=128"
            if k16_key not in results["experiments"]:
                continue

            r16 = results["experiments"][k16_key]
            r128 = results["experiments"][k128_key]

            ffn_frac = r16["ffn_ms"] / r16["full_ms"] if r16["full_ms"] > 0 else 0
            attn_frac = r16["attn_ms"] / r16["full_ms"] if r16["full_ms"] > 0 else 0
            other_frac = r16["other_ms"] / r16["full_ms"] if r16["full_ms"] > 0 else 0

            ffn_r = r128["ffn_ms"] / r16["ffn_ms"] if r16["ffn_ms"] > 0 else 1
            attn_r = r128["attn_ms"] / r16["attn_ms"] if r16["attn_ms"] > 0 else 1
            other_r = r128["other_ms"] / r16["other_ms"] if r16["other_ms"] > 0 else 1

            blended = ffn_frac * ffn_r + attn_frac * attn_r + other_frac * other_r
            measured = r128["full_ms"] / r16["full_ms"] if r16["full_ms"] > 0 else 0

            print(f"\n    L={L}:")
            print(f"      FFN:    {ffn_frac:.1%} × {ffn_r:.2f}x = {ffn_frac * ffn_r:.3f}")
            print(f"      Attn:   {attn_frac:.1%} × {attn_r:.2f}x = {attn_frac * attn_r:.3f}")
            print(f"      Other:  {other_frac:.1%} × {other_r:.2f}x = {other_frac * other_r:.3f}")
            print(f"      Blended:  {blended:.2f}x")
            print(f"      Measured:  {measured:.2f}x")
            print(f"      Error:     {abs(blended - measured):.2f}")

            results["experiments"][f"blended_L={L}"] = {
                "ffn_frac": round(ffn_frac, 3), "attn_frac": round(attn_frac, 3),
                "other_frac": round(other_frac, 3),
                "ffn_ratio": round(ffn_r, 3), "attn_ratio": round(attn_r, 3),
                "other_ratio": round(other_r, 3),
                "blended": round(blended, 3), "measured": round(measured, 3),
            }

    if args.output_json:
        with open(args.output_json, "w") as f:
            json.dump(results, f, indent=2)
        print(f"\nResults saved to {args.output_json}")


if __name__ == "__main__":
    main()
