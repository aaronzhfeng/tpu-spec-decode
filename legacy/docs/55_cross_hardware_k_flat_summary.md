# Doc 55 — Cross-Hardware K-Flat Verification Summary

**Status:** Complete
**Date:** 2026-03-07

---

## The Finding

Verification latency is invariant to K (number of draft tokens) on all datacenter accelerators tested, for Qwen3-4B at single-batch decode.

---

## Complete Data (K=128 vs K=16 full forward pass)

| Hardware | L=64 | L=256 | L=512 | L=1024 | L=2048 | L=4096 |
|----------|------|-------|-------|--------|--------|--------|
| TPU v4-8 | -- | 0.97x | -- | 0.97x | -- | -- |
| TPU v5p-8 | -- | 1.00x | -- | 1.00x | 0.92x* | -- |
| TPU v5p-8 (V2, original) | -- | 1.02x | -- | 1.01x | 0.91x* | 0.95x* |
| H100 SXM | 1.03x | 1.00x | 1.04x | 1.01x | 1.01x | -- |
| RTX 2000 Ada | -- | 1.24x | 1.23x | 1.24x | 1.24x | -- |

*0.91-0.92x at L=2048 is measurement noise (all values sit within 1.73-1.92ms band)

## K-Ceiling Sweep (TPU v5p, L=256)

| K | K/K=16 ratio |
|---|-------------|
| 16 | 1.00x |
| 128 | 1.00x |
| 256 | 1.02x |
| 512 | 1.00x |
| 1024 | 1.00x |

No ceiling found through K=1024.

## Fine-Grained K-Sweep (L=256)

| K | H100 | RTX 2000 |
|---|------|----------|
| 16 | 1.00x | 1.00x |
| 32 | 1.00x | 1.02x |
| 48 | 1.02x | 1.04x |
| 64 | 1.01x | 1.07x |
| 80 | 1.03x | **1.16x** (step) |
| 96 | 1.02x | 1.18x |
| 112 | 1.01x | 1.21x |
| 128 | 1.02x | 1.24x |

RTX 2000 has a step function at K=80 (tensor core tiling). H100 and TPU have no discontinuities.

## Component Decomposition

| Component | H100 (% / K-ratio) | RTX 2000 (% / K-ratio) |
|-----------|--------------------|-----------------------|
| Attention (full module) | 57% / 1.06-1.07x | 38-41% / 1.07-1.16x |
| FFN (MLP) | 16-17% / 1.00-1.03x | 40-42% / 1.14-1.15x |
| Other (norms, etc.) | 26% / 1.02-1.03x | 19-20% / 1.02-1.03x |

H100: attention-dominated (57%) because tensor cores execute FFN so fast. RTX 2000: balanced split.

## Isolated Matmul Comparison

| Component | H100 | RTX 2000 | TPU v4 |
|-----------|------|----------|--------|
| Target FFN K=128/K=16 | 1.01x | 1.09x | 0.96x |
| DFlash FFN K=128/K=16 | 1.02x | 1.09x | 0.95x |
| Attention Q*K^T (KV=256) | 1.14x | 1.58x | 0.97x |
| Attention Q*K^T (KV=1024) | 1.08x | 2.51x | ~1.0x |

Even isolated attention scaling is much smaller on H100 (1.08-1.14x) than RTX 2000 (1.58-2.51x).

## Draft Model Scaling

| K | H100 (ms) | H100 ratio | TPU v5p ratio |
|---|-----------|-----------|--------------|
| 16 | 2.82 | 1.00x | 1.00x |
| 64 | 2.87 | 1.02x | -- |
| 128 | 2.88 | 1.02x | 1.00x |

Draft forward pass is flat on all hardware (architecture property of parallel generation).

## Why It's Flat

1. Weight loading dominates: ~6.5 GB loaded every forward pass, fixed regardless of K
2. K-dependent compute (attention scores) is 2.1% of total FLOPs at K=128
3. Cross-layer overlap: attention compute executes in parallel with FFN weight loading
4. On datacenter hardware (H100, TPU), this overlap is nearly perfect
5. On workstation hardware (RTX 2000), overlap is partial — 1.24x penalty

Simple per-component roofline is insufficient. The full-layer pipeline behavior matters more than individual component arithmetic intensity.

## Implication

Wider blocks (K=32, 64, 128) are free to verify on all datacenter serving hardware. The only barrier to wider-block speculative decoding is draft quality, not verification cost.
