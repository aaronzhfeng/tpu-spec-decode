# Doc 54 — Verification Plan: V10, V11, V12

**Status:** Planned
**Date:** 2026-03-06

---

## Overview

Three verification experiments to validate the memory-bandwidth explanation for the K-flat property.

| ID | Test | Hardware | Type |
|----|------|----------|------|
| V11 | Arithmetic intensity vs measured scaling | None (analytical) | Paper exercise |
| V10 | GPU component decomposition within forward pass | RTX 2000 Ada | Benchmark |
| V12 | GPU full forward pass vs context length | RTX 2000 Ada | Benchmark |

---

## V11: Arithmetic Intensity Analysis (Analytical)

**Question:** Does the roofline framework correctly predict which components are memory-bound vs compute-bound at each K?

**Method:** For each component (FFN, attention QK^T, attention AV), compute arithmetic intensity at K=16, 64, 128, 256 and compare against ridge points (TPU: 96, H100: 295).

**Script:** `benchmarks/v11_roofline_analysis.py` — pure math, no hardware needed.

**Success criteria:** Predicted regime matches measured scaling direction for all components on both GPU and TPU.

---

## V10: GPU Forward Pass Component Profiling

**Question:** What fraction of GPU forward pass time is FFN vs attention vs other, and does the blended ratio explain the measured 1.24x?

**Method:** Run actual Qwen3-4B forward pass on GPU with CUDA event timing around each component, at K=16 and K=128, at L=256 and L=1024.

**Script:** `benchmarks/gpu_forward_decomposition.py`

**Success criteria:** Blended component ratio (FFN_frac × FFN_scaling + attn_frac × attn_scaling) matches measured 1.24x within 10%.

---

## V12: GPU Full Forward Pass vs Context Length

**Question:** Is the 1.24x GPU penalty constant across context lengths, or does it grow?

**Method:** Run full Qwen3-4B forward pass at K=16 and K=128 across L=256, 512, 1024, 2048.

**Script:** `benchmarks/gpu_verify_context_scaling.py`

**Success criteria:** 1.24x ± 0.1 at all context lengths, or document the actual trend.
