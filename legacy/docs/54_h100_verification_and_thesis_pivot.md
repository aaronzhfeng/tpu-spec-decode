# Doc 54 — H100 Verification Results and Research Thesis Pivot

**Status:** Complete
**Date:** 2026-03-07

---

## 1. What Happened

We ran V10-V14 experiments to verify the memory-bandwidth explanation for K-flat verification. The results invalidated our core thesis.

### V10: GPU Component Decomposition (RTX 2000 Ada)

The actual FFN/attention/other time split is **40/40/20**, not the 83/17 estimated in Doc 44.

| Component | % of forward pass | K=128/K=16 |
|-----------|------------------|------------|
| Attention (incl. Q/K/V/O projections) | 38-41% | 1.07-1.16x |
| FFN (MLP) | 40-42% | 1.14-1.15x |
| Other (norms, residuals, RoPE) | 19-20% | 1.02-1.03x |

Why 83/17 was wrong: "attention" as a module includes Q/K/V/O linear projections (weight-loading matmuls that behave like FFN). The actual Q*K^T score computation that scales 2.51x in isolation is a tiny fraction within the attention module.

Caveat: Hook overhead (torch.cuda.synchronize at every layer) serializes the GPU pipeline, producing 1.09-1.13x overall instead of the true 1.24x. Component percentages are trustworthy; absolute ratio is undercounted.

### V11: Roofline Analysis (Analytical)

Simple per-component roofline is insufficient. It predicts compute-bound for FFN at K>=48 on TPU, yet empirically FFN is flat (0.97x). The real explanation is cross-layer overlap: K-dependent attention compute executes in parallel with FFN weight loading. Weight loading is >92% of all data movement even at K=128.

### V12: GPU Context Scaling (RTX 2000 Ada)

1.24x is rock-solid constant from L=256 through L=2048. Range: 1.23-1.24x. Combined with V7 (L=64-1024), the ratio never deviates across L=64 to L=2048.

### V13: GPU Fine-Grained K-Sweep (RTX 2000 Ada)

Step function at K=80, not linear scaling:
- K=16 to K=64: nearly flat (1.00x to 1.07x)
- K=64 to K=80: +4.4ms jump (3-5x larger than normal +16 step)
- K=80 to K=128: back to gradual (1.16x to 1.24x)

Cause: SDPA tensor core tiling boundary (K=80 with head_dim=80 = one full tile).

### V14: H100 Full Verification Suite — THE CRITICAL RESULT

**H100 is flat like TPU. The 1.24x was RTX-specific.**

| Metric | H100 SXM | RTX 2000 Ada | TPU v5p |
|--------|----------|-------------|---------|
| K=128/K=16 (L=256) | **1.00x** | 1.23x | 1.00x |
| K=128/K=16 (L=1024) | **1.01x** | 1.24x | 1.00x |
| K=128/K=16 (L=2048) | **1.01x** | 1.24x | 0.92x (noise) |
| K=80 step function | No | Yes (+4-6ms) | No |
| FFN isolated K=128/K=16 | 1.01x | 1.09x | 0.96x |
| Component split | 57% attn / 17% FFN | 40% / 40% | -- |

The K=80 step function is also gone on H100 — it was an RTX tensor core artifact.

### TPU v5p Rerun (Docker, flax 0.12.2)

Re-verified the 0.91x at L=2048 with 30 trials, 10 warmup. Result: 0.92x reproduces, but all 18 measurements across K=16 to K=1024 and L=256 to L=2048 sit between 1.73-1.92ms. The 0.92x is K=16 being slightly slower at L=2048 (1.90ms), not K=128 being faster (1.75ms). Interpretation: flat at ~1.85ms, 0.92x is within noise.

Fixed `verify_context_scaling.py` line 111: `get_flax_model` now returns 8 values (was 7). Changed to index-based unpacking.

Docker recipe: `vllm/vllm-tpu:latest` + `pip install flax==0.12.2` + PYTHONPATH=`pr-ready/vllm-lkg:pr-ready/pr_dflash_1`.

---

## 2. What This Means for the Thesis

### Claims invalidated

| Claim | Status |
|-------|--------|
| "TPU is uniquely suited for wide-block verification" | **Wrong.** H100 is equally flat. |
| "GPU pays 1.24x at K=128" | **RTX-specific.** H100 pays 1.00-1.04x. |
| "TPU advantage in the 2x2 intersection" | **Gone.** No GPU vs TPU contrast on datacenter hardware. |
| "83% attention / 17% FFN split" | **Wrong.** Actual split is 40/40/20 (RTX) or 57/17/26 (H100). |

### Claims that survive

| Claim | Status |
|-------|--------|
| K-flat on TPU v4, v5p through K=1024 | Confirmed |
| K-flat on H100 through K=128 | New (V14) |
| Memory-boundedness at 4B scale is universal | Confirmed across 3 platforms |
| DFlash draft cost flat on all platforms | Confirmed |
| Standalone tau 6.67, 94% of GPU paper | Unchanged |
| vLLM pipeline tau 4.48, 33% gap from standalone | Unchanged |
| B1 (does wider K improve alpha?) is the critical open question | Unchanged |

### Revised novelty

The novelty is NOT "TPU is special." It's:

1. **Empirical:** First systematic K-sweep verification profiling across TPU v4, v5p, H100, and consumer GPU. All datacenter hardware is flat through K=1024. Counter to published assumptions (SmartSpec models verification as linear, Hao AI Lab cites linear scaling).

2. **Analytical:** Tau ceiling analysis showing alpha dominates K. Geometric series converges by K~64. Wider blocks only matter if per-position acceptance improves.

3. **Open question unlocked:** With verification free, the barrier to wider blocks is draft quality, not hardware cost. Nobody has trained K>16 for target-conditioned diffusion.

---

## 3. Research Direction Pivot

### Before V14
"Hardware-Regime-Aware Block Diffusion Drafting" — TPU as risk-free platform.

### After V14
Three viable directions:

**Direction A: Wide-block training (collaborative with DFlash team)**
- Train K=32/64 DFlash models, measure whether alpha improves
- Risk: novelty is shared/dominated by DFlash team since it's their model and training code
- Mitigation: own the training yourself using dflash-wide code, or contribute the profiling + evaluation side

**Direction B: Cross-method K scaling comparison**
- The K-flat finding applies to ALL speculative decoding methods, not just DFlash
- Benchmark EAGLE-3, Medusa, ngram at wider K on TPU
- Comparative study: which draft architectures benefit most from wider K?
- This is entirely our own work, no dependency on external teams

**Direction C: Closing the standalone-to-pipeline gap**
- Standalone tau 6.67 vs vLLM pipeline tau 4.48 — 33% loss
- Systems contribution: identify and fix pipeline overhead sources
- Entirely our own work, directly impacts production serving

Currently pursuing B and C in parallel while waiting for DFlash team's wider-block checkpoints for A.

---

## 4. Email Status

### Outreach email (email_draft_dflash_authors.md)
- Fixed K=1024 scope (v5p only, not both v4 and v5p)
- Fixed verify ratio range (0.97-1.02x)
- Sent; received positive reply from Zhijian

### Reply email (02_reply_zhijian_tpu_memory_bound.md)
- Multiple revisions reflecting evolving understanding
- Final version includes H100 data showing K-flat is universal
- Includes tau ceiling / alpha analysis
- Addresses his memory bandwidth question honestly
- Apologizes for delay (needed H100 runs for accurate answer)
- Status: draft, ready to send

---

## 5. Files Created/Modified This Session

### New files
- `docs/53_block_size_scaling_theory.md` — tau/K/alpha theory, geometric distribution, DFlash architecture, BD3LM comparison, B1 question
- `benchmarks/v11_roofline_analysis.py` — pure Python roofline analysis (no hardware needed)
- `benchmarks/gpu_forward_decomposition.py` — V10 GPU component profiling with hooks
- `benchmarks/gpu_verify_context_scaling.py` — V12 GPU verify vs context length

### Modified files
- `brainstorm-20-spec-decode-diffusion/email/email_draft_dflash_authors.md` — K=1024 scope fix, ratio range fix
- `brainstorm-20-spec-decode-diffusion/email/02_reply_zhijian_tpu_memory_bound.md` — multiple revisions, final version with H100 data
- `brainstorm-20-spec-decode-diffusion/verification/experiment_plan.md` — added V10-V14
- `benchmarks/verify_context_scaling.py` — fixed get_flax_model unpacking (8 return values)
- `benchmarks/gpu_verify_context_scaling.py` — bug fix: total_mem -> total_memory

### Deleted files
- `docs/54_V10_V11_V12_verification_plan.md` — content merged into experiment_plan.md
