# Doc 45: Literature Novelty Audit — All Core Claims Survive

## Status: CLEAR — All 7 novelty claims confirmed novel across 30+ papers

---

## 1. Purpose

Before committing to proposal v6 and training experiments, we conducted an exhaustive literature sweep to verify that none of our core claims are anticipated by prior work. Four parallel search agents checked every relevant paper in the speculative decoding, diffusion drafting, TPU inference, and hardware-aware ML space.

**Result: All claims are novel. Zero full overlaps found.**

---

## 2. Papers Checked (30+)

### Speculative Decoding — Core
- DFlash (arXiv:2602.06036) — block diffusion drafter, K≤16
- EAGLE-3 (arXiv:2503.01840) — autoregressive drafter
- SmartSpec (arXiv:2406.14066, ICLR 2025) — GPU verification cost linear
- Sequoia (arXiv:2402.12374) — hardware-aware tree topology
- DeepMind Speculative Sampling (arXiv:2302.01318) — original TPU spec decode

### Diffusion Drafters
- SpecDiff-2 (arXiv:2511.00606) — diffusion alignment, γ=16/32
- SpecDiff (arXiv:2408.05636, NAACL 2025) — diffusion decoding, γ≤45
- DEER (arXiv:2512.15176) — mask-and-predict, R≤96
- DiffuSpec (arXiv:2510.02358) — training-free, k≤100
- DART (arXiv:2601.19278) — parallel logits, K=8
- Spiffy (arXiv:2509.18085) — uses K=32 at inference
- FailFast (arXiv:2512.20573) — adaptive drafting
- BD3LM (arXiv:2503.09573, ICLR 2025 Oral) — block diffusion LM, K≤16

### Verification Optimization
- Sparse Verification (arXiv:2512.21911) — attention/FFN/MoE decomposition
- HiSpec (arXiv:2510.01336) — hierarchical verification
- Block Verification (arXiv:2403.10444)
- MatX SD+NSA — blockwise sparse attention for verification

### Hardware / Roofline / TPU
- "Systematic Characterization of LLM Inference on GPUs" (arXiv:2512.01644)
- MagicDec (arXiv:2408.11049) — context-length-dependent verification
- SPIRe (arXiv:2504.06419) — TPU spec decode at K=4
- PACER (arXiv:2602.01274) — adaptive block length
- JAX Scaling Book (jax-ml.github.io/scaling-book)

### Other
- Self Speculative Decoding for Diffusion LLMs (arXiv:2510.04147)
- LLaDA 2.0 (arXiv:2512.15745) — 100B diffusion LM
- SpecEE, TriSpec, SpecInfer, various surveys

---

## 3. Claim-by-Claim Novelty Audit

### N1: TPU Verification Flat K=16→128 at All Context Lengths

| Paper | Overlap? | Detail |
|---|---|---|
| DeepMind (2023) | **Assumed, not measured** | Stated "time to load linear weights dominates compute time, so it is similar for k and 1" on TPU v4. But never measured at K=16→128, never tested context lengths. |
| SmartSpec (ICLR 2025) | **Opposite finding on GPU** | Modeled GPU verify as linear: T = α·N_context + γ·N_query + δ. Complements our claim. |
| SPIRe (MatX) | **TPU, but K=4 only** | Used Cloud TPU for speculative decoding but only tested K=4. No verification cost analysis. |
| Sequoia | **Small-n flatness acknowledged** | Noted that "prior work can get away with setting t(n)=1 because optimal n occurs before t(n) grows." Never measured on TPU, never tested K=128. |
| JAX Scaling Book | **Theoretical support** | States TPU decode is memory-bandwidth bound. Not speculative-decoding-specific. |
| All others | No overlap | No TPU verification measurements exist in any other paper. |

**Verdict: NOVEL.** DeepMind assumed constant verify on TPU; we measure it across K=16–128 and L=64–1024. SmartSpec measured the GPU counterpart. Nobody else has TPU verification data.

---

### N2: FFN Flat on Both Hardware, Attention Is the Sole Differentiator

| Paper | Overlap? | Detail |
|---|---|---|
| arXiv:2512.01644 | **CLOSEST OVERLAP** | Shows FFN is O(1) per step, attention is O(n) during *standard decode* on GPU (A100, Jetson Orin). This is the known GPU characterization. |
| Sparse Verification | **Partial FLOPs decomposition** | Provides separate FLOP formulas for attention, FFN, MoE during verification. But does not measure runtime scaling with K, does not compare hardware. |
| MatX SD+NSA | **Identifies attention as bottleneck** | Notes attention operational intensity drops during verification. GPU only. |
| All others | No overlap | No paper decomposes FFN vs attention during verification and compares across GPU and TPU. |

**Verdict: NOVEL with careful positioning.** The observation that "FFN is O(1), attention is O(n) during decode" is known from arXiv:2512.01644. Our novelty is:
1. Measuring this during *verification* (K>1 query tokens), not standard decode (K=1)
2. *Cross-hardware* comparison (GPU vs TPU)
3. Showing attention is flat on TPU but not GPU — the *hardware divergence*

**Action: Cite arXiv:2512.01644 prominently.** Frame as: "They established FFN O(1)/attention O(n) for standard decode on GPU. We extend this to verification at K=16–128 and show the pattern holds on GPU but *not on TPU*, where attention is also flat."

---

### N3: K=16 Is a GPU Hardware Ceiling, Not a Model Optimum

| Paper | Overlap? | Detail |
|---|---|---|
| DFlash | **Observes symptom, not cause** | Notes "large blocks increase verification cost" but treats K=16 as a quality-vs-cost tradeoff. Does not identify it as GPU-imposed. |
| SpecDiff-2 | **Same symptom** | Finds "degradation for large gamma" on GPU. Attributes to model quality, not hardware. |
| PACER | **Treats K as model parameter** | Optimizes K adaptively based on context difficulty, not hardware. |
| Sequoia | **Hardware-dependent tree size** | Shows optimal tree varies by hardware (A100 vs L40). Closest conceptual neighbor. But addresses tree topology, not block width, and only GPU. |
| All others | No overlap | No paper reframes K as a hardware parameter. |

**Verdict: NOVEL.** DFlash and SpecDiff-2 observe the constraint but attribute it to model/serving tradeoffs, not hardware. Our reframing — that K=16 is GPU-imposed and the optimal K changes with hardware — is new. Cite Sequoia as conceptually related (hardware-aware optimization) but in a different dimension (tree topology on GPU, not block width on TPU).

---

### N4: Diffusion + TPU Intersection (2×2 Matrix)

| Paper | Overlap? | Detail |
|---|---|---|
| All 30+ papers | **Zero overlap** | No paper combines diffusion drafting with TPU analysis. The diffusion drafter community (DFlash, SpecDiff-2, DEER, DART) is exclusively GPU. The TPU inference community does not use diffusion drafters. |

**Verdict: NOVEL — our strongest claim.** The 2×2 matrix (AR vs diffusion) × (GPU vs TPU) showing only one viable quadrant for K=128 has no precedent.

---

### N5: No Block-Diffusion Drafter Trained at K≥64

| Paper | K trained at | Architecture | Relevant? |
|---|---|---|---|
| DFlash | K=8, 10, 16 | Block diffusion, target-conditioned | Most relevant — max K=16 |
| SpecDiff-2 | γ=32 (DiffuCoder) | Standalone 7B diffusion model | **Different architecture** — not a lightweight target-conditioned drafter |
| DEER | Masking R≤96 | Mask-and-predict | **Different architecture** — not block diffusion |
| BD3LM | K=4, 8, 16 | Block diffusion LM (not speculative) | Not a speculative decoding drafter |
| DiffuSpec | Training-free | Uses existing Dream-7B | No training |
| LLaDA 2.0 | K=64 during warmup | 100B full LM pretraining | Not a drafter, not spec decode |
| All others | K≤16 | Various | No K>16 training |

**Verdict: NOVEL with refinement.** Two nuances:
1. **SpecDiff-2 (γ=32):** Different architecture — standalone 7B diffusion model, not a lightweight target-conditioned block-diffusion drafter like DFlash. The "γ" in SpecDiff-2 is a draft window, not a block diffusion block size.
2. **DEER (R≤96):** Different architecture — mask-and-predict, not block diffusion. Architecturally distinct from DFlash.

**Recommended phrasing:** "No prior work has trained a target-conditioned block-diffusion drafter (DFlash architecture) at K≥64." This is bulletproof.

---

### N6: Diffusion Draft Cost Near-Flat on Both GPU (1.22×) and TPU (0.95×)

| Paper | Overlap? | Detail |
|---|---|---|
| All papers | No overlap | No paper measures DFlash draft forward pass scaling with K. DFlash paper reports total speedups but not draft-cost-vs-K. |

**Verdict: NOVEL.** The measurement that diffusion draft cost is architecture-insensitive to K (near-flat on both hardware) is new. Confirms draft K-insensitivity is a property of parallel generation, not hardware.

---

### N7: TPU Advantage Grows With Context Length

| Paper | Overlap? | Detail |
|---|---|---|
| MagicDec | **Related on GPU side** | Shows verification cost grows with context on GPU. Does not test TPU. |
| SmartSpec | **Linear model on GPU** | N_context term in their formula. GPU only. |
| All others | No overlap | No cross-hardware context-length comparison for verification. |

**Verdict: NOVEL.** The specific finding that TPU advantage over GPU grows from ~1.08× at L=64 to ~2.3× at L=1024 has no precedent.

---

## 4. Risk Summary

| Claim | Risk | Mitigation |
|---|---|---|
| N1: TPU verify flat | LOW | DeepMind assumed it; we measure it |
| N2: FFN/attention decomposition | **MODERATE** | Cite arXiv:2512.01644; differentiate as verification + cross-hardware |
| N3: K=16 is GPU ceiling | LOW | DFlash/SpecDiff-2 observe symptom, not cause |
| N4: Diffusion+TPU intersection | **VERY LOW** | Zero overlap in 30+ papers |
| N5: First K≥64 block-diffusion drafter | LOW (with refinement) | Scope to "target-conditioned block-diffusion" to exclude SpecDiff-2/DEER |
| N6: GPU draft also flat | LOW | No prior measurement |
| N7: Context-length advantage | LOW | No cross-hardware comparison exists |

---

## 5. Critical Papers to Cite

### Must-Cite (Differentiation Required)

| Paper | Why | How to differentiate |
|---|---|---|
| **arXiv:2512.01644** "Systematic Characterization of LLM Inference on GPUs" | Shows FFN O(1) / attention O(n) for standard decode on GPU | We extend to verification (not decode), add TPU, show cross-hardware divergence |
| **SmartSpec** (ICLR 2025) | GPU verification linear in K | Direct complement — we provide TPU counterpoint |
| **DFlash** (arXiv:2602.06036) | Starting point, K≤16 | We show K=16 is GPU-imposed; extend to K=128 on TPU |
| **Sequoia** (arXiv:2402.12374) | Hardware-aware spec decode optimization | They optimize tree topology on GPU; we optimize block width on TPU |
| **SpecDiff-2** (arXiv:2511.00606) | Closest to wide-block diffusion (γ=32) | Different architecture (standalone 7B); found degradation on GPU |
| **DEER** (arXiv:2512.15176) | Achieves K=32 acceptance | Different architecture (mask-and-predict); GPU only |

### Should-Cite (Context)

| Paper | Why |
|---|---|
| **DeepMind Speculative Sampling** (2023) | Assumed constant verify on TPU — we measure it |
| **SPIRe** (MatX, arXiv:2504.06419) | Only other paper using TPU for spec decode (K=4 only) |
| **JAX Scaling Book** | Theoretical basis for TPU memory-bound decode |
| **MagicDec** (arXiv:2408.11049) | Context-length-dependent verification on GPU |
| **Sparse Verification** (arXiv:2512.21911) | FLOPs decomposition of verification components |
| **PACER** (arXiv:2602.01274) | Adaptive K — treats as model parameter, we treat as hardware parameter |
| **HiSpec** (arXiv:2510.01336) | Verification bottleneck alternative (early-exit) |

---

## 6. Recommended Refinements to Proposal v6

Based on this audit:

1. **Add arXiv:2512.01644 to related work.** Cite their FFN O(1)/attention O(n) finding for GPU decode. Explicitly state we extend to verification and cross-hardware comparison.

2. **Refine N5 phrasing.** From "no diffusion drafter trained at K>16" to "no target-conditioned block-diffusion drafter trained at K≥64." Acknowledge SpecDiff-2 γ=32 and DEER R=96 as different architectures.

3. **Add SPIRe to related work.** It's the only other TPU speculative decoding paper (K=4, no verification cost analysis). Acknowledging it strengthens our positioning as the first comprehensive TPU SD analysis.

4. **Add DEER to related work.** Achieves K=32 acceptance with diffusion on GPU. Supporting evidence that wide-block diffusion can produce quality drafts.

5. **Strengthen N4 (intersection) in the abstract.** This is our most defensible claim — zero overlap across all papers. Make it prominent.

---

## 7. The Striking Gap

**Not a single paper in the speculative decoding literature runs comprehensive experiments on TPU.** SPIRe (MatX) barely touches it at K=4. DeepMind's 2023 paper assumed constant verify but never measured it. The entire field is GPU-centric (A100, H100, H200, B200, H800, L40).

This means:
- We are first movers on TPU-specific speculative decoding analysis
- There is no prior TPU verification cost data to conflict with ours
- The "diffusion + TPU" intersection is an entirely unoccupied space

---

*Created: February 27, 2026*
*Status: Literature audit complete — all claims novel, one moderate-risk claim (N2) requires careful positioning*
*Method: Four parallel search agents, 30+ papers checked, claim-by-claim analysis*
*Action items: Cite arXiv:2512.01644, refine N5 phrasing, add SPIRe and DEER to related work*
*Builds on: Proposal v6, Docs 33–44*
