# DEV_LOG — TPU Speculative Decoding / DFlash

Development log for porting DFlash (block-diffusion speculative decoding) to TPU, integrating into tpu-inference, and validating hardware scaling.

---

## Phase 1: Setup & Architecture

**Contribution principles.** Reuse over rewrite; integrate DFlash into tpu-inference (not a DFlash fork); use z-lab/dflash only as reference.

**TPU setup.** TRC v5 quota in us-east1-d; v4/v5p setup; JAX install.

**tpu-inference architecture.** tpu-inference is the vLLM TPU backend with `flax_nnx` vs TorchAX paths. Boot sequence, platform integration, runner/manager roles, and the speculative_decoding_manager govern execution. Model loader, JAX vs TorchAX, ngram/eagle3 spec decode, and kernel layout structure the codebase. DP/PP/disaggregated setups use a KV connector; support matrices and test layout guide validation.

---

## Phase 2: DFlash Integration Plan

**Integration plan.** Phase 1: wiring. Phase 2: parity. Critical finding: K/V must be concatenated, not added; `concat_dense` vs `additive_legacy`; non-causal attention correctness. Design uses `dflash_concat_attention` and `concat_dense` by default.

**Test and baselines.** Verification stages, baseline extraction, 70% speedup target. File inventory: DFlashProposer, aux hidden states.

**Environment gap.** tpu-inference tests cannot run on bare TPU VM; need Docker; rely on CI. vllm-tpu Docker works; disk exhaustion and draft model architecture resolution issues observed. KV concat fix checklist A1–A8 done/blocked, M1 done; pytest blocked locally. Validation runbook: preflight, scope guard, pytest subset, smoke, eval, baseline comparison.

---

## Phase 3: Parity & Backport

**Backport.** zhongyan_dev → dflash-integration branch; vLLM dflash support; torchax wrapper; smoke runs passing.

**Parity fixes.** Rejection-aware trimming; aux capture after layer; chat-template parity. Acceptance collapse investigated. Environment playbook: Python 3.11, Docker, disk, permission cleanup; ops vs algorithm health.

**KV cache.** Phase 1 context buffer: τ≈2.38. Phase 2 KV cache failed initially; noise K/V not cached; num_speculative_tokens=15.

---

## Phase 4: Gap Analysis & Architecture

**GPU vs TPU gap.** DynamicCache vs JAX; flash_attention kernel; context accumulation; static KV cache path.

**Phase 3 flash attention KV cache.** On-device KV plus flash_attention; τ 2.24; JIT retracing; placeholder fix applied.

**Architectural verification.** τ≈2.365 stable; manual attention, position scheme, no-cache all identical to reference; implementation judged correct. Acceptance rate investigation: 13 checks passed; cache crop fix applied; τ=2.365 baseline; gap vs paper attributed to checkpoint/benchmark differences.

---

## Phase 5: Breakthrough & Standalone

**Seq_len inflation fix.** Root cause: `attn_metadata.seq_lens` was inflated; switch to `num_tokens_no_spec`. **Result: τ 2.49→4.48; speedup 1.30×→2.31×.** Largest single gain in the project.

**Standalone benchmark.** Standalone JAX benchmark (vLLM-independent): **τ=6.67 (≈94% of GPU)**; Math500 τ=8.72.

**Replication complete.** DFlash TPU replication done. Standalone τ=6.67; vLLM τ=4.48; no vLLM changes needed for tpu-inference PR.

**Output quality.** 4/8 bit-exact; 7/8 same answer; bf16 numerical differences acceptable.

---

## Phase 6: Research Directions & Experiments

**Directions explored.** Dir 1: pipeline gap. Dir 2: iterative refinement; pipeline overhead; stateful drafter fragility.

**Experiments.** Pipeline profiling: 82.8% overhead. Iterative refinement (k-step): τ collapses because model trained on masks. JIT fusion gave small gain; sync barriers overstated overhead; JAX lazy eval pipelines matter. Ablation: verification ~59%; LM head ~1ms; `jax.lax.while_loop` as main optimization target. Seven insights; parallel prediction; target as bottleneck; production gap noted.

**Verification-cost literature.** TriSpec, Sparse Verify, SpecEE—all break output distribution. Layer truncation experiment designed.

---

## Phase 7: Verification Experiments

**Layer truncation.** All 36 target layers required; skipping 1 layer → ~30% τ loss; no truncation headroom.

**Amortized verification.** **Verify(128) ≈ Verify(16) on TPU.** MXU amortization; multi-block draft adds overhead. Flat scaling appears TPU-specific; SmartSpec shows GPU scaling linear.

**Tree speculation.** Negative result; draft overhead dominates.

**TPU-native drafter direction.** K=128 training as path; MXU tile applies to drafter; DFlash K=16 viewed as GPU constraint, not fundamental.

---

## Phase 8: Scaling Validation

**Drafter scaling.** DFlash FFN flat from K=16→128; confirms TPU-native drafter hypothesis.

**GPU comparison.** Plan: run `gpu_matmul_scaling.py` for contrast. Tile boundary experiment: no K=129 jump; memory-bound; flat up to K=256. K=128 training: feasible on PyTorch/GPU; γ tuning; anchors. End-to-end 3.47× on GPU confounded; isolated GPU matmul benchmark needed.

**GPU matmul results.** GPU FFN flat (1.09×); TPU-uniqueness claim weakened; attention differs. Experiment B context scaling: TPU flat at L=64–1024; GPU attention 2.51× at L=1024.

**Mechanism update.** Memory-bound, not tile-bound; attention as differentiator; novelty claims N1–N11 refined.

---

## Phase 9: Literature, V5P & PR

**Literature audit.** All 7 core claims novel across 30+ papers.

**V5P setup and benchmarks.** v5p-8 setup; venv; JAX; Docker; scripts. Results: 3.02× avg speedup; τ=5.42; 94.9% of GPU math tau; v5p ~1.69× faster baseline than v4.

**PR #1868.** DFlash model, proposer, tests, CI. 14 files; 12 review items (license headers, e2e tests, Buildkite CI). Split into 3 PRs; Eagle3 regression addressed; reviewer feedback on tight coupling and PR structure.

**Block size scaling theory.** τ(K) math; geometric acceptance; α, K, throughput.

**H100 verification and thesis pivot.** **H100 shows flat scaling like TPU.** RTX-specific 1.24× step; TPU-uniqueness claim dropped.

**Cross-hardware K-flat summary.** K-flat on datacenter accelerators; RTX step at K=80; no ceiling through K=1024.

---

## Milestones

| Milestone |
|-----------|
| DFlash integration plan |
| K/V concat parity fix |
| Runnable backport |
| Phase 3 flash attention KV cache |
| **Seq_len inflation fix (largest gain)** |
| **Standalone benchmark (94% GPU quality)** |
| **Replication complete** |
| Verification flat at K=16→128 |
| Mechanism: memory-bound, attention differentiator |
| H100 flat — TPU-uniqueness dropped |
| PR #1868 (DFlash model + proposer) |
