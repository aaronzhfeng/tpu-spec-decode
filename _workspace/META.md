# Project Meta: DFlash Speculative Decoding on TPU

**Project:** TPU-native block-diffusion speculative decoding
**Team:** Aaron Feng, Zhongyan Luo, Son Nguyen, Andy Huang
**Advisors:** Hao Zhang, Yiming Zhao (UC San Diego)
**Collaborators:** Google TPU Inference team (Yarong Mu, Chengji Yao)
**Period:** Fall 2025 — Spring 2026

---

## 1. One-Line Summary

Ported DFlash block-diffusion speculative decoding to TPU, achieving 3.13x average speedup across 9 benchmarks; discovered TPU verification is K-flat (0.97x at K=128 vs 1.24x on GPU), enabling risk-free wide-block drafting.

---

## 2. Key Results

### 2.1 DFlash TPU Inference (PR #1868)

| Dataset | Category | Tau | Speedup | Baseline TPOT (ms) | DFlash TPOT (ms) |
|---------|----------|-----|---------|---------------------|-------------------|
| math500 | math | 8.80 | 5.72x | 8.02 | 1.40 |
| aime24 | math | 6.48 | 3.98x | 7.69 | 1.93 |
| aime25 | math | 6.14 | 3.35x | 6.85 | 2.05 |
| gsm8k | math | 5.40 | 3.17x | 7.32 | 2.31 |
| humaneval | code | 5.76 | 3.53x | 7.70 | 2.18 |
| mbpp | code | 6.16 | 2.77x | 7.31 | 2.64 |
| mt-bench | chat | 3.87 | 2.36x | 7.20 | 3.05 |
| alpaca | chat | 2.86 | 1.65x | 6.72 | 4.08 |
| swe-bench | code | 3.35 | 1.60x | 6.85 | 4.27 |
| **Average** | | **5.42** | **3.13x** | **7.30** | **2.66** |

- **Hardware:** TPU v5p-8 (4 chips, 8 cores)
- **Models:** Qwen3-4B (target) + z-lab/Qwen3-4B-DFlash-b16 (draft)
- **Config:** greedy, max_new_tokens=256, max_model_len=2048, 8 samples/dataset, 1 warmup

### 2.2 K-Flat Verification Property

| Metric | GPU (RTX 2000 Ada) | TPU v4-8 | TPU v5p-8 |
|--------|-------------------|----------|-----------|
| K=128/K=16 full verify | 1.24x | 0.97x | 0.97x |
| Context-dependent? | No | No | No |
| K-flat through | K=128 | K=1024 | K=1024 |
| L-flat through | L=1024 | L=1024 | L=4096 |

### 2.3 Cross-Hardware Comparison (vs GPU Paper)

| Dataset | V5P Tau | GPU Paper Tau | Parity | V5P Speedup | GPU Paper Speedup |
|---------|---------|---------------|--------|-------------|-------------------|
| gsm8k | 5.40 | 6.53 | 82.7% | 3.00x | 5.15x |
| math500 | 8.80 | 7.84 | 112.2% | 4.93x | 6.09x |
| aime24 | 6.48 | 7.28 | 89.1% | 3.57x | 5.68x |
| aime25 | 6.14 | 6.63 | 92.5% | 3.39x | 5.21x |

### 2.4 V4 vs V5P Improvement

- Baseline TPOT: 1.69x faster on V5P (10.7ms -> 6.3ms)
- DFlash TPOT: 1.37-2.48x faster on V5P (varies by dataset)
- Tau: stable across hardware generations

---

## 3. Deliverables

### 3.1 Code — PR #1868

**URL:** https://github.com/vllm-project/tpu-inference/pull/1868
**Branch:** `pr_dflash_1` on `aaronzhfeng/tpu-inference`
**Stats:** 14 files changed, +2277/-10 lines

New files:
- `tpu_inference/models/jax/dflash.py` (599 lines) — DFlash JAX draft model
- `tpu_inference/models/jax/qwen3_dflash.py` (529 lines) — Qwen3-specific DFlash variant
- `tpu_inference/layers/common/dflash_attention_interface.py` (129 lines) — concat attention kernel
- `tpu_inference/spec_decode/jax/dflash.py` (374 lines) — DFlashProposer
- `tests/models/jax/test_qwen3_dflash.py` (51 lines) — layer ID tests
- `tests/models/jax/test_qwen3_dflash_attention.py` (289 lines) — attention tests
- `tests/spec_decode/test_dflash.py` (68 lines) — proposer tests
- `.buildkite/features/Speculative_Decoding-_DFlash.yml` (68 lines) — CI pipeline

Modified files:
- `tpu_inference/models/common/model_loader.py` (+2)
- `tpu_inference/models/jax/qwen3.py` (+85)
- `tpu_inference/runner/tpu_runner.py` (+3)
- `tpu_inference/runner/speculative_decoding_manager.py` (+34/-10)
- `tpu_inference/runner/kv_cache_manager.py` (+6/-5)
- `tests/e2e/test_speculative_decoding.py` (+50)

**Local path:** `/home/aaronfeng/tpu-spec-decode/pr-ready/pr_dflash_1/`

### 3.2 Research Proposal (v8, current)

**Path:** `/home/aaronfeng/tpu-spec-decode/brainstorm-20-spec-decode-diffusion/proposals/proposal_v8.md`
**Title:** Hardware-Regime-Aware Block Diffusion Drafting in the Memory-Bound Verification Regime
**Target venue:** ICML / NeurIPS 2026

Three contributions:
1. SD-specific regime map: verification scaling with K and L on TPU vs GPU
2. TPU as risk-free platform for wide-block diffusion drafting
3. Hardware-aware block diffusion training study at K=32/64/128

Previous versions: v1-v7 in same directory (full revision history).

### 3.3 Capstone Report

**Path:** `/home/aaronfeng/tpu-spec-decode/capstone_report/`
- `report.pdf` — compiled final report
- `report.tex` — LaTeX source
- `table/` — standalone_results, cross_comparison, eagle3_comparison, quality_check
- `figure/` — somefig1.pdf, somefig2.png, somefig3.png
- `reference.bib` — bibliography

### 3.4 Presentations

**Path:** `/home/aaronfeng/tpu-spec-decode/slides/`
- `week7_new/main.pdf` — latest slide deck
- `week7/main.pdf` — original version
- LaTeX sources in both directories

**Path:** `/home/aaronfeng/tpu-spec-decode/results/`
- `report.pptx` (2.7M) — results presentation
- `report.pdf` (376K) — results report

### 3.5 DFlash-Wide (K=128 Research Direction)

**Path:** `/home/aaronfeng/tpu-spec-decode/dflash-wide/`
- `training/train.py`, `dataset.py`, `loss.py` — K=128 training code
- `dflash/model/dflash.py` — model implementation
- `dflash/benchmark.py` — benchmark harness
- `dflash/assets/dflash_system.png` — system architecture diagram
- `dflash/assets/speedup.png` — speedup curves
- `dflash/assets/dflash_results.png` — results visualization
- `benchmarks/gpu_scaling.py` — GPU scaling analysis

---

## 4. Experimental Data

### 4.1 V5P Benchmark Results (9 datasets)

**Path:** `/home/aaronfeng/tpu-spec-decode/results/v5p/`
- `standalone_gsm8k.json` (81K)
- `standalone_math500.json` (49K)
- `standalone_aime24.json` (165K)
- `standalone_aime25.json` (101K)
- `standalone_humaneval.json` (71K)
- `standalone_mbpp.json` (112K)
- `standalone_mt-bench.json` (184K)
- `standalone_alpaca.json` (125K)
- `standalone_swe-bench.json` (222K)
- `standalone_all_benchmarks.csv` — aggregate summary
- `standalone_vs_gpu_paper.csv` — GPU comparison
- `standalone_vs_v4.csv` — V4 comparison

### 4.2 V4 Benchmark Results (9 datasets + profiling)

**Path:** `/home/aaronfeng/tpu-spec-decode/results/v4/`
- `standalone_*.json` — 9 benchmark result files
- `profiling_gsm8k.json` — per-sample profiling
- `quality_gsm8k.json` — token-by-token quality
- `refinement_gsm8k.json` — refinement steps analysis
- `fused_gsm8k.json` — fused vs unfused
- `verify_context_scaling.json` — K-flat property (K={16,64,128}, L={64,256,512,1024})
- `gpu_matmul_scaling.json` — GPU matmul scaling
- `gpu_matmul_scaling_extended.json` — extended GPU profiling
- `gpu_draft_speed.json` — GPU draft model speed
- 9 CSV summary files (standalone, cross_comparison, eagle3, quality, vllm_pipeline, acceptance)
- `generate_csvs.py` — CSV generation script
- `report.md` — V4 summary report

### 4.3 V5P Acceptance Studies

**Path:** `/home/aaronfeng/tpu-spec-decode/results/`
- `v5_acceptance_gsm8k.json` (192K) — 20-sample study, tau=5.92
- `v5_acceptance_math500.json` (260K) — 20-sample study, tau=8.68
- `verify_context_scaling.json` (8K) — K-sweep K={16..1024} on v5p

### 4.4 PR Validation Runs

**Path:** `/dev/shm/dflash-test-outputs/`
- `v5p-pr/v5p_pr_*.json` — 9 files, pr_dflash branch validation
- `v5p-pr1/v5p_pr1_*.json` — 10 files, pr_dflash_1 branch validation (includes gsm8k retest)
- `v5p_standalone_*.json` — 9 files, standalone baselines

### 4.5 Verification Test Runs

**Path:** `/home/aaronfeng/tpu-spec-decode/verification/outputs/`
- `contribution/contrib_20260211_*/` — multiple test runs (Feb 11)
- `contribution/contrib_20260210_*/` — multiple test runs (Feb 10)
- Each contains: `env_snapshot.json`, `run_manifest.json`, `summaries/overall.json`, `prompt_records/*.jsonl`

---

## 5. Benchmark & Test Scripts

### 5.1 Benchmark Scripts

**Path:** `/home/aaronfeng/tpu-spec-decode/benchmarks/`
- `standalone_dflash.py` — main standalone benchmark (917 lines)
- `ablation_study.py` — verification cost decomposition (verify=59%, draft=2%)
- `amortized_verification.py` — K-flat property (K=16->256)
- `layer_truncation.py` — layer-wise importance (skip 1/36 = -30% tau)
- `tree_speculation.py` — tree-based drafting
- `drafter_scaling.py` — draft model scaling
- `iterative_refinement.py` — refinement iterations
- `fused_dflash.py` — fused kernel variant
- `gpu_draft_speed.py` — GPU drafting speed
- `gpu_matmul_scaling.py` — GPU matmul scaling curves
- `pipeline_profiling.py` — full pipeline profiling
- `verify_context_scaling.py` — context scaling (L and K sweep)
- `benchmark_block_sizes.py` — block size sensitivity
- `run_all_v5p_pr.sh` — batch run script for all 9 datasets

### 5.2 Test Shell Scripts

**Path:** `/home/aaronfeng/tpu-spec-decode/tests/`
- `standalone_benchmark.sh`, `ablation_study.sh`, `amortized_verification.sh`
- `layer_truncation.sh`, `tree_speculation.sh`, `drafter_scaling.sh`
- `pipeline_profiling.sh`, `verify_context_scaling.sh`
- `benchmark.sh`, `smoke.sh`, `fused_benchmark.sh`, `iterative_refinement.sh`
- `tpu_sanity.sh`, `compare.sh`, `cleanup.sh`, `preflight.sh`
- `configs/` — 8 JSON test configs (smoke_greedy, smoke_stochastic, benchmark_math/code/chat/full, benchmark_math_dflash_only, eagle3_llama_math)
- `lib/common.sh`, `lib/docker_run.sh` — shared utilities

---

## 6. Documentation (52 docs)

**Path:** `/home/aaronfeng/tpu-spec-decode/docs/`

### Setup & Architecture (docs 00-06)
- `00_1_research_overview.md` (22K) — complete research scope
- `00_2_conceptual_overview.md` (17K) — conceptual foundations
- `01_tpu_setup_guide.md` (13K) — TPU provisioning
- `02_contribution_principles.md` (11K) — contribution guidelines
- `03_tpu_inference_repo_deep_dive.md` — repo structure
- `04_tpu_inference_runtime_architecture.md` — runtime design
- `05_tpu_inference_models_and_kernels.md` — model/kernel architecture
- `06_tpu_inference_distributed_operations_and_validation.md` — distributed ops

### DFlash Integration (docs 07-19)
- `07_dflash_integration_plan.md` (13K) — integration strategy
- `08_dflash_test_plan_and_reported_baselines.md` — baseline tests
- `09_tpu_inference_dflash_change_log.md` — code changes
- `10_test_environment_gap.md` — environment analysis
- `10.1_dflash_kv_concat_parity_fix_design.md` — KV cache fix
- `11.1_dflash_kv_concat_fix_execution_checklist.md` — execution plan
- `12_dflash_validation_runbook.md` — validation steps
- `13_dflash_integration_backport_report.md` — backport report
- `14_dflash_parity_findings_and_remediation.md` — parity gaps
- `15_dflash_runtime_behavior_and_environment_playbook.md` — runtime guide
- `16_dflash_kv_cache_progress_and_findings.md` — KV cache progress
- `17_gpu_vs_tpu_dflash_gap_analysis.md` — GPU/TPU gap analysis
- `19_gpu_tpu_architectural_gap_analysis.md` (32K) — deep architectural analysis

### Experiments & Results (docs 20-45)
- `20_acceptance_rate_investigation_report.md` (22K) — acceptance rates
- `21_seq_len_inflation_fix.md` — sequence length fix
- `22_standalone_benchmark.md` — benchmark methodology
- `23_dflash_tpu_replication_complete.md` — replication milestone
- `25_research_directions.md` — future work
- `26_experiment_design.md` — experiment framework
- `27_experiment_results.md` — results summary
- `29_ablation_study_results.md` — ablation (verify=59%, draft=2%)
- `30_insights_and_open_questions.md` — key findings
- `31_verification_cost_literature.md` — literature on verify cost
- `32_layer_truncation_results.md` — layer truncation (skip 1/36 = -30% tau)
- `33_amortized_verification_results.md` — K-flat property (K=16->256)
- `35_tree_speculation_results.md` — tree drafting
- `36_tpu_native_drafter_direction.md` — K=128 proposal
- `37_drafter_scaling_results.md` — drafter scaling
- `38_gpu_comparison_plan.md` — GPU benchmark plan
- `39_tile_boundary_experiment.md` — TPU tile boundary
- `40_k128_training_implementation_assessment.md` — K=128 feasibility
- `41_gpu_evidence_gap_and_corrected_plan.md` — GPU corrections
- `42_gpu_matmul_results.md` — GPU matmul results
- `43_experiment_b_context_scaling_results.md` — context scaling
- `44_renewed_analysis_mechanism_and_novelty.md` — novelty assessment
- `45_literature_novelty_audit.md` — audit against 55 papers
- `dflash_paper.md` — DFlash paper summary

### PR Documentation (docs 48-52)
- `48_pr_dflash_1_review.md` — PR readiness review (14 issues found/fixed)
- `49_pr_dflash_1_changes.md` — PR implementation details
- `50_pr_dflash_1_progress.md` — PR progress (problems and fixes)
- `51_pr_dflash_1_description.md` — PR description (PR 1 of 3)
- `52_pr_dflash_1_reply.md` — PR response notes

---

## 7. Research Materials

### 7.1 Literature

**Path:** `/home/aaronfeng/tpu-spec-decode/brainstorm-20-spec-decode-diffusion/literature/`
- 70+ papers (PDFs + markdown summaries)
- Key papers: SpecInfer, EAGLE1/2/3, DFlash, DART, SpecDiff, BD3LM, LLaDA, Sequoia, HiSpec
- Topics: speculative decoding, discrete diffusion, verification strategies, TPU architecture

**Path:** `/home/aaronfeng/tpu-spec-decode/brainstorm-20-spec-decode-diffusion/`
- `literature.md` (85K) — curated 55-paper index by theme
- `literature_original.md` (15K) — original compilation
- `meta_review.md` (53K) — comprehensive synthesis

### 7.2 Verification Experiments (V1-V9)

**Path:** `/home/aaronfeng/tpu-spec-decode/brainstorm-20-spec-decode-diffusion/verification/`
- `experiment_plan.md` — master plan for V1-V9
- `V1_V2_k_sweep_v5p.md` — K-sweep on v5p, extended L={2048,4096}
- `V3_k_ceiling_v5p.md` — K-ceiling through K=1024
- `V4_roofline_profiling.md` — roofline/MXU utilization
- `V5_context_position_acceptance.md` — context position impact on acceptance
- `V6_batch_size_extrapolation.md` — batch size >1 behavior
- `V7_gpu_full_forward_pass.md` — **CORRECTED**: GPU full verify = 1.24x (not 2.3x from isolated Q×K^T)
- `V8_V9_drafter_decomposition_v5p.md` — FFN/attention decomposition, draft K-sweep

### 7.3 Proposal History (v1-v8)

**Path:** `/home/aaronfeng/tpu-spec-decode/brainstorm-20-spec-decode-diffusion/proposals/`
- `proposal_v1.md` — initial: tile-aware spec decode via MXU
- `proposal_v2.md` — focused: TPU-native K=128 drafter
- `proposal_v3.md` — math/consistency revision
- `proposal_v3.5.md` — intermediate revision
- `proposal_v4.md` — corrected narrative (memory-bandwidth, not tile)
- `proposal_v5.md` — integrated C01-C05 audit feedback
- `proposal_v5.pdf` — compiled v5
- `proposal_v6.md` — assumption audit, novelty inventory, GPU draft data
- `proposal_v7.md` — March novelty audit, block-size quality analysis
- `proposal_v8.md` **(current)** — V7 GPU correction (1.24x), tau ceiling analysis, revised intersection

### 7.4 Communications

**Path:** `/home/aaronfeng/tpu-spec-decode/brainstorm-20-spec-decode-diffusion/email/`
- `email_draft_hao.md` — update to Prof Zhang (v5 access, research progress)
- `email_followup_hao.md` — followup (v5p results, PR ready, K=128 next steps)
- `email_draft_dflash_authors.md` — email to Prof Liu / Jian / Yesheng (training code request)
- `google_channel_dflash_pr.md` — Google channel message (PR announcement)

---

## 8. Infrastructure

### 8.1 Environment Setup

**Path:** `/home/aaronfeng/tpu-spec-decode/preparation/`
- `bootstrap.sh` — one-shot TPU node setup
- `setup_v5p_safe.sh` — V5P environment setup
- `clone_repos.sh` — branch-pinned cloning
- `tpu_sanity_check.py` — hardware verification
- `V5P_ENVIRONMENT.md`, `V5P_SETUP_MANUAL.md` — setup docs

### 8.2 Docker

- Image: `vllm/vllm-tpu:latest`
- Flax pin: `flax==0.12.4`
- PYTHONPATH: `vllm-lkg:pr_dflash_1`
- Model cache: `/dev/shm/hf-cache/`

### 8.3 Hardware Used

| Hardware | Chips | Cores | HBM | Use |
|----------|-------|-------|-----|-----|
| TPU v4-8 | 4 | 8 | 128 GB | initial development, V1-V6 experiments |
| TPU v5p-8 | 4 | 8 | 188 GB | PR validation, V1-V9 experiments |
| NVIDIA RTX 2000 Ada | 1 | - | 16 GB | GPU comparison (V7) |

---

## 9. Visual Assets

### Existing Figures

- `/home/aaronfeng/tpu-spec-decode/dflash-wide/dflash/assets/dflash_system.png` — system architecture diagram
- `/home/aaronfeng/tpu-spec-decode/dflash-wide/dflash/assets/speedup.png` — speedup curves
- `/home/aaronfeng/tpu-spec-decode/dflash-wide/dflash/assets/dflash_results.png` — results visualization
- `/home/aaronfeng/tpu-spec-decode/capstone_report/figure/somefig1.pdf` — figure 1
- `/home/aaronfeng/tpu-spec-decode/capstone_report/figure/somefig2.png` — figure 2
- `/home/aaronfeng/tpu-spec-decode/capstone_report/figure/somefig3.png` — figure 3
- `/home/aaronfeng/tpu-spec-decode/archive/image.png` — documentation image

### Tables Available for Rendering

From `results/v5p/`:
- 9-dataset benchmark table (tau, speedup, TPOT)
- V5P vs GPU paper comparison
- V5P vs V4 comparison
- Per-position acceptance rate histograms (in each JSON)

From `results/v4/`:
- Cross-comparison table (standalone vs vLLM vs GPU paper)
- Eagle3 comparison table
- Quality check table
- Acceptance rate by position CSV

From verification/:
- K-flat property table (GPU 1.24x vs TPU 0.97x)
- Context scaling matrix (K x L)
- Ablation breakdown (verify 59%, draft 2%, overhead 39%)

---

## 10. File Counts

| Category | Files | Size |
|----------|-------|------|
| Documentation (docs/) | 52 | 732K |
| Research papers (literature/) | 70+ | ~200M |
| Proposals (v1-v8) | 11 | 630K |
| Benchmark scripts | 13 .py + 1 .sh | 324K |
| Test scripts + configs | 16 .sh + 8 .json | 132K |
| Result JSONs (all runs) | 56 | 5.8M |
| Result CSVs | 12 | ~10K |
| PR source code (new/modified) | 14 | ~60K |
| Capstone report | 10 | ~400K |
| Slides | 2 PDFs + sources | ~2M |
| Verification runs | 10+ directories | ~5M |
| Visual assets | 7 images | ~1M |
| Email drafts | 4 | ~10K |
| **Total unique assets** | **~250+** | **~215M** |
