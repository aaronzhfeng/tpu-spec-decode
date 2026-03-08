# DFlash Speculative Decoding on TPU

Ported DFlash block-diffusion speculative decoding to TPU (JAX/Flax), achieving 3.13x average speedup across 9 benchmarks. Discovered TPU verification cost is K-flat through K=1024, enabling risk-free wide-block drafting.

**Team:** Aaron Feng, Zhongyan Luo, Son Nguyen, Andy Huang
**Advisors:** Hao Zhang, Yiming Zhao (UC San Diego)
**Collaborators:** Google TPU Inference team (Yarong Mu, Chengji Yao)

## Upstream PRs

- **PR #1868:** [[Spec Decoding] Add DFlash model and proposer](https://github.com/vllm-project/tpu-inference/pull/1868) — model, proposer, unit tests
- **PR #1869:** [Spec Decoding] Integrate DFlash into speculative decoding pipeline
- **PR #1870:** [Spec Decoding] Add DFlash e2e tests and Buildkite CI

## Key Results (TPU v5p-8)

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

Models: Qwen3-4B (target) + z-lab/Qwen3-4B-DFlash-b16 (draft), greedy decoding.

### K-Flat Verification

Verification cost is flat from K=16 through K=1024 on both TPU and datacenter GPU:

| Hardware | K=128/K=16 Verify Ratio |
|----------|------------------------|
| TPU v5p | 1.00x (flat through K=1024) |
| H100 SXM | 1.00-1.04x |
| RTX 2000 Ada | 1.24x |

## Repo Structure

```
tpu-spec-decode/
  README.md
  tpu-inference/        # DFlash TPU code (submodule, dflash-integration branch)
  pr-ready/             # PR branches submitted upstream (see pr-ready/README.md)
  docs/                 # 52 research docs (setup, integration, experiments, PR prep)
  benchmarks/           # Standalone benchmark scripts (tau, speedup, ablation, scaling)
  tests/                # Shell wrappers + JSON test configs
  results/              # V4 and V5P benchmark JSONs + CSVs
  verification/         # Evaluation harness and validation runs
  visualizations/       # Plot scripts and rendered figures
  preparation/          # TPU environment setup (bootstrap.sh, clone_repos.sh)
  requirements.txt
```

## Quick Start (fresh TPU VM)

```bash
git clone --recurse-submodules https://github.com/aaronzhfeng/tpu-spec-decode.git
cd tpu-spec-decode
bash preparation/bootstrap.sh
```

See `preparation/` for full setup docs.

## Running

Unit tests (no serving setup needed):

```bash
pytest tests/models/jax/test_qwen3_dflash_attention.py
pytest tests/models/jax/test_qwen3_dflash.py
pytest tests/spec_decode/test_dflash.py
```

End-to-end (requires PR #1868 + #1869):

```bash
python -m tpu_inference.entrypoint \
  --model Qwen/Qwen3-4B \
  --speculative_config '{"model": "z-lab/Qwen3-4B-DFlash-b16", "num_speculative_tokens": 15, "method": "dflash", "draft_tensor_parallel_size": 1}'
```

## Documentation

The `docs/` folder contains 52 documents covering the full project journey:
- **00-06:** Setup and architecture
- **07-19:** DFlash integration (plan, bugs, gap analysis, KV cache iterations)
- **20-45:** Experiments and results (acceptance rates, ablation, K-flat property, GPU comparison)
- **48-52:** PR preparation and review

## Related Repos

| Repo | Description |
|------|-------------|
| [tpu-dflash-paper](https://github.com/aaronzhfeng/tpu-dflash-paper) | Paper, report, slides |
| [dflash-wide](https://github.com/aaronzhfeng/dflash-wide) | GPU training for wide-block (K=16-128) experiments |
| [DFlash (upstream)](https://github.com/z-lab/dflash) | Original DFlash implementation |
| [tpu-inference (upstream)](https://github.com/vllm-project/tpu-inference) | vLLM TPU backend |

## References

- DFlash: "DFlash: Block Diffusion for Flash Speculative Decoding" (Chen et al., [arXiv:2602.06036](https://arxiv.org/abs/2602.06036))
- SSD/Saguaro: "Speculative Speculative Decoding" (Kumar, Dao, May, ICLR 2026, [arXiv:2603.03251](https://arxiv.org/abs/2603.03251))
