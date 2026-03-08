# tpu-spec-decode

Research repo for porting and extending DFlash speculative decoding on TPU v4/v5.

**Team:** Aaron Feng, Zhongyan Luo, Son Nguyen, Andy Huang
**Mentor:** Hao Zhang, Yiming Zhao

---

## Quick Start (fresh TPU VM)

After SSH-ing into a new TPU node, run the one-shot bootstrap:

```bash
export ROOT_TPU_INF_BRANCH=dflash-integration
export HF_TOKEN=hf_xxxxxxxxxxxx   # optional

bash <(curl -fsSL https://raw.githubusercontent.com/aaronzhfeng/tpu-spec-decode/main/preparation/bootstrap.sh)
```

Or clone first, then bootstrap:

```bash
git clone --recurse-submodules https://github.com/aaronzhfeng/tpu-spec-decode.git
cd tpu-spec-decode
bash preparation/bootstrap.sh
```

See `preparation/NEW_ENVIRONMENT_SETUP.md` for the full GCP + TPU provisioning runbook.

---

## Repo Structure

```
tpu-spec-decode/
├── preparation/           # Setup scripts and runbooks
│   ├── bootstrap.sh           ← entry point for fresh node
│   ├── setup_tpu_v5.sh        ← apt + pip + smoke check
│   ├── clone_repos.sh         ← branch-pinned sub-repo cloning
│   ├── requirements_v5.txt    ← TPU v5 JAX deps
│   ├── check_dflash_support.sh
│   ├── run_dflash_acceptance_smoke.sh
│   └── NEW_ENVIRONMENT_SETUP.md
│
├── tpu-inference/         ← cloned by clone_repos.sh (dflash-integration branch)
├── vllm/                  ← git submodule (aaronzhfeng/vllm fork)
│
├── benchmarks/            # Standalone JAX/TPU benchmark scripts
│   ├── drafter_scaling.py
│   ├── amortized_verification.py
│   ├── layer_truncation.py
│   ├── tree_speculation.py
│   ├── gpu_matmul_scaling.py  ← PyTorch only, run on GPU not TPU
│   └── ...
│
├── tests/                 # Shell wrappers for benchmarks
├── docs/                  # Experiment docs (00–39+)
├── results/               # JSON outputs from benchmark runs
└── slides/                # Presentation decks
```

---

## Sub-repo Branches

| Repo | Default branch | Purpose |
|------|---------------|---------|
| `tpu-inference/` | `dflash-integration` | DFlash JAX port (used for vLLM pipeline) |
| `vllm/` (submodule) | tracked commit | Aaron's vLLM fork |

Override branch defaults by setting env vars before running bootstrap:

```bash
ROOT_TPU_INF_BRANCH=my-branch bash preparation/bootstrap.sh
```

---

## Key Results

| Metric | Value |
|--------|-------|
| TPU standalone τ (avg) | 6.67 (94% of GPU) |
| vLLM pipeline τ | 4.48 |
| vLLM pipeline speedup | 2.31× |
| DFlash vs Eagle3 on TPU | 2.06× higher τ |
| Verify latency K=16→K=256 | flat (~1.7ms, memory-bound) |

See `docs/` for full experimental records (Docs 00–39).

---

## Research Direction

Post-port findings (Docs 30–39) show verification cost is flat from K=16 to K=256 on TPU v4 due to memory-bandwidth boundedness. DFlash's block_size=16 is a GPU design constraint that doesn't apply on TPU. Proposed next step: train DFlash with block_size=128 (TPU-native optimal).
