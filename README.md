# DFlash Speculative Decoding on TPU

Block-diffusion speculative decoding ([DFlash](https://arxiv.org/abs/2602.06036), Z Lab @ UCSD) ported to Google TPUs via the vLLM `tpu-inference` JAX backend. **3.13× average speedup** across 9 benchmarks on TPU v5p with Qwen3-4B; head-to-head **2.29× DFlash vs 1.30× Eagle3** on v5p with Llama-3.1-8B. Companion to the Google Cloud blog post (April 2026) and the [colab notebook](https://colab.research.google.com/drive/1ekk8lY2u843KE9_dpJ36Z_vyv5idL-Pf).

**Team:** Zhaoxiang Feng, Zhongyan Luo, Son Nguyen, Andy Huang (UC San Diego)
**Advisors:** Hao Zhang, Yiming Zhao (UC San Diego)
**Collaborators:** Yarong Mu, Weiren Yu (Google Cloud)

---

## Reproduce in 5 minutes (fresh TPU VM)

This repo's role is to be a thin click-to-reproduce harness for the DFlash work. The actual DFlash code is upstreamed at [vllm-project/tpu-inference](https://github.com/vllm-project/tpu-inference) (PRs [#1868](https://github.com/vllm-project/tpu-inference/pull/1868), [#1869](https://github.com/vllm-project/tpu-inference/pull/1869), [#1870](https://github.com/vllm-project/tpu-inference/pull/1870)) and pulled in here as a `tpu-inference/` submodule on the `dflash-integration` branch.

```bash
# 1. Clone with the tpu-inference submodule populated
git clone --recurse-submodules https://github.com/aaronzhfeng/tpu-spec-decode.git
cd tpu-spec-decode

# 2. Set up Python deps + symlinks (idempotent)
bash preparation/bootstrap.sh

# 3. Unit tests (resolve via tests/models/, tests/spec_decode/ symlinks
#    into tpu-inference/tests/)
pytest tests/spec_decode/test_dflash.py
pytest tests/models/jax/test_qwen3_dflash.py
pytest tests/models/jax/test_qwen3_dflash_attention.py

# 4. Standalone benchmark — single dataset, ~5 min on v6e
DATASET=math500 bash tests/standalone_benchmark.sh \
    --max-samples 8 --max-new-tokens 256

# 5. Full vLLM pipeline benchmark — math suite, ~30 min on v6e
bash tests/benchmark.sh math
```

Tested on TPU v6e (primary). v5p numbers in the tables below come from the same harness; v4 paths are preserved but not validated.

---

## Headline results

### TPU v5p-8 (Qwen3-4B, K=16, greedy)

| Dataset | Category | τ | Speedup | Baseline TPOT (ms) | DFlash TPOT (ms) |
|---|---|---|---|---|---|
| math500 | math | 8.80 | 5.72× | 8.02 | 1.40 |
| aime24 | math | 6.48 | 3.98× | 7.69 | 1.93 |
| aime25 | math | 6.14 | 3.35× | 6.85 | 2.05 |
| gsm8k | math | 5.40 | 3.17× | 7.32 | 2.31 |
| humaneval | code | 5.76 | 3.53× | 7.70 | 2.18 |
| mbpp | code | 6.16 | 2.77× | 7.31 | 2.64 |
| mt-bench | chat | 3.87 | 2.36× | 7.20 | 3.05 |
| alpaca | chat | 2.86 | 1.65× | 6.72 | 4.08 |
| swe-bench | code | 3.35 | 1.60× | 6.85 | 4.27 |
| **Average** | | **5.42** | **3.13×** | **7.30** | **2.66** |

Models: [`Qwen/Qwen3-4B`](https://huggingface.co/Qwen/Qwen3-4B) target + [`z-lab/Qwen3-4B-DFlash-b16`](https://huggingface.co/z-lab/Qwen3-4B-DFlash-b16) draft.

### Head-to-head vs EAGLE-3 on TPU v5p (Llama-3.1-8B)

DFlash (K=10) **2.29×** end-to-end speedup; EAGLE-3 (K=2) **1.30×**. Math is up to 2.69×; coding (mbpp) hits 2.83× (9.81 ms → 3.48 ms TPOT). DFlash's parallel block drafting is largely insensitive to K, while autoregressive drafters pay an O(K) sequential cost.

### K-flat verification

Verification cost is invariant to draft block size on datacenter accelerators:

| Hardware | K=128 / K=16 verify ratio |
|---|---|
| TPU v5p | 1.00× (flat through K=1024) |
| H100 SXM | 1.00–1.01× |
| RTX 2000 Ada | 1.24× |

Wider blocks (K=32, 64, 128) cost essentially the same to verify. The bottleneck is **draft quality** (per-position acceptance probability α), not verification cost.

---

## Repo layout

```
tpu-spec-decode/
├── README.md                # this file (single user-facing entry point)
├── requirements.txt
├── .gitmodules              # tpu-inference + brainstorm submodules
├── preparation/
│   ├── bootstrap.sh         # COLAB-PINNED: fresh-VM setup
│   ├── setup_v5p_safe.sh    # newer venv-based v5p setup (optional)
│   ├── clone_repos.sh       # legacy multi-repo cloner (author's dev workspace)
│   └── setup_tpu_v5.sh      # v4-era setup script (preserved, not invoked)
├── tests/
│   ├── benchmark.sh                COLAB-PINNED: full vLLM pipeline
│   ├── standalone_benchmark.sh     COLAB-PINNED: standalone JAX runner
│   ├── models/    -> tpu-inference/tests/models/    (symlink)
│   └── spec_decode/ -> tpu-inference/tests/spec_decode/ (symlink)
├── benchmarks/              # Python scripts behind standalone_benchmark.sh
├── tpu-inference/           # GIT SUBMODULE — branch dflash-integration
├── verification/            # Test-matrix runner used by benchmark.sh
└── legacy/                  # Frozen development artifacts (not user-facing)
    ├── docs/                # 70 internal markdowns from the project's R&D phase
    ├── deliverables/        # Capstone PDF + poster
    ├── results/             # Earlier-run benchmark JSONs / CSVs
    ├── visualizations/      # Plot generation scripts + rendered figures
    ├── _workspace/          # Author's scratch metadata
    ├── brainstorm/          # Submodule with PR-reply drafts (private context)
    ├── pr-ready/            # Pre-submodule manual clones (gitignored on disk)
    └── (no setup needed inside legacy/ to reproduce the headline numbers)
```

The colab pin and the upstream PRs all reference the **non-`legacy/`** surface only. The `legacy/` directory preserves the development history without cluttering the user-facing flow.

---

## Where the actual code lives

The DFlash JAX implementation is upstreamed to `vllm-project/tpu-inference`:

| PR | Title | Files | Status |
|---|---|---|---|
| [#1868](https://github.com/vllm-project/tpu-inference/pull/1868) | Add DFlash model and proposer | 7 new (model, proposer, unit tests) | In review |
| [#1869](https://github.com/vllm-project/tpu-inference/pull/1869) | Integrate DFlash into pipeline | 5 modified (runner, manager, loader) | In review |
| [#1870](https://github.com/vllm-project/tpu-inference/pull/1870) | Add DFlash e2e tests and CI | 2 (e2e tests, Buildkite pipeline) | In review |

The `tpu-inference/` submodule in this repo is pinned to [`dflash-integration`](https://github.com/aaronzhfeng/tpu-inference/tree/dflash-integration) on our fork ([`aaronzhfeng/tpu-inference`](https://github.com/aaronzhfeng/tpu-inference)), which contains all three PR sets composed together for end-to-end reproduction.

A torchax proposer follow-up to PR #1868 is in preparation; once it lands, DFlash will be available on both the JAX and PyTorch serving paths of vLLM TPU.

---

## Related work

- **DFlash paper:** Chen, Liang, Liu (2026). [arXiv:2602.06036](https://arxiv.org/abs/2602.06036)
- **Reference GPU implementation:** [z-lab/dflash](https://github.com/z-lab/dflash)
- **EAGLE-3:** Li, Wei, Zhang, Zhang (2025). [arXiv:2503.01840](https://arxiv.org/abs/2503.01840)
- **vLLM TPU:** [vllm-project/tpu-inference](https://github.com/vllm-project/tpu-inference)
- **Colab notebook:** [drive/1ekk8lY2u843KE9_dpJ36Z_vyv5idL-Pf](https://colab.research.google.com/drive/1ekk8lY2u843KE9_dpJ36Z_vyv5idL-Pf)

---

## Acknowledgement of the `legacy/` directory

Earlier development of this project produced significant supporting material — 70 markdown notes, a capstone deliverable, plot scripts, scratch workspaces, and several manual sub-repo clones. To keep the user-facing flow above as small as possible without losing that history, we folded all of it into `legacy/` rather than deleting it. Anyone wanting to read the full development trail can browse there directly. Nothing under `legacy/` is required to reproduce the headline numbers; the colab and the bootstrap script use only the top-level surface.
