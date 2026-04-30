# PR-Ready Branches

Local clones of [aaronzhfeng/tpu-inference](https://github.com/aaronzhfeng/tpu-inference) used to prepare and submit the DFlash upstream PRs.

| Folder | Branch | Purpose |
|--------|--------|---------|
| `main/` | `main` | Synced with upstream `vllm-project/tpu-inference` main |
| `dflash-integration/` | `dflash-integration` | Working branch with full DFlash integration (seq_len fix) |
| `pr_dflash/` | `pr/dflash` | Original single-commit PR (before split) |
| `pr_dflash_1/` | `pr/dflash-1` | Final 3-PR split: model+proposer (#1868), pipeline (#1869), tests+CI (#1870) |
| `vllm-lkg/` | — | Last-known-good vLLM upstream for compatibility reference |

## Upstream PRs

- **PR #1868:** [Spec Decoding] Add DFlash model and proposer — model, proposer, unit tests
- **PR #1869:** [Spec Decoding] Integrate DFlash into speculative decoding pipeline
- **PR #1870:** [Spec Decoding] Add DFlash e2e tests and Buildkite CI

See `PR_PROGRESS.md` for the detailed branch setup timeline and issues encountered.
