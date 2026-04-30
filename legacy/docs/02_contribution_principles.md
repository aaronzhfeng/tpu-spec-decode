# Contribution Principles: Reuse, Not Redo

**Last updated:** February 8 2026. For TPU access and quota (v5, zone `us-east1-d`), see [02_tpu_setup_guide.md](02_tpu_setup_guide.md).

---

## Core Philosophy

**We extend existing work, not replace it.**

Our goal is to add **DFlash as a speculative decoding method** to the **tpu-inference** stack. We do that by integrating into the tpu-inference repo (JAX/Flax implementation, new proposer, etc.), using the DFlash repo only as a **reference** for behavior and math.

> **Integrate into the target stack; use reference repos for behavior, not as the place to add TPU code.**

---

## The Wrong Way ❌

**1. Rewriting in isolation**
```
src/
└── tpu_dflash/           # Complete rewrite in JAX, separate from tpu-inference
    ├── model/
    ├── attention.py      # New implementation from scratch
    └── ...               # 2000+ lines of "new" code
```
- Duplicates effort, diverges from upstream, not PR-able into tpu-inference.

**2. Adding TPU support inside the DFlash repo**
```
forked/dflash/
└── model/
    ├── dflash.py         # MODIFIED: backend selection, TPU path
    └── backends/tpu.py   # TPU-specific code in z-lab/dflash
```
- Our integration **target** is **tpu-inference**, not DFlash. DFlash stays GPU/reference; TPU logic lives in tpu-inference.

---

## The Right Way ✅

**Integrate DFlash into tpu-inference:** add it as a new speculative method (e.g. `method == "dflash"`) in the runner and speculative decoding manager, implement the draft model and attention in JAX/Flax inside tpu-inference, and use [z-lab/dflash](https://github.com/z-lab/dflash) only as reference.

```
This repo (tpu-spec-decode):
  external/dflash/                       # REFERENCE ONLY — do not add TPU code here
  └── model/dflash.py, ...               # Use for behavior, baselines, math

Your tpu-inference fork (outside this repo):
  tpu_inference/
  ├── spec_decode/jax/
  │   └── dflash.py                      # DFlash proposer / JAX impl (new or extended)
  ├── runner/
  │   └── speculative_decoding_manager.py  # Wire in "dflash" method
  └── ...
```

Benefits:
- Single integration point (tpu-inference); aligns with EAGLE-3 and ngram patterns
- DFlash repo stays upstream-clean; we don't maintain a fork of DFlash for TPU
- Changes are PR-able to vllm-project/tpu-inference
- Clear separation: reference (DFlash) vs integration target (tpu-inference)

---

## Guiding Questions

Before writing new code, ask:

1. **Does this already exist?** → Use it
2. **Where does this code belong?** → DFlash logic for TPU goes in **tpu-inference**, not in the DFlash repo
3. **Can I extend the existing code?** → Add to it (e.g. add a new method in the manager, don’t fork DFlash for TPU)
4. **Is a full rewrite necessary?** → Almost never

---

## Repository Structure (within this repo only)

This document and the paths below are scoped to **this repo**: `development/tpu-spec-decode` (our private repo).

```
tpu-spec-decode/
├── docs/                 # This doc and other project docs
├── external/             # READ-ONLY reference clones (inside this repo)
│   └── dflash/          # z-lab/dflash — for DFlash behavior & baselines
├── preparation/         # Scripts, clone helpers, sanity checks
├── verification/        # Validation scripts and configs
└── ...
```

**Integration target (outside this repo):** Work on a fork of **vllm-project/tpu-inference** in your own location. This repo does not contain the tpu-inference fork; it only holds docs, reference clones, and tooling.

### Roles

| Location | Purpose | Modify for TPU? |
|----------|---------|-----------------|
| **This repo** `external/dflash` | Reference: DFlash algorithm, math, GPU impl, baselines | **No** — read-only |
| **Your tpu-inference fork** (elsewhere) | Integration target: add DFlash as a speculative method | **Yes** — add JAX/Flax DFlash there |

This lets us:
- Keep DFlash upstream untouched; use `external/dflash` only for comparison and behavior reference
- Keep all TPU-specific DFlash code in the tpu-inference fork (one place to maintain and PR)

---

## Workflow

### 1. Initial Setup (inside this repo)

```bash
# From this repo root (tpu-spec-decode/)
# Reference: DFlash (read-only, for behavior and baselines)
git clone https://github.com/z-lab/dflash.git external/dflash
```

For the integration target (tpu-inference), use your own fork in your preferred location **outside this repo**. Add upstream: `git remote add upstream https://github.com/vllm-project/tpu-inference.git`.

### 2. Making Changes

Work in **your tpu-inference fork** (outside this repo):

- Create a branch (e.g. `dflash-integration`), add DFlash as a speculative method (JAX impl in `spec_decode/jax/`, wire in runner).
- Use **this repo’s** `external/dflash` only to compare behavior and align with the reference.
- Test on TPU (see [02_tpu_setup_guide](02_tpu_setup_guide.md) — zone `us-east1-d` for TRC quota).
- Commit and push to your fork.

### 3. Syncing with Upstream

In your tpu-inference fork: `git fetch upstream`, `git merge upstream/main`, resolve conflicts as needed.

### 4. Opening PR
- Open a PR from your tpu-inference fork (e.g. `your-fork:tpu-inference:dflash-integration`) → `vllm-project/tpu-inference:main`.
- Do **not** open a PR to z-lab/dflash for TPU-specific code; that stays in the tpu-inference fork.

---

## Examples of Good Changes

### ✅ Adding a new speculative method in tpu-inference
```python
# In speculative_decoding_manager.py (or equivalent) — wire in DFlash alongside EAGLE-3
if method == "dflash":
    from tpu_inference.spec_decode.jax import dflash
    proposer = dflash.DFlashProposer(...)
elif method == "eagle":
    ...
```

### ✅ New JAX/Flax module under tpu-inference
```
tpu_inference/spec_decode/jax/dflash.py   # DFlash proposer + draft model in JAX
# Keep it focused; use ref_repos/dflash for algorithm reference, not for copying TPU code into DFlash.
```

### ✅ Preserving existing behavior
- Adding `method == "dflash"` as an option without changing EAGLE-3 or ngram paths
- Using the same runner/manager interface so DFlash plugs in like other proposers

---

## Real-World Reference: Lookahead Reasoning TPU Port

For **how to structure our DFlash integration**, the main reference is the existing **tpu-inference** code (EAGLE-3 proposer, speculative decoding manager). For **general patterns** when adding TPU support to a codebase (flags, device-agnostic naming, conditional branches), the [Lookahead-Reasoning TPU branch](https://github.com/ConstBob/Lookahead-Reasoning/compare/main...tpu) is a useful reference. Key patterns:

### Pattern 1: Add Flags, Don't Remove

```python
# test_models.py - BEFORE
parser.add_argument('--target_gpu_id', type=str, default='0,1')
parser.add_argument('--draft_gpu_id', type=str, default='2')

# test_models.py - AFTER (adds options, keeps compatibility)
parser.add_argument('--use_tpu', action='store_true', help='Enable TPU mode')
parser.add_argument('--target_device_id', type=str, default='0,1', 
                    help='For GPU: CUDA device IDs, for TPU: TPU chip IDs')
parser.add_argument('--draft_device_id', type=str, default='2',
                    help='For GPU: CUDA device IDs, for TPU: TPU chip IDs')
```

### Pattern 2: Rename for Generality

```python
# vllm_model.py - BEFORE
def __init__(self, model, eos=None, gpu_ids="", ...):
    self.gpu_ids = gpu_ids

# vllm_model.py - AFTER
def __init__(self, model, eos=None, device_ids="", ...):
    self.device_ids = device_ids
```

### Pattern 3: Conditional Branches (Both Paths Work)

```python
# Initialize models based on TPU or GPU mode
if args.use_tpu:
    # TPU mode: set TPU environment variables
    os.environ['TPU_CHIPS_PER_PROCESS_BOUNDS'] = f'1,{chip_count},1'
    os.environ['TPU_PROCESS_BOUNDS'] = '1,1,1'
    os.environ['TPU_VISIBLE_CHIPS'] = args.target_device_id
    target_model = Targeter(args.model, eos=None, 
                           target_device_id=args.target_device_id, ...)
else:
    # GPU mode: use CUDA_VISIBLE_DEVICES (original behavior)
    os.environ['CUDA_VISIBLE_DEVICES'] = args.target_device_id
    target_model = Targeter(args.model, eos=None,
                           target_device_id=args.target_device_id, ...)
```

### Pattern 4: Config-Driven Backend Selection

```python
# vllm_model.py
if vllm_config.get('use_tpu', False):
    # TPU mode: remove GPU-specific parameters
    engine_args = AsyncEngineArgs(
        model=model,
        tensor_parallel_size=tensor_parallel_size,
        # Note: NO gpu_memory_utilization (TPU doesn't use this)
        ...
    )
else:
    # GPU mode: keep original parameters
    engine_args = AsyncEngineArgs(
        model=model,
        tensor_parallel_size=tensor_parallel_size,
        gpu_memory_utilization=0.7,  # GPU-specific
        ...
    )
```

### Key Takeaways

| Pattern | Why It Works |
|---------|--------------|
| Add `--use_tpu` flag | Opt-in, doesn't break existing usage |
| Rename `gpu_ids` → `device_ids` | Generic naming works for both |
| `if use_tpu: ... else: ...` | Both paths preserved and tested |
| Remove GPU-only params in TPU branch | Clean separation without breaking GPU |

---

## Examples of Bad Changes

### ❌ Putting TPU DFlash code in the DFlash repo
- Our integration target is **tpu-inference**. Adding `backends/tpu.py` or TPU branches in z-lab/dflash is the wrong place; it won’t be used by the inference stack and diverges from upstream.

### ❌ Rewriting core logic from scratch
- Don’t rewrite the entire attention or draft model in isolation. Reuse tpu-inference patterns (e.g. EAGLE-3) and add a new method/proposer.

### ❌ Changing interfaces that break existing code
- Add optional parameters with defaults; don’t change function signatures used by EAGLE-3 or other methods.

### ❌ Massive refactoring
- Work within the existing tpu-inference structure; don’t reorganize the whole codebase.

---

## Measuring Success

A good contribution should:
- [ ] Have a small diff (< 200 lines ideally)
- [ ] Not break existing functionality
- [ ] Be testable in isolation
- [ ] Be understandable in a 5-minute code review
- [ ] Be merge-able without controversy

---

*"The best code is the code you don't have to write."*
