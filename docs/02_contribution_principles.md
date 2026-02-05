# Contribution Principles: Reuse, Not Redo

## Core Philosophy

**We extend existing work, not replace it.**

When porting code to new hardware (GPU → TPU), contributing to open source, or adding features, we follow a strict principle:

> **Add compatibility, preserve functionality.**

---

## The Wrong Way ❌

```
# Creating a parallel universe
src/
└── tpu_dflash/           # Complete rewrite in JAX
    ├── model/
    ├── attention.py      # New implementation from scratch
    └── ...               # 2000+ lines of "new" code
```

Problems:
- Duplicates effort
- Diverges from upstream immediately
- Hard to maintain two codebases
- Can't benefit from upstream improvements
- Not PR-able

---

## The Right Way ✅

```
forked/
└── dflash/               # Fork of z-lab/dflash
    ├── model/
    │   ├── dflash.py     # MODIFIED: Added backend selection
    │   └── backends/     # NEW: TPU backend (minimal addition)
    │       └── tpu.py    # ~50 lines
    └── requirements-tpu.txt  # NEW: TPU deps
```

Benefits:
- Minimal diff from upstream
- Easy to review and merge
- Stays in sync with upstream
- Clear separation of concerns
- PR-ready

---

## Guiding Questions

Before writing new code, ask:

1. **Does this already exist?** → Use it
2. **Can I extend the existing code?** → Add to it
3. **Is a full rewrite necessary?** → Almost never

---

## Repository Structure

```
tpu-spec-decode/
├── external/             # Upstream originals (READ-ONLY reference)
│   ├── dflash/          # z-lab/dflash (original)
│   ├── tpu-inference/   # vllm-project/tpu-inference
│   └── ...
├── forked/              # Our forks (WHERE WE WORK)
│   └── dflash/          # aaronzhfeng/dflash (our fork)
├── docs/
└── preparation/
```

### Why Two Folders?

| Folder | Purpose | Git Remote |
|--------|---------|------------|
| `external/` | Reference upstream code | `origin` = upstream |
| `forked/` | Our changes | `origin` = our fork, `upstream` = original |

This lets us:
- Compare our changes against original (`diff external/dflash forked/dflash`)
- Keep upstream reference untouched
- Work freely in forked/ without fear of losing reference

---

## Workflow

### 1. Initial Setup
```bash
# Clone upstream to external/ (reference)
git clone https://github.com/z-lab/dflash.git external/dflash

# Clone our fork to forked/ (working copy)
git clone https://github.com/aaronzhfeng/dflash.git forked/dflash

# Add upstream remote to our fork
cd forked/dflash
git remote add upstream https://github.com/z-lab/dflash.git
```

### 2. Making Changes
```bash
cd forked/dflash
git checkout -b feature/tpu-support

# Make minimal changes...
# Test...

git commit -m "Add TPU backend support"
git push origin feature/tpu-support
```

### 3. Syncing with Upstream
```bash
cd forked/dflash
git fetch upstream
git merge upstream/main
# Resolve any conflicts
```

### 4. Opening PR
- Go to GitHub
- Open PR from `aaronzhfeng/dflash:feature/tpu-support` → `z-lab/dflash:main`

---

## Examples of Good Changes

### ✅ Adding a Backend
```python
# In dflash.py - add 3 lines
if self.config._attn_implementation == "tpu":
    from .backends.tpu import tpu_attention_forward
    attn_fn = tpu_attention_forward
```

### ✅ Adding Device Flexibility
```python
# Change from:
device_map="cuda:0"

# To:
device = "cuda:0" if torch.cuda.is_available() else "xla:0"
```

### ✅ New Optional File
```
model/backends/tpu.py  # ~50 lines, self-contained
```

---

## Real-World Reference: Lookahead Reasoning TPU Port

The [Lookahead-Reasoning TPU branch](https://github.com/ConstBob/Lookahead-Reasoning/compare/main...tpu) 
shows exactly how to add TPU support properly. Key patterns:

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

### ❌ Rewriting Core Logic
```python
# Don't rewrite the entire attention mechanism
# Just add a new backend option
```

### ❌ Changing Interfaces
```python
# Don't change function signatures that break existing code
# Add optional parameters with defaults instead
```

### ❌ Massive Refactoring
```python
# Don't reorganize the entire codebase
# Work within the existing structure
```

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
