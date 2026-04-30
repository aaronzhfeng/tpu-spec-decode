# DFlash TPU Porting Plan

## Philosophy

**We are NOT creating a new repo.** We are:
- Forking DFlash and adding TPU compatibility
- Preserving all existing GPU functionality
- Making changes suitable for a pull request upstream
- Adding a new backend, not replacing the existing one

---

## 1. Approach: Add TPU Backend

### Strategy

The original DFlash uses transformers' `ALL_ATTENTION_FUNCTIONS` which supports multiple backends (eager, flash_attention_2, sdpa, etc.). We will:

1. **Add a TPU/JAX attention backend** alongside existing options
2. **Keep PyTorch model structure** - no full JAX rewrite
3. **Use PyTorch/XLA** for TPU execution of the existing model
4. **Replace only GPU-specific ops** with TPU-compatible alternatives

### Why PyTorch/XLA (not pure JAX)?

- Preserves original code structure
- Minimal changes required
- PR-friendly (not a rewrite)
- PyTorch/XLA is mature on TPU v4/v5

---

## 2. Repository Structure (Working in Fork)

```
dflash/                          # Forked from z-lab/dflash
├── model/
│   ├── __init__.py
│   ├── dflash.py               # MODIFY: Add backend selection
│   ├── utils.py                # KEEP: Unchanged
│   └── backends/               # NEW: Backend implementations
│       ├── __init__.py
│       ├── cuda.py             # Wrapper for flash-attention-2
│       └── tpu.py              # TPU-compatible attention
├── benchmark.py                # MODIFY: Add TPU benchmarks
├── requirements.txt            # MODIFY: Add TPU dependencies
├── requirements-tpu.txt        # NEW: TPU-specific deps
└── README.md                   # MODIFY: Add TPU instructions
```

### Our Repo Structure

```
tpu-spec-decode/                 # This repo - orchestration & docs
├── docs/
│   ├── 00_research_overview.md
│   ├── 01_tpu_setup_guide.md
│   ├── 02_contribution_principles.md  # Reuse, not redo philosophy
│   └── 03_dflash_tpu_plan.md   # This file
├── external/                   # Upstream originals (READ-ONLY reference)
│   └── dflash/                 # z-lab/dflash
├── forked/                     # Our forks (WHERE WE WORK)
│   └── dflash/                 # aaronzhfeng/dflash ← Work HERE
└── preparation/
    └── clone_repos.sh
```

---

## 3. What Needs to Change in DFlash

### 3.1 Attention Backend (`model/dflash.py`)

**Current code** (line 86-99):
```python
attn_fn: Callable = eager_attention_forward
if self.config._attn_implementation != "eager":
    attn_fn = ALL_ATTENTION_FUNCTIONS[self.config._attn_implementation]
attn_output, attn_weights = attn_fn(
    self, q, k, v, attention_mask, ...
)
```

**Our change**: Add TPU backend option
```python
attn_fn: Callable = eager_attention_forward
if self.config._attn_implementation == "tpu":
    from .backends.tpu import tpu_attention_forward
    attn_fn = tpu_attention_forward
elif self.config._attn_implementation != "eager":
    attn_fn = ALL_ATTENTION_FUNCTIONS[self.config._attn_implementation]
```

### 3.2 New File: `model/backends/tpu.py`

```python
"""TPU-compatible attention for DFlash."""

import torch

def tpu_attention_forward(
    module,
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    attention_mask: torch.Tensor | None,
    dropout: float = 0.0,
    scaling: float | None = None,
    **kwargs,
) -> tuple[torch.Tensor, None]:
    """
    TPU-compatible attention using PyTorch's scaled_dot_product_attention.
    
    This replaces flash-attention-2 which is CUDA-only.
    Works with PyTorch/XLA on TPU.
    """
    # Use PyTorch's native SDPA (XLA-compatible)
    attn_output = torch.nn.functional.scaled_dot_product_attention(
        query, key, value,
        attn_mask=attention_mask,
        dropout_p=dropout if module.training else 0.0,
        is_causal=False,  # DFlash uses non-causal attention
        scale=scaling,
    )
    return attn_output, None
```

### 3.3 Device Handling

**Current code** uses `.to("cuda:0")`. We need:
```python
# Change from:
device_map="cuda:0"

# To support both:
import torch
device = "cuda:0" if torch.cuda.is_available() else "xla:0"
device_map=device
```

### 3.4 Dependencies

**New `requirements-tpu.txt`**:
```
torch
torch_xla[tpu] @ https://storage.googleapis.com/libtpu-releases/wheels/libtpu/...
transformers>=4.40.0
```

---

## 4. Implementation Steps

### Phase 1: Setup (Day 1) ✅ DONE

- [x] Fork z-lab/dflash to our GitHub → [aaronzhfeng/dflash](https://github.com/aaronzhfeng/dflash)
- [x] Set up repo structure (external/ vs forked/)
- [ ] Create `tpu-support` branch
- [ ] Set up TPU VM with PyTorch/XLA

### Phase 2: Core Changes (Day 2-3)

- [ ] Add `model/backends/` directory
- [ ] Implement `backends/tpu.py` with SDPA attention
- [ ] Modify `dflash.py` to support backend selection
- [ ] Add device-agnostic handling (cuda vs xla)
- [ ] Test basic model loading on TPU

### Phase 3: Testing & Validation (Day 4-5)

- [ ] Verify outputs match GPU version
- [ ] Run benchmark.py on TPU
- [ ] Compare speedup vs baseline autoregressive
- [ ] Document TPU-specific setup

### Phase 4: PR Preparation (Day 6-7)

- [ ] Clean up code, add docstrings
- [ ] Update README with TPU instructions
- [ ] Create PR to upstream z-lab/dflash
- [ ] Address review feedback

---

## 5. Key Technical Details

### 5.1 Why SDPA Works

PyTorch's `scaled_dot_product_attention`:
- Works on TPU via PyTorch/XLA
- Supports non-causal attention (required by DFlash)
- No CUDA dependencies
- Good performance on TPU v4

### 5.2 DFlash's Non-Causal Attention

Unlike standard LLM attention, DFlash uses **bidirectional** attention:
- Draft tokens can attend to ALL context tokens
- This is diffusion-style (denoise all positions simultaneously)
- Set `is_causal=False` in SDPA

### 5.3 Position Embeddings

RoPE (Rotary Position Embedding) is pure math - no GPU ops needed.
The existing implementation should work on TPU via XLA.

---

## 6. Testing Plan

### Unit Tests
```python
# Test attention output matches between backends
def test_attention_backend_equivalence():
    # Run same input through CUDA and TPU backends
    # Assert outputs are close (allowing for float precision)
```

### Integration Tests
```python
# Test full spec_generate produces same output
def test_spec_generate_tpu():
    # Load model on TPU
    # Generate with spec_generate
    # Compare acceptance rates with GPU baseline
```

### Benchmarks
```bash
# On GPU
python benchmark.py --backend cuda --model Qwen/Qwen3-8B

# On TPU  
python benchmark.py --backend tpu --model Qwen/Qwen3-8B
```

---

## 7. Immediate Next Steps

1. ~~Fork DFlash to our GitHub~~ ✅ Done: [aaronzhfeng/dflash](https://github.com/aaronzhfeng/dflash)
2. **Create branch** `feature/tpu-support`
3. **Start with minimal change**: Just the TPU attention backend
4. **Test on TPU VM** with Qwen3-8B-DFlash

```bash
# Work in our fork
cd forked/dflash
git checkout -b feature/tpu-support

# Add TPU backend
mkdir -p model/backends
# Create model/backends/tpu.py
# Modify model/dflash.py

# Test on TPU VM
python -c "
import torch
import torch_xla.core.xla_model as xm
device = xm.xla_device()
print(f'Using device: {device}')
"
```

### Comparing Changes
```bash
# See what we've changed vs upstream
diff -r external/dflash/model forked/dflash/model
```

---

*Created: February 2026*
*Approach: Fork + PR, not separate repo*
*See: `02_contribution_principles.md` for philosophy*