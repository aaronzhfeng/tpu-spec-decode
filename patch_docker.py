"""Patch the Docker image's tpu_inference to add DFlash support.

This script applies minimal, targeted modifications to the stock
vllm/vllm-tpu:latest container's tpu_inference code to enable
DFlash speculative decoding.
"""

import os
import shutil
import sys

BASE = "/workspace/tpu_inference"
SRC = "/mnt/dflash_src"  # mounted from host

# Supported layouts for DFlash source (SRC):
# 1) Standard: tpu_inference/spec_decode/jax/dflash.py and
#              tpu_inference/layers/common/dflash_attention_interface.py
# 2) Model:    model/dflash.py and model/dflash_attention_interface.py
# 3) Flat:     dflash.py and dflash_attention_interface.py at top level of SRC
STANDARD_COPIES = [
    ("tpu_inference/spec_decode/jax/dflash.py",
     "tpu_inference/spec_decode/jax/dflash.py"),
    ("tpu_inference/layers/common/dflash_attention_interface.py",
     "tpu_inference/layers/common/dflash_attention_interface.py"),
]
MODEL_LAYOUT_COPIES = [
    ("model/dflash.py", "tpu_inference/spec_decode/jax/dflash.py"),
    ("model/dflash_attention_interface.py",
     "tpu_inference/layers/common/dflash_attention_interface.py"),
]
FLAT_COPIES = [
    ("dflash.py", "tpu_inference/spec_decode/jax/dflash.py"),
    ("dflash_attention_interface.py",
     "tpu_inference/layers/common/dflash_attention_interface.py"),
]


def _copy_list(copies):
    for src_rel, dst_rel in copies:
        src = os.path.join(SRC, src_rel)
        dst = os.path.join(BASE, dst_rel)
        os.makedirs(os.path.dirname(dst), exist_ok=True)
        shutil.copy2(src, dst)
        print(f"  Copied {src_rel} -> {dst_rel}")


def copy_new_files():
    """Copy DFlash-specific new files into the container."""
    # Copy the Docker-adapted draft model from the separate mount point
    docker_adapted = "/mnt/qwen3_dflash_docker.py"
    dst = os.path.join(BASE, "tpu_inference/models/jax/qwen3_dflash.py")
    os.makedirs(os.path.dirname(dst), exist_ok=True)
    shutil.copy2(docker_adapted, dst)
    print("  Copied qwen3_dflash_docker.py -> qwen3_dflash.py (Docker-adapted)")

    # Prefer flat layout (dflash_src/) so we don't require full tpu_inference tree
    if all(os.path.isfile(os.path.join(SRC, s)) for s, _ in FLAT_COPIES):
        print("  Using flat layout (dflash.py, dflash_attention_interface.py)")
        _copy_list(FLAT_COPIES)
        return
    if all(os.path.isfile(os.path.join(SRC, s)) for s, _ in STANDARD_COPIES):
        _copy_list(STANDARD_COPIES)
        return
    if all(os.path.isfile(os.path.join(SRC, s)) for s, _ in MODEL_LAYOUT_COPIES):
        print("  Using model/ layout")
        _copy_list(MODEL_LAYOUT_COPIES)
        return

    print("ERROR: DFlash source files not found under /mnt/dflash_src", file=sys.stderr)
    print("Required: dflash.py and dflash_attention_interface.py (flat layout),", file=sys.stderr)
    print("  or standard tpu_inference/ paths, or model/ layout.", file=sys.stderr)
    print("Contents of /mnt/dflash_src:", file=sys.stderr)
    try:
        for n in sorted(os.listdir(SRC))[:20]:
            kind = "dir" if os.path.isdir(os.path.join(SRC, n)) else "file"
            print(f"  [{kind}] {n}", file=sys.stderr)
    except OSError as e:
        print(f"  (listdir failed: {e})", file=sys.stderr)
    sys.exit(1)


def patch_file(filepath, patches):
    """Apply text patches to a file.

    Each patch is (old_text, new_text). old_text must appear exactly once.
    """
    full_path = os.path.join(BASE, filepath)
    with open(full_path, "r") as f:
        content = f.read()

    for old_text, new_text in patches:
        count = content.count(old_text)
        if count == 0:
            print(f"  WARNING: patch target not found in {filepath}:")
            print(f"    {old_text[:80]}...")
            continue
        if count > 1:
            print(f"  WARNING: patch target found {count} times in {filepath}, skipping:")
            print(f"    {old_text[:80]}...")
            continue
        content = content.replace(old_text, new_text)
        print(f"  Patched {filepath}")

    with open(full_path, "w") as f:
        f.write(content)


def patch_tpu_runner():
    """Add DFlash import and initialization to tpu_runner.py."""
    patch_file("tpu_inference/runner/tpu_runner.py", [
        # Add import
        (
            "from tpu_inference.spec_decode.jax.eagle3 import Eagle3Proposer",
            "from tpu_inference.spec_decode.jax.dflash import DFlashProposer\n"
            "from tpu_inference.spec_decode.jax.eagle3 import Eagle3Proposer",
        ),
        # Add elif for dflash initialization
        (
            '            elif self.speculative_config.method == "eagle3":\n'
            '                self.drafter = Eagle3Proposer(self.vllm_config, self)',
            '            elif self.speculative_config.method == "eagle3":\n'
            '                self.drafter = Eagle3Proposer(self.vllm_config, self)\n'
            '            elif self.speculative_config.method == "dflash":\n'
            '                self.drafter = DFlashProposer(self.vllm_config, self)',
        ),
    ])


def patch_speculative_decoding_manager():
    """Add DFlash dispatch to speculative_decoding_manager.py."""
    patch_file("tpu_inference/runner/speculative_decoding_manager.py", [
        # Add import
        (
            "from tpu_inference.spec_decode.jax.eagle3 import Eagle3Proposer",
            "from tpu_inference.spec_decode.jax.dflash import DFlashProposer\n"
            "from tpu_inference.spec_decode.jax.eagle3 import Eagle3Proposer",
        ),
        # Add dflash dispatch — route to eagle3 method since DFlash inherits from Eagle3
        (
            '        elif self.runner.speculative_config.method == "eagle3":\n'
            '            self._draft_token_ids = self.propose_eagle3_draft_token_ids(\n'
            '                sampled_token_ids,\n'
            '                aux_hidden_states,\n'
            '                attn_metadata,\n'
            '                spec_decode_metadata,\n'
            '                scheduler_output,\n'
            '                input_ids,\n'
            '            )\n'
            '        else:',
            '        elif self.runner.speculative_config.method == "eagle3":\n'
            '            self._draft_token_ids = self.propose_eagle3_draft_token_ids(\n'
            '                sampled_token_ids,\n'
            '                aux_hidden_states,\n'
            '                attn_metadata,\n'
            '                spec_decode_metadata,\n'
            '                scheduler_output,\n'
            '                input_ids,\n'
            '            )\n'
            '        elif self.runner.speculative_config.method == "dflash":\n'
            '            # DFlash reuses the eagle3 proposal path (DFlashProposer inherits Eagle3Proposer)\n'
            '            self._draft_token_ids = self.propose_eagle3_draft_token_ids(\n'
            '                sampled_token_ids,\n'
            '                aux_hidden_states,\n'
            '                attn_metadata,\n'
            '                spec_decode_metadata,\n'
            '                scheduler_output,\n'
            '                input_ids,\n'
            '            )\n'
            '        else:',
        ),
    ])


def patch_kv_cache_manager():
    """Extend KV cache manager to support DFlash multi-layer draft cache."""
    patch_file("tpu_inference/runner/kv_cache_manager.py", [
        # Change eagle3-only check to include dflash
        (
            'if self.runner.speculative_config and self.runner.speculative_config.method == "eagle3":',
            'if self.runner.speculative_config and self.runner.speculative_config.method in ("eagle3", "dflash"):',
        ),
        # Change single-layer loop to support multi-layer for dflash
        (
            '                # Eagle3 has only 1 layer\n'
            '                for i in range(1):',
            '                # Eagle3 has 1 layer; DFlash may have multiple\n'
            '                num_draft_layers = 1\n'
            '                if self.runner.speculative_config.method == "dflash":\n'
            '                    num_draft_layers = int(getattr(hf_config, "num_hidden_layers", 1))\n'
            '                for i in range(num_draft_layers):',
        ),
    ])


def patch_compilation_manager():
    """Extend compilation manager to support DFlash precompilation."""
    patch_file("tpu_inference/runner/compilation_manager.py", [
        # Add dflash to eagle3 precompilation check
        (
            '        if self.runner.speculative_config.method == "eagle3":\n'
            '            self._precompile_eagle3_helpers()',
            '        if self.runner.speculative_config.method in ("eagle3", "dflash"):\n'
            '            self._precompile_eagle3_helpers()',
        ),
    ])


def patch_qwen3():
    """Add auxiliary hidden state extraction to Qwen3 model."""
    # This is the most complex patch — we need to:
    # 1. Add helper functions for DFlash target layer ID computation
    # 2. Add aux_hidden_state_layers to Qwen3Model.__init__
    # 3. Override Qwen3Model.__call__ to capture hidden states
    # 4. Update Qwen3ForCausalLM.__call__ to return aux hidden states

    full_path = os.path.join(BASE, "tpu_inference/models/jax/qwen3.py")
    with open(full_path, "r") as f:
        content = f.read()

    # 1. Add helper functions before the Qwen3Attention class
    helper_code = '''
def _build_target_layer_ids(num_target_layers: int,
                            num_draft_layers: int) -> list[int]:
    """Evenly space draft layers across the target model."""
    if num_draft_layers <= 0:
        return []
    if num_draft_layers == 1:
        return [num_target_layers // 2]
    start = 2
    end = max(start, num_target_layers - 3)
    span = end - start
    return [
        int(round(start + (i * span) / (num_draft_layers - 1)))
        for i in range(num_draft_layers)
    ]


def _get_dflash_target_layer_ids(target_num_layers: int,
                                 draft_hf_config) -> list[int]:
    cfg = getattr(draft_hf_config, "dflash_config", None)
    if isinstance(cfg, dict):
        layer_ids = cfg.get("target_layer_ids", None)
        if layer_ids:
            return [int(x) for x in layer_ids]
    elif cfg is not None and hasattr(cfg, "target_layer_ids"):
        layer_ids = getattr(cfg, "target_layer_ids")
        if layer_ids:
            return [int(x) for x in layer_ids]

    num_target_layers = int(
        getattr(draft_hf_config, "num_target_layers", target_num_layers))
    num_draft_layers = int(getattr(draft_hf_config, "num_hidden_layers", 1))
    return _build_target_layer_ids(num_target_layers, num_draft_layers)

'''

    # Find the Qwen3Attention class definition and insert before it
    anchor = "class Qwen3Attention"
    if anchor in content:
        content = content.replace(anchor, helper_code + anchor)
        print("  Patched qwen3.py: added DFlash helper functions")
    else:
        print("  WARNING: Could not find Qwen3Attention class in qwen3.py")

    # 2. Add aux_hidden_state_layers to Qwen3Model.__init__
    # Find the end of Qwen3Model.__init__ — just before Qwen3ForCausalLM class
    init_anchor = "class Qwen3ForCausalLM"
    if init_anchor in content:
        # We need to add the aux_hidden_state_layers initialization in Qwen3Model.__init__
        # Find the lm_head initialization section
        lm_head_anchor = (
            "        if model_config.hf_config.tie_word_embeddings:\n"
            "            self.lm_head = self.embed.embedding\n"
            "        else:\n"
            "            self.lm_head = nnx.Param(\n"
            "                init_fn(rng.params(), (hidden_size, vocab_size), dtype),\n"
            '                sharding=(None, "model"),\n'
            "            )\n"
        )
        aux_init_code = (
            "\n"
            "        self.aux_hidden_state_layers: tuple[int, ...] = ()\n"
            '        if vllm_config.speculative_config and vllm_config.speculative_config.method == "dflash":\n'
            "            draft_hf_config = (\n"
            "                vllm_config.speculative_config.draft_model_config.hf_config)\n"
            "            self.aux_hidden_state_layers = tuple(\n"
            "                _get_dflash_target_layer_ids(hf_config.num_hidden_layers,\n"
            "                                             draft_hf_config))\n"
        )
        if lm_head_anchor in content:
            content = content.replace(
                lm_head_anchor,
                lm_head_anchor + aux_init_code
            )
            print("  Patched qwen3.py: added aux_hidden_state_layers init")
        else:
            print("  WARNING: Could not find lm_head anchor in qwen3.py")

    # 3. Override Qwen3ForCausalLM.__call__ to return aux hidden states
    old_call = (
        "        kv_caches, x = self.model(\n"
        "            kv_caches,\n"
        "            input_ids,\n"
        "            attention_metadata,\n"
        "        )\n"
        "        return kv_caches, x, []"
    )
    new_call = (
        "        kv_caches, x, aux = self.model(\n"
        "            kv_caches,\n"
        "            input_ids,\n"
        "            attention_metadata,\n"
        "        )\n"
        "        return kv_caches, x, aux"
    )
    if old_call in content:
        content = content.replace(old_call, new_call)
        print("  Patched qwen3.py: updated Qwen3ForCausalLM.__call__")
    else:
        print("  WARNING: Could not find Qwen3ForCausalLM.__call__ anchor")

    with open(full_path, "w") as f:
        f.write(content)

    # 4. Override Qwen3Model.__call__ in qwen2.py (base class)
    # Actually, since Qwen3Model inherits from Qwen2Model and doesn't override __call__,
    # we need to either: (a) override in Qwen3Model, or (b) modify Qwen2Model.__call__
    # Let's add an override to Qwen3Model by inserting it before Qwen3ForCausalLM
    with open(full_path, "r") as f:
        content = f.read()

    qwen3_call_override = '''
    def __call__(
        self,
        kv_caches,
        input_ids=None,
        attention_metadata=None,
        inputs_embeds=None,
    ):
        if inputs_embeds is not None:
            x = inputs_embeds
        else:
            x = self.embed(input_ids)
        aux_hidden_states = []
        for i, layer in enumerate(self.layers):
            kv_cache = kv_caches[i]
            kv_cache, x = layer(
                kv_cache,
                x,
                attention_metadata,
            )
            kv_caches[i] = kv_cache
            if i in self.aux_hidden_state_layers:
                aux_hidden_states.append(x)
        x = self.norm(x)
        return kv_caches, x, aux_hidden_states

'''

    # Insert the __call__ override just before Qwen3ForCausalLM
    anchor = "\nclass Qwen3ForCausalLM"
    if anchor in content:
        content = content.replace(anchor, qwen3_call_override + anchor)
        print("  Patched qwen3.py: added Qwen3Model.__call__ override")
    else:
        print("  WARNING: Could not find Qwen3ForCausalLM anchor for __call__ override")

    with open(full_path, "w") as f:
        f.write(content)


def patch_model_loader():
    """Register DFlash model in model_loader.py."""
    patch_file("tpu_inference/models/common/model_loader.py", [
        # Add import
        (
            "    from tpu_inference.models.jax.qwen3 import Qwen3ForCausalLM",
            "    from tpu_inference.models.jax.qwen3 import Qwen3ForCausalLM\n"
            "    from tpu_inference.models.jax.qwen3_dflash import Qwen3DFlashForCausalLM",
        ),
        # Add registry entries after Eagle3
        (
            '    _MODEL_REGISTRY["Eagle3LlamaForCausalLM"] = EagleLlama3ForCausalLM',
            '    _MODEL_REGISTRY["Eagle3LlamaForCausalLM"] = EagleLlama3ForCausalLM\n'
            '    _MODEL_REGISTRY["DFlashDraftModel"] = Qwen3DFlashForCausalLM\n'
            '    _MODEL_REGISTRY["Qwen3DFlashForCausalLM"] = Qwen3DFlashForCausalLM',
        ),
    ])


def patch_vllm_speculative_config():
    """Patch vLLM to accept 'dflash' as a speculative decoding method."""
    spec_config = "/workspace/vllm/vllm/config/speculative.py"
    with open(spec_config, "r") as f:
        content = f.read()

    # 1. Add "dflash" to EagleModelTypes (type system)
    old = 'EagleModelTypes = Literal["eagle", "eagle3", MTPModelTypes]'
    new = 'EagleModelTypes = Literal["eagle", "eagle3", "dflash", MTPModelTypes]'
    if old in content:
        content = content.replace(old, new)
        print("  Added 'dflash' to EagleModelTypes")
    else:
        print("  WARNING: Could not find EagleModelTypes definition")

    # 2. Add dflash to the method passthrough (skip auto-detection when method is set)
    old_pass = 'if self.method in ("eagle", "eagle3"):\n                    pass'
    new_pass = 'if self.method in ("eagle", "eagle3", "dflash"):\n                    pass'
    if old_pass in content:
        content = content.replace(old_pass, new_pass)
        print("  Added 'dflash' to method passthrough")
    else:
        print("  WARNING: Could not find method passthrough")

    # 3. Make eagle3 target validation also cover dflash
    # There may be multiple occurrences of method == "eagle3", only patch the validation one
    old_validate = (
        'eagle3_target_supported = ["llama", "qwen", "minicpm", "gpt_oss"]'
    )
    if old_validate in content:
        # Replace the check near the validation
        content = content.replace(
            'self.method == "eagle3"\n            and self.draft_model_config',
            'self.method in ("eagle3", "dflash")\n            and self.draft_model_config',
        )
        print("  Extended eagle3 target validation to include dflash")

    # 4. Make is_eagle property include dflash
    old_is_eagle = 'return self.method in ("eagle", "eagle3", "mtp")'
    new_is_eagle = 'return self.method in ("eagle", "eagle3", "dflash", "mtp")'
    if old_is_eagle in content:
        content = content.replace(old_is_eagle, new_is_eagle)
        print("  Extended is_eagle property to include dflash")

    # NOTE: We do NOT add "dflash" to the EAGLEConfig replacement block
    # (line ~388: if self.method in ("eagle", "eagle3"):)
    # DFlash uses its own Qwen3Config, not EAGLEConfig.

    with open(spec_config, "w") as f:
        f.write(content)


def main():
    print("=== DFlash Patch Script ===")
    print()

    print("0. Patching vLLM SpeculativeConfig...")
    patch_vllm_speculative_config()
    print()

    print("1. Copying new DFlash files...")
    copy_new_files()
    print()

    print("2. Patching tpu_runner.py...")
    patch_tpu_runner()
    print()

    print("3. Patching speculative_decoding_manager.py...")
    patch_speculative_decoding_manager()
    print()

    print("4. Patching kv_cache_manager.py...")
    patch_kv_cache_manager()
    print()

    print("5. Patching compilation_manager.py...")
    patch_compilation_manager()
    print()

    print("6. Patching qwen3.py (aux hidden states)...")
    patch_qwen3()
    print()

    print("7. Patching model_loader.py...")
    patch_model_loader()
    print()

    # Verify imports work
    print("8. Verifying imports...")
    try:
        # Need to invalidate import caches after patching
        import importlib
        if "tpu_inference" in sys.modules:
            # Remove cached modules
            to_remove = [k for k in sys.modules if k.startswith("tpu_inference")]
            for k in to_remove:
                del sys.modules[k]

        from tpu_inference.spec_decode.jax.dflash import DFlashProposer
        print("  DFlashProposer import: OK")
    except Exception as e:
        print(f"  DFlashProposer import: FAILED - {e}")

    try:
        from tpu_inference.layers.common.dflash_attention_interface import dflash_concat_attention
        print("  dflash_concat_attention import: OK")
    except Exception as e:
        print(f"  dflash_concat_attention import: FAILED - {e}")

    print()
    print("=== Patch complete ===")


if __name__ == "__main__":
    main()