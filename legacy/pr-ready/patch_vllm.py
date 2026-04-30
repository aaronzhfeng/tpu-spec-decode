"""Patch vLLM speculative config to support dflash method."""
import vllm.config.speculative as sc
import inspect

src_file = inspect.getfile(sc)
with open(src_file) as f:
    code = f.read()

# 1. Add "dflash" to SpeculativeMethod Literal
code = code.replace(
    '"draft_model",',
    '"draft_model", "dflash",'
)

# 2. Add dflash check before the else clause that rejects unknown methods
code = code.replace(
    '                else:\n'
    '                    self.method = "draft_model"\n'
    '                    raise NotImplementedError(',
    '                elif self.method == "dflash":\n'
    '                    pass\n'
    '                else:\n'
    '                    self.method = "draft_model"\n'
    '                    raise NotImplementedError('
)

with open(src_file, "w") as f:
    f.write(code)
print(f"Patched {src_file}")
