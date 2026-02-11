# DFlash source files for patch_docker.py

This directory contains the two JAX modules required by `patch_docker.py`:

- **dflash.py** — `DFlashProposer` (subclass of Eagle3Proposer) for the speculative decoding runner.
- **dflash_attention_interface.py** — `dflash_concat_attention` for DFlash concat K/V attention.

Mount this directory as `/mnt/dflash_src` when running the patch so the script can copy these files into the container (flat layout).

Example (from repo root):

```bash
sudo docker run --rm --privileged --net host --shm-size=16G \
  -v $(pwd)/patch_docker.py:/mnt/patch_docker.py:ro \
  -v $(pwd)/qwen3_dflash_docker.py:/mnt/qwen3_dflash_docker.py:ro \
  -v $(pwd)/dflash_src:/mnt/dflash_src:ro \
  vllm/vllm-tpu:latest \
  python3 /mnt/patch_docker.py
```
