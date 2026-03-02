# TPU v5 Environment Baseline

Captured on 2026-03-01 from a fresh TPU v5 VM.

## GCP Identity

| Field | Value |
|-------|-------|
| Project | `hao-ai-lab-trc` |
| Zone | `us-central1-a` |
| Instance | `t1v-n-4a77ebd0-w-0` |
| Accelerator | **v5p-8** |
| gcloud SDK | 438.0.0 (`/snap/bin/gcloud`) |

## OS & Python

- **OS:** Ubuntu 22.04.2 LTS (Jammy Jellyfish)
- **Kernel:** Linux 5.19.0-1027-gcp
- **Python:** 3.10.6 (`/usr/bin/python3`)
- **pip:** 22.0.2 (system package, `/usr/lib/python3/dist-packages/pip`)
- **venv:** available (`python3 -m venv`)
- **virtualenv:** available (`/usr/local/bin/virtualenv`)
- **conda:** not installed

## ML Stack

Nothing is pre-installed. No JAX, PyTorch, Flax, NumPy, vLLM, or libtpu.

## System Resources

| Resource | Value |
|----------|-------|
| CPUs | 208 vCPUs |
| RAM | ~440 GiB (462,240,488 kB) |
| Swap | None |
| Root disk | 97 GB total, 87 GB free (11% used) |
| Shared memory (`/dev/shm`) | 221 GiB |

## TPU Devices

**4 TPU v5p chips** detected:

- `/dev/vfio/0` through `/dev/vfio/3` — permissions `666` (world read/write), owned by root
- No `/dev/accel*` devices present (differs from v4)
- `lspci` shows 4x Google Device 0062 (unassigned class `ff00`)

The v5p uses VFIO-based device passthrough instead of the accel subsystem used on v4. JAX/libtpu installation steps will differ from v4.

## Networking

| Interface | Address | Notes |
|-----------|---------|-------|
| `ens5` | `10.128.0.4/32` | Primary NIC (gVNIC) |
| `docker0` | `169.254.123.1/24` | Docker bridge (down, no containers) |
| `lo` | `127.0.0.1/8` | Loopback |

### Outbound Connectivity

- **PyPI** (`pypi.org`): reachable (HTTP 200)
- **GCS** (`storage.googleapis.com`): reachable (HTTP 400 = endpoint is up, no request body)

## Pre-installed System Packages

| Package | Version |
|---------|---------|
| docker-ce | 24.0.4 |
| docker-compose-plugin | 2.19.1 |
| docker-buildx-plugin | 0.11.1 |
| git | 2.34.1 |
| curl | 7.81.0 |
| python3-pip | 22.0.2 |

## Permissions & Limits

- **sudo:** passwordless (via `google-sudoers` group)
- **Docker:** requires `sudo` — user is **not** in the `docker` group
- **User groups:** `adm dialout cdrom floppy audio dip video plugdev netdev lxd ubuntu google-sudoers`
- **Open file limit (`ulimit -n`):** 100,000
- **Max user processes (`ulimit -u`):** 1,805,594
- **`vm.max_map_count`:** 65,530

## Services

- **SSH:** active and running

## Migration Notes (v4 → v5)

1. **Accelerator is v5p-8** — this is the high-end v5 variant (not v5e). Confirm your JAX/libtpu builds target v5p specifically.
2. **ML stack must be installed from scratch** — the image ships with zero ML libraries, including no libtpu.
3. **TPU device path changed** — v5p uses `/dev/vfio/` not `/dev/accel*`. JAX/libtpu installation and device detection will differ.
4. **pip is outdated** — system pip is 22.0.2; upgrade early (`pip3 install --upgrade pip`).
5. **Docker requires sudo** — add user to docker group (`sudo usermod -aG docker $USER`) or prefix with `sudo`.
6. **No conda** — use `python3 -m venv` or `virtualenv` for environment isolation.
7. **Generous shared memory** — 221 GiB `/dev/shm` is available for multiprocessing / data loading.
8. **Ample host resources** — 208 vCPUs and 440 GiB RAM for preprocessing, compilation, etc.
