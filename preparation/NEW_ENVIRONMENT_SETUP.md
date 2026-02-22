# New Environment Setup (TRC TPU)

This runbook is for setting up a fresh Cloud TPU environment and bootstrapping this repo's DFlash workflow.

## 1) GCP project and CLI setup

```bash
export PROJECT_ID=hao-ai-lab-trc
gcloud auth login
gcloud components update
gcloud components install alpha beta
gcloud config set project "${PROJECT_ID}"
gcloud config set compute/zone us-east1-d
```

Enable TPU API:

```bash
gcloud services enable tpu.googleapis.com
```

Create TPU service agent:

```bash
gcloud beta services identity create \
  --service tpu.googleapis.com \
  --project "${PROJECT_ID}"
```

Reference: <https://docs.cloud.google.com/tpu/docs/setup-gcp-account>

## 2) Service account for TPU VMs

Create a user-managed service account:

```bash
export SA_NAME=tpu-vm-sa
gcloud iam service-accounts create "${SA_NAME}" \
  --display-name="TPU VM Service Account"
export SA_EMAIL="${SA_NAME}@${PROJECT_ID}.iam.gserviceaccount.com"
```

Grant required roles:

```bash
for role in \
  roles/tpu.admin \
  roles/storage.admin \
  roles/logging.logWriter \
  roles/monitoring.metricWriter
do
  gcloud projects add-iam-policy-binding "${PROJECT_ID}" \
    --member="serviceAccount:${SA_EMAIL}" \
    --role="${role}"
done
```

## 3) Create TPU (queued resource recommended)

Check available accelerator types in zone:

```bash
gcloud compute tpus accelerator-types list --zone us-east1-d
```

Create queued resource (replace accelerator type with an available one):

```bash
export TPU_NAME=aaron-v5-node
export QR_NAME=aaron-v5-queued
export ACCELERATOR_TYPE=v5litepod-4

gcloud compute tpus queued-resources create "${QR_NAME}" \
  --node-id="${TPU_NAME}" \
  --project="${PROJECT_ID}" \
  --zone=us-east1-d \
  --accelerator-type="${ACCELERATOR_TYPE}" \
  --runtime-version=tpu-ubuntu2204-base \
  --service-account="${SA_EMAIL}"
```

Check status:

```bash
gcloud compute tpus queued-resources describe "${QR_NAME}" --zone us-east1-d
gcloud compute tpus tpu-vm list --zone us-east1-d
```

## 4) Connect and run bootstrap

SSH:

```bash
gcloud compute tpus tpu-vm ssh "${TPU_NAME}" --zone us-east1-d
```

On TPU VM — run the one-shot bootstrap (clones repo + sub-repos on correct branches + installs all Python deps):

```bash
export ROOT_TPU_INF_BRANCH=dflash-integration
export ZHONGYAN_DFLASH_BRANCH=zhongyan_dev
export ZHONGYAN_TPU_INF_BRANCH=zhongyan_dev
export ZHONGYAN_VLLM_BRANCH=zhongyan_dev
export HF_TOKEN=hf_xxxxxxxxxxxx   # optional — skip if not needed

bash <(curl -fsSL https://raw.githubusercontent.com/aaronzhfeng/tpu-spec-decode/main/preparation/bootstrap.sh)
```

Or clone manually and run:

```bash
git clone https://github.com/aaronzhfeng/tpu-spec-decode.git
cd tpu-spec-decode
ROOT_TPU_INF_BRANCH=dflash-integration \
ZHONGYAN_DFLASH_BRANCH=zhongyan_dev \
ZHONGYAN_TPU_INF_BRANCH=zhongyan_dev \
ZHONGYAN_VLLM_BRANCH=zhongyan_dev \
bash preparation/bootstrap.sh
```

**What bootstrap does (in order):**
1. Clones `tpu-spec-decode` (or uses existing clone)
2. Installs apt packages: `git docker.io jq python3-pip`
3. Clones all sub-repos on the specified branches via `clone_repos.sh`
4. Installs Python dependencies: JAX + tpu-inference stack + vLLM (editable)
5. Runs a JAX device smoke check

If a repo does not have a non-main branch yet, override intentionally:

```bash
ALLOW_MAIN_BRANCH=1 ZHONGYAN_DFLASH_BRANCH=main bash preparation/bootstrap.sh
```

**Note:** Log out and back in (or run `newgrp docker`) after bootstrap for Docker group membership to take effect.

## 5) Preflight and smoke validation

```bash
bash preparation/check_dflash_support.sh docker
bash preparation/run_dflash_acceptance_smoke.sh
```

Greedy smoke:

```bash
MANIFEST=verification/contribution/manifests/dflash_acceptance_smoke_greedy.json \
bash preparation/run_dflash_acceptance_smoke.sh
```

Stochastic smoke:

```bash
MANIFEST=verification/contribution/manifests/dflash_acceptance_smoke.json \
bash preparation/run_dflash_acceptance_smoke.sh
```

## 7) Cleanup when done

```bash
gcloud compute tpus queued-resources delete "${QR_NAME}" --zone us-east1-d
# or
gcloud compute tpus tpu-vm delete "${TPU_NAME}" --zone us-east1-d
```

## Notes

- If you see warnings about project tags (for example, missing `environment` tag), ask project admins to confirm org-policy requirements.
- Prefer deleting unused TPU and disk resources quickly to avoid quota lockups.
