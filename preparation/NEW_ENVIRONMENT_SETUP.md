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

## 4) Connect and install basics

SSH:

```bash
gcloud compute tpus tpu-vm ssh "${TPU_NAME}" --zone us-east1-d
```

On TPU VM:

```bash
sudo apt-get update
sudo apt-get install -y git docker.io jq
sudo usermod -aG docker "$USER"
newgrp docker
```

## 5) Clone and branch-pin repos

```bash
cd ~
git clone https://github.com/aaronzhfeng/tpu-spec-decode.git
cd tpu-spec-decode
```

Bootstrap branch-pinned repos (no worktrees):

```bash
ROOT_TPU_INF_BRANCH=dflash-integration \
ZHONGYAN_DFLASH_BRANCH=zhongyan_dev \
ZHONGYAN_TPU_INF_BRANCH=zhongyan_dev \
ZHONGYAN_VLLM_BRANCH=zhongyan_dev \
bash preparation/clone_repos.sh
```

If a repo does not have a non-main branch yet, override intentionally:

```bash
ALLOW_MAIN_BRANCH=1 ZHONGYAN_DFLASH_BRANCH=main bash preparation/clone_repos.sh
```

## 6) Preflight and smoke validation

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
