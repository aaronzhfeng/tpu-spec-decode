# TPU v5 Access Log

Tracks the full history of obtaining and troubleshooting TPU v5p access through TRC.

## Current Status

IN PROGRESS: Deleted `aaron-v5p-node6` (SSH-bricked, held all 8 chips of v5p quota). Waiting for delete operation to complete, then will re-create fresh as `aaron-v5p-node7`.

## Timeline

### Feb 4-6: Quota Request

Edgar Chen (Google) requested 4x v5p-8 (32 cores total) for project `hao-ai-lab-trc` through the TPU University Research Program, valid until May 4, 2026. TRC granted quota on Feb 6. Initial allocation was for zone `us-east1-d`.

### Feb 11: Permission Denied

Yiming attempted to create a queued resource in us-east1-d:

```bash
gcloud compute tpus queued-resources create capstone-queued \
  --node-id capstone-node \
  --project hao-ai-lab-trc \
  --zone us-east1-d \
  --accelerator-type v5p-8 \
  --runtime-version v2-alpha-tpuv5
```

Error: Permission Denied (Code 7) - "User does not have permission to submit requests into this queue for accelerator type v5p-8 in location us-east1-d."

Multiple lab members hit the same error. Emailed TRC support (CC'd Professor Hao Zhang).

### Feb 12: Zone Fix

TRC (Jonathan Caton) responded: quota was allocated to the wrong zone. Granted new quota in `us-east5` and `us-central1`.

### Feb 17: Stuck in WAITING_FOR_RESOURCES

Created v5p-8 in us-central1-a:

```bash
gcloud alpha compute tpus queued-resources create aaron-v5p-qr6 \
  --node-id=aaron-v5p-node6 \
  --project=hao-ai-lab-trc \
  --zone=us-central1-a \
  --accelerator-type=v5p-8 \
  --runtime-version=tpu-ubuntu2204-base \
  --guaranteed
```

Node stuck in WAITING_FOR_RESOURCES for days. Direct TPU VM creation showed quota exhausted ("Limit 8 in region us-central1") despite no running VMs. Emailed TRC.

### Feb 22: SSH Broken

TRC resolved the resource issue - node now shows READY/HEALTHY. However, SSH times out:

```
ssh: connect to host 136.119.198.146 port 22: Operation timed out
```

Likely cause: SSH daemon broken during aggressive package installs during environment setup. IAP tunnel also hangs. Emailed TRC asking for SSH restore or reset without losing TPU allocation.

### Feb 22 - Feb 28: No Response

TRC has not responded. Node remains READY/HEALTHY but inaccessible.

### Feb 28: Delete and Re-create

Confirmed via compute region quota that v5p is tracked separately under TPU API (`TPUV5PPerProjectPerRegionForTPUAPI`), not in compute region quotas. The compute quotas only show v5 lite entries (all at 0 usage).

Attempted to create `aaron-v5p-node7` without deleting - confirmed quota is fully consumed by the bricked node:

```
Quota 'TPUV5PPerProjectPerRegionForTPUAPI' exhausted. Limit 8 in region us-central1
Quota 'TPUV5PPerProjectPerZoneForTPUAPI' exhausted. Limit 8 in zone us-central1-a
```

Proceeded to delete `aaron-v5p-node6`:

```bash
gcloud compute tpus tpu-vm delete aaron-v5p-node6 \
  --zone=us-central1-a \
  --project=hao-ai-lab-trc
```

Delete operation in progress (operation ID: `operation-1772322324439-64beaf2936125-546d7332-ff326c15`). Once complete, will re-create as `aaron-v5p-node7`.

## Recovery Plan

Run these from local machine or Cloud Shell (not from inside a TPU VM).

### Step 0: Check current state

```bash
# What nodes exist?
gcloud compute tpus tpu-vm list --project hao-ai-lab-trc --zone us-central1-a

# What queued resources exist?
gcloud compute tpus queued-resources list --project hao-ai-lab-trc --zone us-central1-a

# Detailed node status
gcloud compute tpus tpu-vm describe aaron-v5p-node6 \
  --zone us-central1-a \
  --project hao-ai-lab-trc

# Check quota usage
gcloud compute tpus queued-resources describe aaron-v5p-qr6 \
  --zone us-central1-a \
  --project hao-ai-lab-trc
```

### Step 1: Try to recover SSH before deleting

```bash
# Attempt 1: SSH with explicit key and verbose logging
gcloud compute tpus tpu-vm ssh aaron-v5p-node6 \
  --zone us-central1-a \
  --project hao-ai-lab-trc \
  -- -v

# Attempt 2: IAP tunnel (bypasses external IP / firewall issues)
gcloud alpha compute tpus tpu-vm ssh aaron-v5p-node6 \
  --zone us-central1-a \
  --project hao-ai-lab-trc \
  --tunnel-through-iap

# Attempt 3: SCP a reset script (if SCP works but SSH doesn't)
gcloud compute tpus tpu-vm scp reset_ssh.sh aaron-v5p-node6:~ \
  --zone us-central1-a \
  --project hao-ai-lab-trc

# Attempt 4: Use startup script to fix SSH on next reboot
gcloud compute tpus tpu-vm update aaron-v5p-node6 \
  --zone us-central1-a \
  --project hao-ai-lab-trc \
  --metadata startup-script='#!/bin/bash
sudo systemctl restart sshd
sudo systemctl enable ssh'

# Attempt 5: Stop and restart the node (may re-init SSH)
gcloud compute tpus tpu-vm stop aaron-v5p-node6 \
  --zone us-central1-a \
  --project hao-ai-lab-trc

gcloud compute tpus tpu-vm start aaron-v5p-node6 \
  --zone us-central1-a \
  --project hao-ai-lab-trc

# Then retry SSH
gcloud compute tpus tpu-vm ssh aaron-v5p-node6 \
  --zone us-central1-a \
  --project hao-ai-lab-trc

# Attempt 6: Check if there's a firewall rule blocking port 22
gcloud compute firewall-rules list --project hao-ai-lab-trc

# Attempt 7: Try a different worker (multi-host TPUs have multiple VMs)
gcloud compute tpus tpu-vm ssh aaron-v5p-node6 \
  --zone us-central1-a \
  --project hao-ai-lab-trc \
  --worker=0
```

### Step 2: If recovery fails, delete and re-create

```bash
# Delete the queued resource (this also deletes the node)
gcloud compute tpus queued-resources delete aaron-v5p-qr6 \
  --project hao-ai-lab-trc \
  --zone us-central1-a

# If queued-resource delete hangs, try force flag
gcloud compute tpus queued-resources delete aaron-v5p-qr6 \
  --project hao-ai-lab-trc \
  --zone us-central1-a \
  --force

# If node persists after QR deletion, delete it directly
gcloud compute tpus tpu-vm delete aaron-v5p-node6 \
  --zone us-central1-a \
  --project hao-ai-lab-trc

# Verify cleanup
gcloud compute tpus tpu-vm list --project hao-ai-lab-trc --zone us-central1-a
gcloud compute tpus queued-resources list --project hao-ai-lab-trc --zone us-central1-a
```

### Step 3: Re-create fresh

```bash
gcloud alpha compute tpus queued-resources create aaron-v5p-qr7 \
  --node-id=aaron-v5p-node7 \
  --project=hao-ai-lab-trc \
  --zone=us-central1-a \
  --accelerator-type=v5p-8 \
  --runtime-version=tpu-ubuntu2204-base \
  --guaranteed

# Monitor until ACTIVE
watch -n 30 gcloud compute tpus queued-resources describe aaron-v5p-qr7 \
  --zone us-central1-a \
  --project hao-ai-lab-trc

# SSH once ACTIVE
gcloud compute tpus tpu-vm ssh aaron-v5p-node7 \
  --zone us-central1-a \
  --project hao-ai-lab-trc
```

### Step 4: Safe bootstrap (incremental, verify SSH between steps)

```bash
# First: verify SSH works and JAX sees TPU
python3 -c "import jax; print(jax.devices())"

# Then run bootstrap incrementally - do NOT run everything at once
# Clone repos first
cd ~ && git clone https://github.com/aaronzhfeng/tpu-spec-decode.git
cd tpu-spec-decode && bash preparation/clone_repos.sh

# Verify SSH still works from another terminal before continuing
# Then install dependencies
pip install -r preparation/requirements_v5.txt

# Verify SSH again, then run smoke test
python3 preparation/tpu_sanity_check.py
```

## Lessons Learned

- Do not run aggressive package installs immediately on a fresh TPU VM. Run bootstrap incrementally and verify SSH remains functional between steps.
- TRC response time is unpredictable (same-day for zone fix, 6+ days and counting for SSH issue).
- Quota can appear consumed by phantom resources - queued resources that fail may hold quota without running VMs.
- Keep node names incrementing (node6 -> node7) to avoid name collision during cleanup.

## Key Contacts

- TRC Support: trc-support@google.com
- Edgar Chen (Google TPU University Program): edgarchen@xwf.google.com
- Aditi Joshi (Google): aditijoshi@google.com

## Project Details

- Project ID: hao-ai-lab-trc
- Accelerator: v5p-8 (4 chips)
- Zone: us-central1-a (originally us-east1-d, moved by TRC)
- Quota valid until: May 4, 2026
