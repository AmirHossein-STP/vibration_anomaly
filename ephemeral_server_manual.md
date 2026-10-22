> **Superseded 2026-10-22 by [remote/README.md](remote/README.md)** (the `remote/box` tooling that implements this architecture). Kept as history. The storage volume is `/dev/vdb` (found by label `vault`), not `/dev/vdx`.

# Manual: Ephemeral GPU Compute with Persistent Vault Architecture

**Target Audience:** AI Agents, Data Scientists, and MLOps Engineers.
**Purpose:** This document explains how to replicate the "Ephemeral Server + Persistent Vault" architecture for new machine learning projects. You should feed this document to an AI agent when asking it to set up remote execution for a new repository.

---

## 1. The Core Architecture

When renting hourly, high-performance GPU instances (like RTX 3090 / 4090), the instance's fast SSD is ephemeral and is completely destroyed when the rental ends. 

To prevent data loss while maintaining maximum I/O performance, we use a two-drive architecture:
1.  **The Workspace (SSD):** The fast, ephemeral drive where the code runs, models train, and data is actively read/written.
2.  **The Vault (HDD):** A separate, persistent block storage volume mounted to the instance. It survives instance deletion and serves as the permanent home for datasets and results.

### The Data Lifecycle
1.  **Code** is synced from the local machine directly to the SSD workspace.
2.  **Raw Data** is synced from the local machine directly to the HDD Vault (saving massive upload bandwidth since it persists and allows rsync deltas).
3.  **On Boot**, the server copies datasets from the HDD Vault into the SSD Workspace.
4.  **Execution** happens entirely on the SSD for maximum speed.
5.  **Post-Execution**, a script immediately backs up results (artifacts, models, logs) from the SSD back to the HDD Vault before fetching them to the local machine.

---

## 2. The Server Environment

When setting up scripts for this architecture, assume the following about the remote server:
*   **Operating System / Environment:** Linux (Ubuntu/Debian), pre-installed with CUDA 12.8, PyTorch, TensorFlow, and Docker.
*   **Default Hostname:** `homa@185.13.231.184` (This should be configurable via environment variables or a shared `_common.sh` file, as the IP changes when a new instance is rented).
*   **Storage Devices:** 
    *   `/` (Root): The ephemeral SSD.
    *   `/dev/vdx`: The attached persistent block volume.

---

## 3. Implementation Guide for Agents

To implement this architecture in a new project, an AI agent should create the following suite of bash scripts (usually placed in `scripts/remote/`).

### A. The Safe Mounter (`mount_storage.sh`)
**Goal:** Safely partition, format, and mount the block volume to `/mnt/vdx1` without erasing existing data on subsequent rentals.

```bash
#!/usr/bin/env bash
set -euo pipefail
VAULT_MNT="/mnt/vdx1"
DISK="/dev/vdx"
PART="${DISK}1"

# Skip if disk doesn't exist
if ! lsblk "$DISK" >/dev/null 2>&1; then exit 0; fi

# Partition if unpartitioned
if ! lsblk "$PART" >/dev/null 2>&1; then
    echo -e "n\n\n\n\n\nw" | sudo fdisk "$DISK"
    sleep 2
fi

# Format ONLY if it lacks an ext4 filesystem
if ! sudo blkid "$PART" | grep -q ext4; then
    sudo mkfs.ext4 "$PART"
fi

# Mount and take ownership
if ! grep -q "$VAULT_MNT" /proc/mounts; then
    sudo mkdir -p "$VAULT_MNT"
    sudo mount "$PART" "$VAULT_MNT"
fi
sudo chown -R $USER:$USER "$VAULT_MNT"
```

### B. Shared Configuration (`_common.sh`)
**Goal:** Define variables used by all scripts so the IP/Paths only need to be updated in one place.
*   `REMOTE_HOST="${REMOTE_HOST:-homa@185.13.231.184}"`
*   `REMOTE_DIR="${REMOTE_DIR:-/home/homa/project-name}"`
*   `VAULT_DIR="${VAULT_DIR:-/mnt/vdx1/project-name}"`

### C. The Bootstrapper (`bootstrap.sh`)
**Goal:** Prepare a fresh ephemeral server for execution.
1.  **Sync Code:** Use `rsync` to push the local `.git` directory to `$REMOTE_DIR/.git` and run `git checkout -f` on the server.
2.  **Mount Vault:** Upload and execute `mount_storage.sh`.
3.  **Prepare Environment:** Install python versions (e.g., using `uv`), create virtual environments, and install dependencies.
4.  **Restore Data:** Run `rsync` to copy datasets, weights, and experiment tracking databases (like MLflow) from `$VAULT_DIR` to `$REMOTE_DIR`.

### D. The Data Syncer (`sync_data.sh`)
**Goal:** Push new datasets or large models from the developer's local machine to the cloud.
*   **Crucial Logic:** The `rsync` target MUST be `$REMOTE_HOST:$VAULT_DIR/data/`, **not** the SSD. 
*   This ensures that `rsync` compares against the persistent HDD, uploading only the deltas (diffs) instead of hundreds of megabytes every time a new server is rented.

### E. The Executor (`run_remote.sh`)
**Goal:** Run experiments and safely extract results.
1.  SSH into the server and execute the requested command inside the virtual environment on the SSD.
2.  **The Hook:** Wrap the execution in a way that captures the exit status, and then immediately runs a backup:
    ```bash
    rsync -a $REMOTE_DIR/artifacts/ $VAULT_DIR/artifacts/
    # Backup MLflow or TensorBoard logs here too
    ```
3.  Fetch the artifacts back to the local machine via `rsync`.
4.  Exit with the original status code of the experiment.

---

## 4. Notes on Machine Learning Tracking (MLflow, Weights & Biases)

*   **File-based tracking (MLflow/SQLite):** If using a local SQLite database for tracking, ensure it is backed up to the Vault in `run_remote.sh` and restored in `bootstrap.sh`. Be aware that absolute paths may break if the remote username changes across rentals. A path repair script is recommended.
*   **Cloud-based tracking (W&B/Comet):** The ephemeral server will have outbound internet access. Cloud trackers are highly recommended as they bypass the need to sync SQLite databases back and forth.
