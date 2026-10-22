# Remote compute: the rented GPU box and the persistent vault

**The only runbook for rented compute.** It replaces `ephemeral_server_manual.md`
and `notes/11_gpu_server_runbook.md`, which stay as history. Built and first used
on 2026-10-22.

## The setup in one table

| Place | Lifetime | Holds |
|---|---|---|
| **Mac** (this repo) | permanent | code, `remote/.box.env` (current IP, gitignored), pulled evidence |
| **Vault**: the provider's "storage server", a 32 GiB block volume attached to the box, ext4 label `vault`, mounted at `/mnt/vault` | permanent | datasets, env caches, `outputs/` mirror, run records |
| **Box SSD** (`~/work`), 100 GB | hours: **deleted with the rental** | code checkout, venv, staged data, live runs |

Rule: anything expensive to rebuild lives on the vault, and anything cheap to rebuild lives on the SSD.
**Every dataset is downloaded by the box itself, straight onto the vault**, never
through the laptop (the laptop's uplink is ~500 KB/s).

## What every rented box has (provider image)

Every compute rental comes with the same image (owner, 2026-10-22). Measured on
payesh-20, 2026-10-22:

| | |
|---|---|
| OS | **Ubuntu 22.04** (22.04.5), user `homa`, passwordless `sudo` |
| GPU | NVIDIA RTX 3090 24 GB, driver 570.169, **CUDA 12.8** (runtime; `nvcc` is not on PATH) |
| CPU / RAM / disk | 8 cores, 29 GiB, 100 GB SSD (~76 GB free) |
| Python | 3.10.12 system python |
| **PyTorch** | **2.8.0+cu128**, preinstalled on system python, CUDA works |
| **TensorFlow** | **2.19.0**, preinstalled, sees the GPU |
| **Docker** | 28.3.3 |
| SSH | the provider installs the Mac's `~/.ssh/id_ed25519.pub` at rent time, so key login works with no password |

Consequences, all handled by `bootstrap.sh`:
- The venv is `uv venv --system-site-packages`, so the image's torch/TF are used as they are. **Never
  pip-install torch** (notes/11: 10 min lost, and `download.pytorch.org` stalls here).
- `remote/requirements-server.txt` pins numpy to the image's 2.1.2 and has no torch.
- Missing tools (`unrar`, `7z`) are installed from `.deb`s cached on the vault.
- Docker images are re-pulled on the SSD when needed (fast downlink). They are not stored on the vault.

## Every rental

```bash
remote/box up <ip>            # ~30 s: code -> vault -> venv -> outputs -> stage -> syncd
remote/box status             # GPU, vault, datasets, sync daemon, runs
remote/box run NAME -- python3 scripts/analysis/whatever.py --args
remote/box logs last -f       # or: logs <part of a run id>
remote/box pull               # vault results -> local outputs/ (any time)
remote/box down               # BEFORE the rental ends: final sync, pull, clean unmount
```

`box run` pushes the current working tree first (uncommitted changes included,
~330 KB, seconds). The command then runs detached on the box, in the repo, in the venv,
with `PYTHONPATH=src`. It survives ssh drops. Each run leaves
`runs/<id>/{cmd.txt, git.json, diff.patch, env.txt, log, exit_code, duration_s}`,
which records exactly which code ran.

Results: the sync daemon copies `~/work/<project>/outputs/` and `~/work/runs/` to the
vault **every 60 s**, and once more when a run ends. If the box disappears, at most a
minute is lost. The next `box up` restores `outputs/` (feature caches included) from the vault.

Other commands: `box push`, `box stage NAME[:PART]`, `box ssh [CMD]`, `box tunnel 8888`
(Jupyter/TensorBoard), `box pull --all` (also `.pt/.npz/.npy` binaries).

## First rental with a NEW (blank) storage volume, once only

```bash
remote/box up <ip> --init-vault
```
`mount_vault.sh --init` partitions and formats **only** if exactly one disk qualifies:
not the root disk, no partitions, no filesystem or partition-table signature, and
20–100 GiB. Tested on loop devices: a disk with a filesystem is refused, a
5 GiB disk is refused, and on a formatted disk `--init` just mounts. From then on the
vault is found by **label**, never by device name (it was `vdb` on payesh-20, and
`vdx` in the provider's help).

**One box at a time** may mount the vault (ext4 is not a shared filesystem). The
provider attaches the volume to one server anyway. `.mount-history` on the vault
logs every mount.

## Datasets

Registry: `remote/datasets/registry.yaml`. Fetcher: `remote/datasets/fetch.py`
(runs on the box). Committed manifests: `remote/datasets/manifests/*.json`.

```bash
remote/box data list                  # registry vs vault
remote/box data fetch NAME...|all     # detached run; resumable (.part files)
remote/box data verify [NAME|all]     # re-hash vs vault manifest AND vs the committed manifests
remote/box data manifests             # copy vault manifests into the repo -> commit them
remote/box data push LOCAL DEST       # laptop data -> $VAULT/DEST (private recordings)
remote/box stage cwru:de12 paderborn:v1   # vault -> SSD, linked where the code looks
```

| Dataset | On the vault | Source that works from the box | Staged as |
|---|---|---|---|
| cwru | 161 npz (12k DE, 48k DE, FE, Normal), 0.63 GB | GitHub `srigas/CWRU_Bearing_NumPy` | `cwru:de12` → `data/raw` (the 64 files in `remote/datasets/cwru_de12.txt`); `cwru` → `data/cwru_full` |
| paderborn | condensed **v1** (12 kHz vibration, the old store) + **v2** (64 kHz vibration + speed, torque, force) | Zenodo record 15845309 (31 of 32 codes; **KA08 is missing**, so v1 KA08 is merged from the laptop's store) | `paderborn:v1` → `data/pu/prepared_pu`; `paderborn:v2` → `data/pu/prepared_pu_v2` |
| xjtu_sy | condensed: one `(minutes, 32768, 2)` float32 `.npy` per bearing | MediaFire (author's folder) | `xjtu_sy` → `data/xjtu_sy` |
| ottawa_2018 | 60 raw `.mat` (vibration + encoder, 200 kHz, time-varying speed) | Mendeley `v43hmbwxpm` v2 | `data/ottawa_2018` |
| ottawa_2023 | 60 raw `.mat` (constant speed, 42 kHz) | Mendeley `y2px5tg92h` v5 | `data/ottawa_2023` |
| mfpt | 20 test-rig `.mat` | GitHub `mathworks/RollingElementBearingFaultDiagnosis-Data` | `data/mfpt` |
| ims | `IMS.zip`, **SEALED** (read-only, never extracted) | data.nasa.gov (mirror: NASA S3) | cannot be staged; `fetch.py unseal ims --commit <sha> --confirm WP11` at WP11 only |

Every file is checked against the publisher's own checksum (sha256 on Mendeley and MediaFire, md5 on Zenodo, git blob sha1
on GitHub), and its sha256 is recorded. Paderborn and XJTU-SY raw archives are downloaded to the SSD,
condensed onto the vault, then deleted. Their raw checksums stay in the manifest, so a re-download can be
verified.

**Not reachable from the box (2026-10-22). Do not plan around them:**
- **Kaggle / Google Cloud Storage / Google Drive**: "service not available in your location".
  So the old Kaggle → laptop → server route is dead from the server side, and Kaggle cannot be a runtime either.
- KAt DataCenter (uni-paderborn.de): 503, then no route.
- engineering.case.edu (CWRU): timeout.
- data.nasa.gov: up, then timing out on the same day. Retries succeeded.

Measured downlink: PyPI 15 MB/s, MediaFire 28 MB/s (6 parallel), Mendeley 14 MB/s (8 parallel),
Zenodo 11 MB/s (6 parallel), GitHub 6–10 MB/s, data.nasa.gov 7 MB/s.

Untrusted data: each archive is listed before extraction and refused if any member is absolute
or contains `..`. It is extracted into a new empty directory, which is then checked for symlinks.
Scripts never run with a data directory on `sys.path`.

### Space budget (32 GiB)

About 9 GB of datasets, ~0.4 GB of env cache, and ext4 overhead. The rest (~20 GB) is for `outputs/` and run
records. `fetch` refuses to start a dataset whose expected size would leave less than 1 GB free.
`box status` shows usage.

## Vault layout

`VAULT.md` on the vault itself describes it (rewritten by every `box up`):

```
/mnt/vault/  VAULT.md  .vault-id  .mount-history
  env/{bin/uv, uv-cache/, apt-debs/}
  datasets/<name>/<version>/{MANIFEST.json, SOURCE.md, raw/ | condensed/, SEALED?}
  projects/<project>/{outputs/, runs/<id>/}
```
A second project sets its own `PROJECT=` in its copy of `remote/box.conf` and gets its
own `projects/<name>/`. The datasets are shared.

## Traps (kept from notes/11, still true)

- `rsync --exclude 'data/'` is not anchored. `box push` builds its list from `git ls-files` instead.
- `scp` fails on this provider (no sftp subsystem). `box` uses ssh pipes and rsync over ssh.
- `cmd | tail; $?` reads tail's status. Scripts here use `set -euo pipefail`.
- `pgrep -f` matches its own ssh command line. Run state comes from `runs/<id>/pid` and `exit_code`.
- New ssh connections can stall while a transfer runs. `box` multiplexes everything over one
  ControlMaster connection and retries only on connection failure (exit 255).
- The IP is recycled between rentals. `box up` resets `remote/.known_hosts`.

## Recovery

- **Box died mid-run.** `box up <new ip>`, `box status` (the run shows as DIED), rerun it. Outputs up to
  the last sync are already restored.
- **Box died mid-fetch.** Rerun the same `box data fetch`. Verified files are skipped, and `.part` files resume.
- **Vault lost.** Make a new blank volume, run `box up <ip> --init-vault`, then `box data fetch all`. Afterwards
  `box data verify all` checks every file against the manifests committed in git. Run evidence
  survives only as far as the last `box pull`, which `box down` always does.

## Files

```
remote/box                     laptop CLI
remote/box.conf                project settings (tracked)
remote/server/*.sh             bootstrap, mount_vault, syncd, run, stage, status (+ VAULT.md)
remote/datasets/               registry.yaml, fetch.py, safe_extract.py, condense_pu.py,
                               condense_xjtu.py, cwru_de12.txt, manifests/
remote/requirements-server.*   venv lock for the box (py3.10, numpy pinned, no torch)
tests/test_remote_fetch.py     fetcher/extractor safety tests (no network)
```
