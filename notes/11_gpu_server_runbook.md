# GPU server runbook (185.13.231.x, rented per minute)

Written after two rentals on 2026-09-17, the first of which was deleted
with all its results still on it, and the second of which lost ~25 of its
60 minutes to setup mistakes that are all avoidable. Read this BEFORE
renting again. The goal is to be running real work within 8 minutes.

## The one rule

**The box is ephemeral. When the subscription ends, the machine is DELETED
-- disk and all.** Not stopped, not suspended. Everything not pulled back
to the laptop is gone.

So: pull results continuously, never "at the end". A background loop that
rsyncs `outputs/` every 25s costs nothing and is the difference between
having your evidence and having only your conclusions. The first rental
ended with eight seeds' worth of conclusions committed to the docs and
zero artifacts to back them up -- which later made an unreproducible
result impossible to diagnose.

Billing is **per minute**, so idle time is money. Have the commands ready
before the box exists.

## What the box already has

Verified on the 2026-09-17 rental:

| | |
|---|---|
| GPU | NVIDIA GeForce RTX 3090, 24 GB, capability 8.6 |
| CPU | 8 cores |
| Disk | ~76 GB free on `/home` |
| OS | Ubuntu 22.04, Python 3.10.12 |
| **torch** | **2.8.0+cu128 ALREADY INSTALLED on system python3, CUDA working** |

**Do not build a venv.** The second rental burned ~10 minutes downloading
4.6 GB of CUDA wheels into a fresh venv before discovering system python3
already had a working CUDA torch. A venv without `--system-site-packages`
cannot see it, so the build is pure waste.

What is missing and must be installed (fast, ~1 min -- the server's own
downlink is quick):

```bash
python3 -m pip install --user scipy scikit-learn matplotlib pandas pyyaml tqdm
```

`numpy` is already present. Run everything with `PYTHONPATH=src python3`.

If you ever DO need a specific torch, install from plain PyPI
(`pip install --user torch==2.6.0`), not from `download.pytorch.org`: that
index resolves CUDA dependencies via `pypi.nvidia.com`, which stalled for
10+ minutes with `ReadTimeoutError` on this provider.

## The network, which is the real constraint

Wildly asymmetric, and this shapes every plan:

- **Server downlink: fast** (~8.5 MB/s; it pulled 4.6 GB in ~9 min). Installing
  packages on the server is cheap.
- **Laptop -> server uplink: slow** (~400-700 KB/s, ~370 ms RTT). Moving the
  437 MB Paderborn bundle takes ~15 min. Moving 179 MB of CWRU takes ~7.
- **While a transfer runs, sshd stops accepting connections.** Not a drop --
  `ping` stays at 0% loss while `ssh` times out. Every control command
  during a transfer must be wrapped in a retry loop.

Consequences:
1. Prefer having the server download things itself over uploading them.
2. Trim datasets before sending. For arm A, Paderborn's target condition is
   N09 only, which is 25% of each `.npz` -- repacking 17 codes to N09-only
   took 437 MB down to **63 MB** and cost ~1 minute of local CPU.
3. Start the big transfer FIRST, in the background, then do setup while it runs.

## Traps that cost real time

Each of these was hit on 2026-09-17.

**`rsync --exclude 'data/'` is not anchored.** It excluded `src/vibration_ad/data/`
and `src/domain_adaptation/data/` -- the entire loader package -- so the repo
arrived subtly broken. Use `--exclude '/data/'` (leading slash = repo root
only), and verify afterwards:

```bash
ssh $H 'ls ~/proj/src/domain_adaptation/data/*.py | wc -l'
```

**`scp` fails on this provider** with `subsystem request failed on channel 0`
(no sftp server). Use a pipe instead -- more reliable than scp anyway:

```bash
ssh $H 'cat > /tmp/run.sh && chmod +x /tmp/run.sh' < run.sh     # single file
tar czf - -C src/pkg data | ssh $H 'cd ~/proj/src/pkg && tar xzf -'   # a tree
```

**`cmd | tail` then `$?` reads tail's exit code, not the command's.** Two
rsyncs were reported as succeeding ("exit code 0") while both had actually
failed with "No such file or directory". Capture the real status:

```bash
rsync ... ; rc=$?          # not: rsync ... | tail -3; rc=$?
```

**`pgrep -f somescript.sh` matches its own ssh command line** and reports a
job as running when it never started. Check for the job's log file instead.

**`mkdir -p` over ssh can time out** exactly when you need it (link busy),
and then the rsync that depends on it fails. Retry it, and confirm the
directory exists before starting the transfer.

**Long jobs must be detached** or an ssh drop kills them:

```bash
ssh $H 'cd ~/proj && setsid nohup ./run.sh > /tmp/run.log 2>&1 < /dev/null & echo LAUNCHED'
```

## Startup checklist

```bash
H=homa@<ip>
ssh-keygen -R <ip>                      # the IP gets recycled between rentals
ssh -o StrictHostKeyChecking=accept-new $H 'nvidia-smi; python3 -c "import torch;print(torch.__version__, torch.cuda.is_available())"'
```

1. **Start the big data transfer in the background immediately.** It is the
   long pole; everything else fits inside it.
2. Push the repo: `rsync -az --exclude '/data/' --exclude '/outputs/' --exclude '.git' --exclude '.venv' ./ $H:proj/`
3. Verify the loader packages survived (see the exclude trap above).
4. `pip install --user` the missing deps. Do NOT build a venv.
5. Start the continuous results-pull loop on the laptop.
6. Launch the runs detached; watch the log, not the ssh session.
7. Set reminders for the subscription end with real slack (15 min), and
   pull everything before it.

## Timings from the 2026-09-17 rental

Useful for deciding what fits in an hour:

| | |
|---|---|
| deps install (system python3) | ~1 min |
| CWRU 179 MB upload | ~7 min |
| Paderborn N09 subset, 63 MB | ~3 min |
| Paderborn full 437 MB | ~15-20 min |
| arm A run (train + evaluate, warm cache) | **~40 s** |
| 16 arm A runs | ~6 min |

The compute is never the constraint. The uplink is.
