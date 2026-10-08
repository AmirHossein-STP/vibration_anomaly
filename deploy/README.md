# Operating the three services

Implements `DEPLOYMENT_PLAN.md`: files in, files out, command line, a person approves every model.

| Service | What it does | Needs from you | Runbook |
|---|---|---|---|
| `bearing` | Healthy/watch/faulty verdict per recording for a **rolling-element bearing**, any machine | bearing geometry, shaft speed, sampling rate, a few HEALTHY recordings | [runbooks/bearing.md](runbooks/bearing.md) |
| `vibration` | Autoencoder trained on ONE machine's healthy vibration; flags what it cannot reconstruct | tens of minutes of HEALTHY recordings of that machine | [runbooks/vibration.md](runbooks/vibration.md) |
| `hydraulic` | Per-component (cooler/valve/accumulator, pump experimental) anomaly score for 60 s, 17-sensor cycles | cycle files + `status.csv` of known conditions | [runbooks/hydraulic.md](runbooks/hydraulic.md) |

## One-time setup (server)

```bash
cp deploy/.env.example deploy/.env      # set REGISTRY, TAG, HOST_UID/GID (id -u / id -g), DATA_DIR, MODELS_DIR
cd deploy
docker compose pull                      # or: docker compose build   (when building locally)
./ad bearing init                        # creates the folders + a starter config/service.yaml; repeat per service
./ad bearing doctor
```

## The folder contract (same for every service)

```
<DATA_DIR>/<service>/in/        you drop files to score          (sidecars: <file>.meta.json)
<DATA_DIR>/<service>/train/     you drop HEALTHY files for train / calibrate
<DATA_DIR>/<service>/config/    service.yaml  (edit this; service.example.yaml shows every setting)
<DATA_DIR>/<service>/out/<date>/<job_id>/   results.csv, results.json, manifest.json, run.log
<DATA_DIR>/<service>/out/score_log.csv      one line per infer job (watch this for drift)
<MODELS_DIR>/<service>/versions/<version>/  models (never edit by hand: promote checks hashes)
<MODELS_DIR>/<service>/ACTIVE               which version scores files
```

## Commands (all services)

| Command | Effect |
|---|---|
| `init` / `doctor` | create folders and a config / check folders, config, active model |
| `train` | build a **candidate** model from `train/` (never changes the active model) |
| `calibrate` | build a **candidate** with a threshold set from healthy files in `train/` |
| `models` / `show <v>` | list versions / show a version's report and integrity |
| `promote <v>` | **approval step**: makes a candidate the active model (asks you to type `promote`; `--yes` skips) |
| `rollback` | go back to the previously active model |
| `infer [--archive] [--model v]` | score everything in `in/` with the active model |

Exit codes: `0` ok, `1` finished but some files were rejected (or a doctor check failed), `2` could not run.

## Result files

`results.csv` / `results.json` rows share these columns (services add a few):
`file, status (ok|error), verdict, score, flagged_fraction, n_windows, message, <service columns>, model_version, job_id`.
`verdict` is `healthy | watch | faulty` (bearing, vibration) or `healthy | anomaly` per component (hydraulic).
`status=error` rows carry the reason in `message` and never stop the other files.
`manifest.json` records SHA-256 of every input, the model version and config hash, so any result can be traced.

## Daily operation

```bash
./ad bearing infer --archive            # score new files, move scored ones to in/processed/<job>/
# cron, every 15 min:
*/15 * * * *  cd /srv/ad/deploy && ./ad bearing infer --archive >> /var/log/ad-bearing.log 2>&1
```

Look at `out/score_log.csv`: a healthy score that creeps up for weeks, or every file flagged at once, means
"re-calibrate with fresh healthy data" (or "the sensor / machine changed"), not "everything broke".

## Updating and rolling back software

```bash
REGISTRY=registry.example.com/team deploy/release.sh      # on the build machine (tests + push, tag = git commit)
# on the server:
sed -i 's/^TAG=.*/TAG=<new commit>/' deploy/.env && docker compose pull && ./ad bearing doctor
# going back = set the previous TAG again. Models are on the volume, so they survive image changes.
```

## Back up

`MODELS_DIR` (small) and `DATA_DIR/<service>/{config,out}` nightly to another machine. Raw input files are yours to keep.

## What this does NOT claim

Read the "scope" section of each runbook. In short: CWRU/ZeMA accuracy numbers do not transfer to your machine;
the hydraulic pump model is experimental; the autoencoder is per-machine; fault *type* is never reported.
