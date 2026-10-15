# Deployment plan: three services, handed to someone else

> **Status 2026-10-15: historical record; the forward plan is [MASTER_PLAN.md](MASTER_PLAN.md). Section 8 (implementation status) remains the description of what `src/serving_core/` implements.**

Goal (from the request): each subproject runs on a server on its own, takes
**new inputs**, writes **outputs in a fixed format**, and lets an operator
**trigger training, then go back to inference**. The operator is a different
person, so nothing may depend on this repo's folder layout, on Kaggle, or on
knowing the code.

## 1. Short answer

**Yes, use Docker** — one image per subproject, run together with Docker
Compose. Docker alone is not enough, though. Today the code is *research code*
(scripts, notebooks, hard-coded paths). Docker would only freeze that. The
work that makes it "production" is a thin **service layer** in front of each
pipeline: a fixed input contract, a fixed output contract, a model registry,
and a training job runner. Docker is the last 20%.

Why Docker here, specifically:

| Reason | Evidence from this repo |
|---|---|
| Reproducibility of the environment | Torch version, CUDA build and the "second box did not reproduce" investigation (`notes/12_reproduction_investigation.md`) cost days |
| Hand-off to a non-author | The README says "needs `.venv/` at the project root; scripts call `.venv/bin/python`" — fragile for anyone else |
| Three projects, three dependency sets | Project 3 needs scipy/sklearn/torch; project 2 only torch+numpy; isolation avoids version fights |
| Rollback | Image tag + model version = a complete, restorable release |

Alternatives considered and rejected:

- **venv + systemd, no Docker**: fine for one box and one operator who knows
  Python. Rejected because the operator is someone else and you want
  clean rollback.
- **Kubernetes / cloud ML platforms**: overkill for three small models (the
  largest trains in minutes on CPU). Revisit only if there are many sites.
- **Kaggle as the runtime**: it is a development tool (no internet, no
  scheduling, no stable I/O). Not a production target.

## 2. Target architecture (decided: files in, files out, CLI operator)

Decisions from the owner (2026-10-06): CPU server now (GPU maybe later);
inputs are **files**; outputs are **files**; the operator is comfortable with a
**command line**; a **person approves** every new model; the server **can pull**
images from a registry. So the primary interface is a CLI plus a folder
contract. An HTTP API is *not* built now (add later only if needed).

```
  operator                       one container per subproject (same image shape)
  --------   docker compose run  +----------------------------------------------+
  drops files ------------------>|  CLI:  infer | train | calibrate | models |   |
  in  /data/<svc>/in             |        promote | rollback | doctor            |
                                 +----------+-----------------------------------+
  reads results <----------------- /data/<svc>/out/<YYYY-MM-DD>/<job_id>/ (JSON + CSV)
                                            |
                          volumes (outside the image, backed up):
                          /data/<svc>/train   healthy data for training/calibration
                          /models/<svc>/<version>/   versioned model registry
```

Folder contract per service (same names everywhere):

| Folder | Who writes | Content |
|---|---|---|
| `in/` | operator | new files to score (format per service, section 3) |
| `train/` | operator | healthy files for (re)training or calibration |
| `out/<date>/<job_id>/` | service | `results.json`, `results.csv`, `run.log`, `manifest.json` (input hashes, model version, threshold) |
| `/models/<svc>/<version>/` | service | checkpoint, normalizer, resolved config, **threshold**, metrics, config+data hash |
| `/models/<svc>/ACTIVE` | `promote` / `rollback` only | pointer to the active version |

Commands (all run as `docker compose run --rm <svc> <command>`):

1. `infer` - score everything in `in/` with the ACTIVE model; write `out/...`.
   Bad files are reported per file in `results.json` (they never abort the batch).
2. `train` - train a **candidate** on `train/`; writes a new version folder and
   a metrics report (several seeds, spread shown). It never changes ACTIVE.
3. `calibrate` - fit/set the threshold from known-healthy files; prints a
   verdict ("reads clean" / "healthy baseline already reads as faulty").
4. `models` - list versions with metrics and which one is ACTIVE.
5. `promote <version>` - **the human approval step**; refuses a version whose
   report is missing or whose config/data hash does not match.
6. `rollback` - back to the previous ACTIVE.
7. `doctor` - checks folders, permissions, input formats, disk space, versions.

Operator flow: drop files in `in/` -> `infer` -> read `out/`. Retrain flow:
add healthy files to `train/` -> `train` -> read the candidate report ->
`promote` (or leave it) -> `infer`.

Shared code: one small internal package (`serving_core`) with input
validation, registry, manifest writing and logging, reused by all three
services; each project keeps its own science code underneath. Automation is
just cron or a systemd timer calling `infer` on the folder.

Release: CI builds each image, tags it with the git commit, pushes it to your
registry; the server does `docker compose pull && docker compose up`. Pin
`python:3.12-slim` and CPU-only torch wheels (smaller images, no CUDA).
GPU later = a separate `*-gpu` image tag used only for `train`; nothing else
changes. No secrets in images; the registry login stays on the server.

## 3. Per-subproject specifics and the blockers you must know about

### 3.1 Subproject 1: CWRU autoencoder (`vibration_ad`)

- **Input contract**: single-channel vibration, 12 kHz, ≥ 8384 samples
  (~0.7 s). Today only `.npz` with key `DE` works → add `.npy`/`.csv`,
  explicit `fs`, resampling, length check.
- **Output**: per-window error + flag, per-recording flagged fraction,
  `model_version`, `threshold`.
- **Training input**: a folder of *known-healthy* recordings of the machine
  (no labels). Threshold = 99th percentile of a held-out healthy part.
- **Blockers**: `train.py` assumes the CWRU file naming/RPM split
  (`data/dataset.py`); production training needs a "just a folder of healthy
  files" data path with a random-by-time validation split. Threshold from
  only 56 validation windows was unstable (3.5% false alarms vs the nominal
  1%) → require a minimum healthy duration and warn below it.
- **Effort**: small. Training is CPU-only and fast.

### 3.2 Subproject 2: hydraulic sensor fusion (`hydraulic_fusion_ad`)

- **Input contract**: one cycle = 17 series at three native rates
  (7×6000, 2×600, 8×60 samples per 60 s). Define it as one folder or one
  `.npz` per cycle with named channels; validate shapes and units.
- **Output**: per cycle and per component (cooler/valve/pump/accumulator):
  score, threshold, flag, per-branch errors (for diagnosis).
- **Training input**: cycles that are healthy *for a given component*
  (needs a status column like `profile.txt`).
- **Blockers (important)**: (a) threshold must come from a held-out healthy
  *calibration set* and a target false-alarm rate chosen by the business, not
  the fixed 99th percentile (valve/pump recall was ~0 with it);
  (b) the **pump** model is not good enough (ROC-AUC 0.656) → ship it
  disabled/"experimental"; (c) the cooler score is confounded with
  campaign time (r = 0.88) → do not market it as validated; (d) it is tied to
  this one test rig's sensor layout — a different machine means a new
  `sensor_groups` config and retraining.
- **Effort**: medium (the multi-rate input format is the real work).

### 3.3 Subproject 3: domain adaptation / zero-shot bearing detection

- **Blocker #1 (do this first)**: `domain_adaptation/infer.py` loads the
  *spectrogram baseline*, which the project showed does **not** transfer
  between machines. The method that works (envelope-order features + logistic
  regression + healthy-only calibration) exists only as the analysis script
  `scripts/analysis/envelope_transfer.py`. Productionising means: move it into
  a module, **save** the scaler and coefficients, and add a calibration step
  that stores a per-machine threshold. Do not deploy the current `infer.py`.
- **Input contract**: signal file(s), `fs`, `rpm` (or a speed channel),
  and a **bearing geometry** (number of balls, ball diameter, pitch diameter,
  contact angle) in a small YAML. It only applies to rolling-element bearings.
- **Training input**: here "training" has two meanings. (1) The fit on public
  labelled data (CWRU, optionally Paderborn) — rarely re-run, seconds on CPU.
  (2) **Calibration** on the operator's own *healthy* recordings — this is the
  step the operator will actually trigger.
- **Output**: flagged fraction per recording, verdict, calibration health
  ("this machine's healthy baseline already reads as faulty" warning), model
  and threshold version.
- **Known weaknesses to document for the operator**: strong at 900 rpm
  (balanced 0.82) but weak at 1500 rpm (0.69, unexplained); fault *typing* is
  not reliable — report detection only.
- **Effort**: medium, mostly refactoring plus one new calibration CLI.

## 4. Phases

| Phase | Work | Output |
|---|---|---|
| 0. Decisions (you) | Answer the questions in §6 | One-page spec per service |
| 1. Core library | `serving_core`: input validation, registry, job runner, structured logs, config + data hash in every model version | Library + tests |
| 2. Productionise science code | Fix the blockers in §3 (esp. 3.3 and the threshold calibration in 3.2); remove hard-coded paths; every model run writes a self-contained version folder | Versioned models loadable by path |
| 3. CLI | `infer/train/calibrate/models/promote/rollback/doctor` per project over the folder contract | Three runnable CLIs |
| 4. Docker | One `Dockerfile` per service (pinned `python:3.12-slim`, CPU torch wheel, non-root user), `docker-compose.yml`, volumes, CI that builds and pushes to the registry | `docker compose run` works on a clean machine |
| 5. Tests | Smoke tests exist for all three; add contract tests (bad shape, wrong `fs`, short signal), a golden-output test per model, an end-to-end "train -> promote -> infer -> rollback" test in CI | Green pipeline |
| 6. Hand-off | Operator runbook (one page each): folder contract, how to train, how to roll back, what each warning means, who to call | `RUNBOOK_*.md` |
| 7. Monitor | Each `infer` run appends score statistics to a daily CSV; a cron job flags drift (healthy score creeping up) and "all windows flagged" | Daily report file |

Rough size: phases 1–4 are on the order of 2–4 weeks of focused work for the
three services together, dominated by phase 2 for subprojects 2 and 3.
Treat this as an estimate, not a commitment; the multi-rate input format and
packaging the envelope pipeline are the two largest unknowns.

## 5. Lessons from this repo that must be built in

- **A stale cache invalidated a whole experiment** (`_cache_fingerprint`
  fix). Every model version stores a hash of config + training-data manifest;
  inference refuses a mismatched normalizer.
- **Artifacts were lost when a rented server was deleted.** Model versions
  and job logs live on a mounted volume, with a nightly copy off the box.
- **A fixed 99th-percentile threshold was wrong for 3 of 4 hydraulic
  components.** Threshold = an explicit, stored, per-model value from a
  held-out healthy calibration set, with the target false-alarm rate recorded.
- **Single-seed results misled for weeks.** A training job should train
  several seeds and report the spread (or ensemble them) before a candidate
  can be promoted; promotion is always a human decision.
- **Per-unit false alarms.** A machine whose healthy data already reads as
  faulty will do so consistently — keep the calibration verdict as a hard gate
  before monitoring starts.
- **Do not claim more than was measured.** Each service's `/version` and docs
  carry its validated scope (e.g. "hydraulic pump: experimental").

## 6. Decisions recorded (2026-10-06)

| Question | Answer | Consequence |
|---|---|---|
| GPU? | Not now, maybe later | CPU images; optional `-gpu` tag for `train` later |
| Input form | Files | Folder contract, no collector/streaming component |
| Output form | Files | JSON + CSV with a documented schema (still to define, see below) |
| Operator | Comfortable with CLI | No web UI; runbook with copy-paste commands |
| Retraining | A person approves | `train` only makes candidates; `promote` is manual |
| Registry | Server can pull | Normal registry flow, no tarball transfer |

Still open (small, can be decided during phase 1): exact output schema
(column names, units, one row per window or per recording), file formats
accepted per service (`.npy/.npz/.csv`; for the hydraulic rig one folder or
`.npz` per cycle), and the minimum healthy duration `train`/`calibrate`
must see before they will produce a model.

## 7. Recommended first step

Start with **subproject 3's packaging** (the envelope-order pipeline as a
saved model + calibration CLI) because it is the one where "what is deployed"
and "what works" currently disagree, and it exercises every part of the
shared core (input contract, calibration, versioning). Then subproject 1
(simplest, proves the Docker/registry flow end to end), then subproject 2.

## 8. Implementation status (2026-10-08)

Built, following sections 2-6:

| Piece | Where |
|---|---|
| Shared core: folder contract, versioned registry with hash checks, `promote`/`rollback` (human approval), per-file error isolation, job manifests, `score_log.csv` | `src/serving_core/` |
| `bearing` service (subproject 3): envelope-order features, bundled CWRU source model, `calibrate` from healthy files, watch/faulty verdicts | `src/serving_core/services/bearing.py` |
| `vibration` service (subproject 1): multi-seed autoencoder `train`, `calibrate` (new threshold, same weights), minimum-data checks | `.../services/vibration.py` |
| `hydraulic` service (subproject 2): per-component detectors, calibrated thresholds, labelled-cycle check, pump opt-in only; ZeMA converter `python -m serving_core.tools.zema_to_cycles` | `.../services/hydraulic.py` |
| Docker: one multi-target `Dockerfile`, compose, `deploy/ad` operator wrapper, `deploy/release.sh` (build, smoke-test, push) | `deploy/` |
| Operator docs | `deploy/README.md`, `deploy/runbooks/*.md` |
| Tests | `tests/test_serving_core.py` (7, numpy only), `tests/test_serving_torch.py` (4, needs torch) |

Verified:
* Bearing service on REAL Paderborn data (900 rpm, 72 healthy calibration recordings, 268 scored): `faulty` verdict on 4% of healthy and 76% of real-damage recordings; `watch` or `faulty` on 19% / 97%. The calibrate step flagged the known problem unit (K005). Same numbers on the host (numpy 2.5) and inside the Docker image (numpy 1.26).
* All 11 tests pass on the pinned versions (torch 2.2.2, numpy 1.26.4), run on the host.
* All three images build; `init` and `doctor` run in each. Docker caught one real bug (a username lookup that fails under an arbitrary container uid), now fixed with a regression test.

NOT verified, so do not assume:
* The torch tests inside the vibration/hydraulic images: the Docker Desktop VM went read-only (disk/memory limit of that VM) during the run. Rerun on a server: `docker run --rm --user root --entrypoint sh -v $PWD/tests:/app/tests -w /app ad-hydraulic:<tag> -c "pip install pytest && python -m pytest tests"`.
* `deploy/release.sh`, `deploy/ad` and the compose file were written but not executed (no registry here; the compose file was not run end to end).
* Real-data behaviour of `vibration` and `hydraulic`: only synthetic data was available locally (no ZeMA raw files; CWRU is not a "your own machine" test). Their verdict levels (`watch_fraction`, `faulty_fraction`) are defaults, not validated limits.
* The bearing `watch`/`faulty` levels (25% / 50% of windows) are likewise engineering defaults; the study only measured window-level rates.

Method note found while building: the CWRU in-domain figure quoted in notes/13 (0.971 +- 0.032) came from folds split by WINDOW; split by FILE it is 0.88 +- 0.20. The bundled model uses the honest number in its report.
