# Runbook: `bearing`

**Question it answers:** "does this bearing recording look like a healthy one of *this* machine?"
Detection only (no fault type). Rolling-element bearings only (for fluid-film bearings this is the wrong tool).

## Scope and honesty
* Method: envelope-order features + a logistic regression fitted on public CWRU data (bundled), threshold set from **your** healthy recordings.
* Measured on Paderborn data it never saw (900 rpm): ~0.82 balanced accuracy at 10% window false alarms; at 1500 rpm ~0.69 (unexplained drop).
  A quick check through this very service on the same data: healthy recordings flagged `faulty` 4%, real-damage recordings flagged `faulty` 76% (`watch or faulty`: 19% / 97%).
* False alarms come **per bearing unit**, not at random. One healthy Paderborn unit (K005) read as abnormal in 40% of its windows. That is why `calibrate` reports each healthy recording and warns.
* Not validated on a turbomachine yet. Treat the first weeks as a trial.

## What you need
1. Bearing geometry from the datasheet: number of balls, ball diameter, pitch diameter (same unit), contact angle.
2. Shaft speed in rpm (constant in `config`, or per file in a sidecar) and the sampling rate of your files (>= 11 kHz).
3. Several HEALTHY recordings, >= 200 s in total (1 window = 1 s), from different days / loads / speeds if possible.

## First time
```bash
./ad bearing init
nano <DATA_DIR>/bearing/config/service.yaml       # fill bearing: {n_balls, ball_diameter, pitch_diameter}, fs_hz, rpm
cp healthy_*.npy <DATA_DIR>/bearing/train/        # .npy / .npz / .csv, one channel
./ad bearing calibrate                            # read the summary + WARNINGS
./ad bearing promote <version-printed-above>
```
`calibrate` says **READS ABNORMAL** if your own healthy recordings already reach the WATCH level. Do not rely on verdicts then:
add healthy data covering normal operating conditions first.

## Every day
```bash
cp new_*.npy <DATA_DIR>/bearing/in/               # optional per-file speed: new_1.npy.meta.json  ->  {"rpm": 1480, "fs_hz": 25600}
./ad bearing infer --archive
```
Read `out/<date>/<job>/results.csv`. Columns: `verdict` (`healthy` < 40% windows flagged <= `watch` < 50% <= `faulty`), `flagged_fraction`, `score` (median), `rpm`, `fs_hz`.
The 40% / 50% levels were measured on Paderborn (notes/14): with a threshold from other healthy bearings, about 10% of a NEW healthy bearing's recordings read `watch` and 5% `faulty` on average, but one healthy bearing in six alarmed far more. They are a starting point, not validated limits for your machine (`verdict:` in the config).

## Messages you may see
| Message | Meaning / action |
|---|---|
| `sampling rate unknown` / `shaft speed unknown` | set `fs_hz` / `rpm` in config or a sidecar |
| `rpm=... too low / too high` | the 1 s window and 12 kHz analysis range limit measurable speeds (about 300 rpm up to the BPFI harmonics limit) |
| `too short` | recording shorter than 1 s after resampling |
| `NaN/inf samples`, `constant (flat line)` | bad recording / sensor disconnected; fix the data |

## Refitting the public-data model (rare)
Only if you want a different source set: put labelled CWRU `.npz` files in `source/`, run `./ad bearing train`, then
`./ad bearing calibrate --base <that version>`. The bundled model scores 0.88 +- 0.20 balanced accuracy in CWRU cross-validation split by file.
