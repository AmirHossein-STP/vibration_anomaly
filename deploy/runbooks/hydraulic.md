# Runbook: `hydraulic`

**Question it answers:** "is this 60 s cycle abnormal for component X?" - one detector per component, each trained only on cycles where
that component is healthy.

## Scope and honesty (from the study on the ZeMA test rig)
| Component | Study result | What to do |
|---|---|---|
| cooler | ROC-AUC 1.00, but the score tracked campaign time (r = 0.88) | promising, unproven: check that it is not just "later in the log" |
| valve | ROC-AUC 0.89; the fixed 99th-percentile threshold gave recall 0 | usable **only with a calibrated threshold** (this service does that) |
| accumulator | ROC-AUC 0.87; same remark | same |
| pump | ROC-AUC 0.66 | **experimental**, off by default (`--components pump`); never act on it alone |

This is one rig's sensor layout (17 sensors, 7x100 Hz + 2x10 Hz + 8x1 Hz). A different machine needs a new layout in code and retraining.

## Input format
* **One cycle = one `.npz`** with arrays named `PS1..PS6, EPS1` (6000 samples), `FS1, FS2` (600), `TS1..TS4, VS1, CE, CP, SE` (60).
* For `train`: also `train/status.csv` - columns `file,cooler,valve,pump,accumulator` (ZeMA condition codes: cooler 100/20/3, valve 100/90/80/73, pump 0/1/2, accumulator 130/115/100/90), **rows in time order**. Blank = unknown.
* The public ZeMA files convert with: `python -m serving_core.tools.zema_to_cycles <raw_dir> <out_dir>`.

## First time
```bash
./ad hydraulic init
cp cycles/*.npz status.csv <DATA_DIR>/hydraulic/train/
./ad hydraulic train --components cooler,valve,accumulator --seeds 3     # CPU: tens of minutes
./ad hydraulic show <version>      # per component: threshold, calibration quantiles, and (if degraded cycles were labelled) ROC-AUC / recall
./ad hydraulic promote <version>
```
* Threshold = a quantile of a **held-out healthy calibration block** for `calibration.target_false_alarm` (default 5%). Choose that rate for your process; it is a business decision.
* Needs >= 150 healthy cycles per component. With no labelled degraded cycles in `status.csv`, detection performance is reported as UNMEASURED.

## Every day
```bash
cp cycles/*.npz <DATA_DIR>/hydraulic/in/
./ad hydraulic infer --archive
```
One row per cycle and component: `verdict` (`healthy | anomaly`), `score`, `threshold`, per-branch errors `error_slow / error_medium / error_fast`
(which sensor family moved: slow = temperatures/efficiency, medium = flows, fast = pressures/motor power), `component_note`.
A single cycle over threshold is not an alarm by itself; look at runs of cycles.
