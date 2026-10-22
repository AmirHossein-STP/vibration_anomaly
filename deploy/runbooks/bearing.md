# Runbook: `bearing` (method v1.0)

Monitors rolling-element bearings on many machines. Each **unit**, meaning one bearing position, is
compared with **its own healthy history**. No fault data and no training are needed. The method and
the evidence behind every number below are in `notes/22_method_v1.md` (the specification) and notes/17–26.

## Scope and honesty
- **What it detects:** damage that changes the bearing's envelope spectrum against that unit's
  own healthy baseline. It names the race (inner/outer) when the bearing geometry is known.
  - Tested on 7 public rigs and 11 bearing types.
  - Typing: 0.74–0.83 on most rigs, up to 1.00 on MFPT.
  - Detection AUC 0.84–1.00 with the unit's own baseline.
  - One bearing type in five failed (HUST 6205): its healthy data was not self-consistent.
- **What it does not cover:**
  - ball and cage faults (mostly missed);
  - machines without a shaft-speed signal;
  - a brand-new unit with no healthy history of its own (a fleet baseline is not adequate).
- **Its false-alarm rate is set, not guaranteed.** At the default settings, expect about 10% of
  healthy windows over threshold and about one trend event per day of healthy operation
  (notes/26). Treat `watch` as advisory.

## What you need, per unit
1. **Healthy recordings**: at least 30 min in total, spread over the unit's operating states,
   each recording ≥ 1 s (better 10 s or more), accelerometer on or near the bearing housing.
2. **Shaft speed for every recording** (rpm), required. Give it per file in a sidecar
   `<file>.meta.json` with `{"rpm": 1480}`, per unit in the config, or globally. Use the
   tachometer, or the drive's logged setpoint if it is speed-controlled. Speed estimated
   from vibration is not good enough (notes/20).
3. **The bearing**, in one of these forms:
   - a catalog designation (`python -c "from domain_adaptation.bearings import CATALOG; print(sorted(CATALOG))"`);
   - the geometry: `n_balls`, `ball_diameter`, `pitch_diameter` and `contact_angle_deg`;
   - the manufacturer's fault factors: `bpfo` and `bpfi` in orders of shaft speed.

   With none of these the unit still runs, in **geometry-free** mode. It detects, but it
   does not type, and it needs a longer baseline (notes/23).

## Folder layout (one sub-folder per unit)
```
data/bearing/
  config/service.yaml
  train/<unit>/*.npy|npz|csv     healthy recordings (+ optional <file>.meta.json)
  in/<unit>/*.npy|npz|csv        new recordings to score
  out/                           results per job, score_log.csv, bearing_trend.csv
```
Files placed directly in `train/` or `in/` belong to `default_unit`, unless their sidecar says
`{"unit": "..."}`. Use unique file names across units: `--archive` moves processed files
into one folder.

## Config example (`config/service.yaml`)
```yaml
fs_hz: 25600
rpm: 1480                       # default; sidecars / units override
bearing: {designation: "SKF 6205-2RS JEM"}
units:
  pump_A_DE: {bearing: {designation: "SKF 6205-2RS JEM"}}
  pump_A_NDE: {bearing: {n_balls: 8, ball_diameter: 7.94, pitch_diameter: 33.5}}
  fan_3: {bearing: {bpfo: 3.05, bpfi: 4.95}, rpm: 2950}
  old_blower: {bearing: {designation: null}}     # unknown bearing -> geometry-free
```

## First time
1. `ad bearing init`, then put healthy recordings in `train/<unit>/` and fill in the config.
2. `ad bearing calibrate`. Read every unit's line:
   - **`reads clean`**: each healthy recording, judged against the *rest* of that unit's baseline
     (cross-fitted), stays below the watch level.
   - **`READS ABNORMAL`**: the unit's "healthy" data is not self-consistent. One recording looks
     unlike the others, which is the most common failure in every study. Add healthy data
     spanning the operating states, or find out what differs, **before** trusting that unit.
3. `ad bearing promote <version>`. Calibration never activates a model by itself.

## Every day
- Put recordings in `in/<unit>/` and run `ad bearing infer --archive`. Per file you get:
  - **`verdict`**: `healthy`, `watch` (≥ 40% of windows over threshold) or `faulty` (≥ 50%);
  - **`fault_type`** and **`dominant`**: which race, and which line rose most (z against baseline);
  - **`level_z`, `physics_z`, `cusum_level`, `cusum_physics`, `trend_alarm`**: the unit's trend
    across recordings, kept in `out/bearing_trend.csv`. Under constant speed and load the level
    CUSUM is the better early trigger (notes/18). The physics score says what it is.
- A unit not in the active model is **rejected**, not scored. Calibrate it first.

## When to recalibrate
- After maintenance that changes the bearing, the sensor or its mounting.
- When the unit starts running in an operating state its baseline did not cover.
- Old versions stay in the registry: `ad bearing rollback` returns to the previous one.

## Messages you may see
- `shaft speed unknown` / `rpm ... too low/high`: give the speed. Below 300 rpm, 1 s windows are too short.
- `unit X has no baseline`: put its healthy recordings in `train/X/` and calibrate.
- `calibrated with method ...; this code runs 1.0`: the model predates method v1.0. Recalibrate.
- `geometry unknown -> geometry-free detection`: expected when the bearing is not given.
