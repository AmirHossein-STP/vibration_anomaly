# Runbook: `vibration`

**Question it answers:** "does this recording look like the healthy vibration this machine produced when I trained the model?"

## Scope and honesty
* One model = **one machine** (the network learns what that machine sounds like; the study showed it does not move to other machines).
* On the public CWRU bench it detected every fault (AUC 1.0) but that bench is easy; **no number here applies to your machine** until you test it.
  The study's threshold came from 56 windows and gave 3.5% false alarms instead of 1% -> this service insists on more healthy data.
* Recall on real faults is **unmeasured** by construction: the model never sees a fault. Plan an injected-fault or seeded test before relying on it.
* Fault type is not reported.

## What you need
* Healthy recordings of the machine: at least ~500 training windows and 50 validation windows (about 1 min is the absolute floor; **tens of minutes** across normal operating conditions is the real recommendation). 1 window = 0.7 s.
* The sampling rate (>= 2 kHz; resampled to 12 kHz internally, so signals should carry content below 6 kHz).

## First time
```bash
./ad vibration init
nano <DATA_DIR>/vibration/config/service.yaml     # set fs_hz
cp healthy_*.npy <DATA_DIR>/vibration/train/
./ad vibration train --seeds 3                    # CPU, minutes; prints the candidate's report
./ad vibration show <version>                     # per-seed validation error, threshold, spread across seeds
./ad vibration promote <version>
```
Every file is split in time: the first 80% trains, the last 20% (after a 1-window gap) sets the threshold.
A warning about "threshold differs between seeds" means the model is unstable on that little data.

## Every day
```bash
cp new_*.npy <DATA_DIR>/vibration/in/
./ad vibration infer --archive
```
`verdict`: `healthy` < 10% windows over threshold <= `watch` < 30% <= `faulty` (defaults in `verdict:`; not validated on your machine).

## When the machine changes (new baseline after maintenance)
`./ad vibration calibrate` (new healthy files in `train/`) keeps the weights and only re-sets the threshold;
`./ad vibration train` retrains from scratch. Both only create candidates: review, then `promote`.
