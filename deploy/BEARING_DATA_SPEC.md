# What a site must record for bearing monitoring (method v1.0)

Hand this to anyone installing sensors or exporting data for the `bearing` service.
Every requirement comes from a measured result; the note in brackets is the evidence.

## Hard requirements (without these the service refuses or is unreliable)
1. **Raw acceleration waveforms**, not only overall values or spectra.
   - Sampling rate **≥ 25 kHz** (the method uses up to 6 kHz after resampling to 12 kHz; the
     margin protects against aliasing).
   - Each recording **≥ 1 s**, better **10 s**. 0.1 s snapshots cannot resolve fault lines
     (notes/26).
2. **Shaft speed for every recording.**
   - A tachometer, keyphasor or encoder is preferred. It is required if speed changes within a
     recording, because the envelope is then order-tracked (notes/17).
   - For speed-controlled drives, the **logged setpoint per recording** is acceptable.
   - Speed estimated from vibration is **not** acceptable: it cost 0.14 AUC (notes/20).
3. **Bearing identity per measurement point**, in one of these forms:
   - the designation (manufacturer + part number);
   - the geometry (number of balls, ball diameter, pitch diameter, contact angle);
   - the manufacturer's fault frequency factors (BPFO/BPFI).

   Without it the unit runs geometry-free: it detects, but it cannot say which race (notes/23).
4. **Healthy baseline per unit.**
   - **At least 30 min** of recordings from a period known to be healthy, **covering every
     operating state** the unit will run in (notes/26, notes/24 §4).
   - The baseline must be the unit's **own**: another unit's data is not a substitute (notes/17 §5.2).
5. **Operating state per recording**: speed, plus load if it varies (notes/25).

## Strongly recommended
- An accelerometer on the bearing housing, radial, stud- or adhesive-mounted (not magnetic, for
  high frequencies). Note its sensitivity and the unit of the stored values (g or m/s²).
- A recording at least every 10 min, so the trend statistic sees a rise across several recordings
  (notes/18, notes/26).
- A **maintenance and alarm log**: every alarm with what was found. The first confirmed fault on a
  unit is the most valuable data the project can get (MASTER_PLAN §3, "first real label").
- Unique file names, and one folder per unit (`train/<unit>/`, `in/<unit>/`).

## Known limits to tell the site
- Ball and cage faults are mostly not detected (notes/17, 19, 24).
- About 10% of healthy windows exceed the threshold by design. The trend alarm gives about one event per
  day of healthy operation (notes/26). Treat `watch` as advisory.
- Accelerated laboratory tests gave minutes to hours of warning. Real machines degrade
  more slowly, but this has not been measured yet.

See also: `deploy/inventory_template.csv` (one row per measurement point) and
`deploy/PILOT_ACCEPTANCE.md` (what must hold before monitoring starts).
