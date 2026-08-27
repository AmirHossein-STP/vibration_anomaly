# `data/hydraulic_raw/`

Expected contents at runtime (not committed -- populated by the Kaggle
notebook's step 4, copying from the attached
`jjacostupa/condition-monitoring-of-hydraulic-systems` input dataset; see
`scripts/build_kaggle_notebook_hydraulic.py`):

- `profile.txt` -- one row per cycle, 5 tab-separated columns (see
  `notes/09_hydraulic_notes.md` section 1 for the column meanings).
- One `<SENSOR>.txt` per sensor named in `configs/hydraulic.yaml`'s
  `data.sensor_groups` (17 files: `PS1`-`PS6`, `EPS1`, `FS1`-`FS2`,
  `TS1`-`TS4`, `VS1`, `CE`, `CP`, `SE`) -- tab-separated, one row per cycle,
  one column per sample within that cycle.

Unlike `data/raw/` (CWRU, DVC-tracked, ~187MB, committed as a DVC pointer),
there is no local copy of this dataset and it isn't DVC-tracked: it's
substantially larger (2205 cycles x up to 6000 samples/sensor) and, so far,
only ever touched inside the Kaggle kernel, same as how CWRU ran before a
local PyTorch venv existed. If a local run becomes worth setting up later,
`hydraulic_fusion_ad.data.catalog.build_catalog()` reads from whatever
`config.paths.raw_dir` resolves to -- point it at a manually-downloaded copy
of the dataset with the file layout above.
