# Data directory

## `raw/`

64 `.npz` files from the [CWRU_Bearing_NumPy](https://github.com/srigas/CWRU_Bearing_NumPy)
mirror of the Case Western Reserve University Bearing Data Center dataset,
downloaded by `scripts/01_download_data.sh` (→ `vibration_ad.data.download`).
`MANIFEST.txt` is the exact list of files fetched (also used by
`download.py` as the source of truth for what to fetch).

The whole `raw/` directory (MANIFEST.txt + all 64 `.npz` files, ~187MB) is
**DVC-tracked**, not git-tracked directly: `data/raw.dvc` (small, git-tracked)
is the content hash/pointer, the actual data lives in `.dvc/cache` +
the working-tree checkout. See `notes/07_dvc_and_kaggle.md` for the full
DVC setup, including the current gap (no remote configured yet, so
`dvc push`/`pull` only works locally until one is added).

**Subset chosen**: only the 12kHz Drive-End files (`*_DE12.npz`) and the
`*_Normal.npz` baseline files, across all four RPM/load families (1730,
1750, 1772, 1797 RPM = 3, 2, 1, 0 HP motor load). We skip the 48kHz
Drive-End (`DE48`) and Fan-End (`FE`) variants to keep one consistent
sampling rate and sensor location throughout — see
`notes/01_dataset_notes.md` for why.

### Filename convention

```
{rpm}_Normal.npz
{rpm}_{fault}_{size_mils}_{end}.npz
```

| Field | Meaning | Values |
|---|---|---|
| `rpm` | motor speed (≈ load) at recording time | 1730, 1750, 1772, 1797 |
| `fault` | fault location | `IR` (inner race), `B` (ball), `OR@3`/`OR@6`/`OR@12` (outer race, clock position relative to the load zone) |
| `size_mils` | fault diameter, thousandths of an inch | 7, 14, 21, 28 (not every size exists for every fault type) |
| `end` | accelerometer position / sample rate | `DE12` (Drive-End, 12kHz) — the only one we use here |

Each `.npz` has one or more of the keys `DE`, `FE`, `BA` (accelerometer
arrays); we always read `DE`. Parsing is implemented in
`src/vibration_ad/data/catalog.py`.

### Ground truth

The fault fields above ARE the ground truth. We deliberately don't use them
for training (see `notes/00_project_overview.md`), but they're what makes
it possible to actually score the unsupervised detector in
`evaluate.py` — see `notes/04_evaluation_notes.md`.

## `processed/`

Generated cache, gitignored, produced by `vibration_ad.data.dataset.build_raw_datasets`:
`train_spectrograms.npy`, `val_spectrograms.npy` (+ `val_meta.csv`),
`test_spectrograms.npy` (+ `test_meta.csv`), `catalog.csv`. Delete this
directory (or pass `--force-data` to `train.py`) to force a full
re-windowing/re-STFT from `raw/`, e.g. after changing `configs/default.yaml`'s
`windowing`/`stft` sections.
