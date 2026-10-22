"""Condense one Paderborn LDM run-to-failure bearing (time-varying load and
speed) from the lossless int16 repack (128 kHz for B01-B09, 64 kHz from B10)
to 12 kHz, the rate every
analysis here uses. Called per bearing by fetch.py's `stream` stage, so the
50 GB set never has to fit on the vault or the SSD at once.

Input  (<raw-dir> = data/<B>/): vibration.npy int16 (records, 2, 204800),
       meta.json (step/offset volts per channel, 128 kHz), CSVs, log PDF.
Output (<out>/):
    vibration_12k.npy  int16 (records, 2, 19200), SAME volt grid as the source:
                       volts = offset[c] + step[c] * code, g = volts * g_per_volt
    meta.json          the source meta plus fs_hz 12000 and the clip count
    *.csv, *.pdf       copied unchanged (operating conditions per record, log)

Resampling is scipy resample_poly 3/32 on volts, then rounded back to the
source's own quantisation step, so the only added error is <= step/2: the
same size as the original quantisation.

Usage: python remote/datasets/condense_ldm.py --raw-dir <data/B01> --out <dir>
"""
from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path

import numpy as np
from scipy.signal import resample_poly

CHUNK = 32
RATIOS = {128000: (3, 32), 64000: (3, 16)}   # B01-B09 at 128 kHz, B10-B17 at 64 kHz (description PDF §2)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--raw-dir", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--src", help="unused; uniform condenser interface")
    a = ap.parse_args()

    meta = json.loads((a.raw_dir / "meta.json").read_text())
    fs_in = int(meta["sampling_rate_hz"])
    if fs_in not in RATIOS:
        raise SystemExit(f"unexpected sampling rate {fs_in} Hz (expected 128 or 64 kHz)")
    UP, DOWN = RATIOS[fs_in]
    codes = np.load(a.raw_dir / "vibration.npy", mmap_mode="r", allow_pickle=False)
    n_rec, n_ch, n = codes.shape
    step = np.asarray(meta["step_volts"], dtype=np.float64)[:, None]
    offset = np.asarray(meta["offset_volts"], dtype=np.float64)[:, None]
    n_out = n * UP // DOWN
    a.out.mkdir(parents=True, exist_ok=True)
    out = np.lib.format.open_memmap(a.out / "vibration_12k.npy", mode="w+", dtype=np.int16,
                                    shape=(n_rec, n_ch, n_out))
    clipped = 0
    for i in range(0, n_rec, CHUNK):
        volts = offset + step * np.asarray(codes[i:i + CHUNK], dtype=np.float64)
        res = resample_poly(volts, UP, DOWN, axis=-1)
        q = np.rint((res - offset) / step)
        clipped += int((q > 32767).sum() + (q < -32768).sum())
        out[i:i + CHUNK] = np.clip(q, -32768, 32767).astype(np.int16)
    out.flush()
    del out

    meta.update({"fs_hz": 12000, "condensed_from_hz": fs_in, "resample": f"scipy resample_poly {UP}/{DOWN} on volts",
                 "records": n_rec, "samples_per_record": n_out, "clipped_samples": clipped,
                 "file": "vibration_12k.npy"})
    (a.out / "meta.json").write_text(json.dumps(meta, indent=1))
    for p in a.raw_dir.iterdir():
        if p.suffix.lower() in (".csv", ".pdf"):
            shutil.copy2(p, a.out / p.name)
    print(f"  {a.raw_dir.name}: {n_rec} records x {n_ch} ch -> 12 kHz ({clipped} clipped samples)", flush=True)


if __name__ == "__main__":
    main()
