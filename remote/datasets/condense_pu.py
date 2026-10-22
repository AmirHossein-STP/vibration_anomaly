"""Condense raw Paderborn archives (one .rar per bearing code) into two
stores, from a single pass over the raw files. Called by fetch.py on the
server; the raw archives (5.2 GB) stay on the ephemeral SSD.

    v1/  the store every analysis so far reads (data/pu/prepared_pu):
         vibration_1 resampled to 12 kHz, float32, one .npz per code,
         written by the SAME prepare_code() the Kaggle kernel used, so it
         reproduces the laptop's store.
    v2/  native rate, more channels: vibration_1 @ 64 kHz plus speed,
         torque and force (~4 kHz), float32, keys "<condition>_<run>__<channel>".
         WP2's measured-speed test (H-c) needs the speed channel.

Usage (fetch.py does this):
    python remote/datasets/condense_pu.py --raw-dir <dir of K*.rar> --out <dir> \
        --work <scratch dir> --src <repo>/src [--jobs N]
    python remote/datasets/condense_pu.py --merge-v1 <other v1 store> --out <dir>
        copy codes the raw set lacks (Zenodo has no KA08) from another v1
        store, with their manifest rows
"""
from __future__ import annotations

import argparse
import shutil
import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

V2_CHANNELS = ["vibration_1", "speed", "torque", "force"]
V1_COLUMNS = ["code", "condition", "run", "rpm", "fs_hz", "n_samples",
              "fault_type", "damage_mode", "universal_label"]


def _imports(src: str):
    if src not in sys.path:
        sys.path.insert(0, src)
    import numpy as np
    from domain_adaptation.data.prepare_pu import prepare_code
    from domain_adaptation.data.pu_bearing_codes import PU_BEARING_CODES
    from domain_adaptation.data.pu_loader import (
        FILENAME_RE, UnreadableMatFile, load_channels, rpm_for_condition)
    return np, prepare_code, PU_BEARING_CODES, FILENAME_RE, UnreadableMatFile, load_channels, rpm_for_condition


def condense_v2(code_dir: Path, out_dir: Path, src: str) -> dict:
    np, _, codes, FILENAME_RE, UnreadableMatFile, load_channels, rpm_for_condition = _imports(src)
    code = code_dir.name
    arrays, rows, unreadable = {}, [], []
    for mat_path in sorted(code_dir.glob("*.mat")):
        m = FILENAME_RE.match(mat_path.name)
        if not m or m.group("code") != code:
            continue
        condition, run = m.group("condition"), int(m.group("run"))
        try:
            chans = load_channels(mat_path, V2_CHANNELS)
        except (UnreadableMatFile, KeyError) as exc:
            unreadable.append(mat_path.name)
            print(f"    v2 SKIP {exc}", flush=True)
            continue
        row = {"code": code, "condition": condition, "run": run, "rpm": rpm_for_condition(condition),
               "fs_vibration_hz": 64000, "fs_lowrate_hz": 4000,
               "fault_type": codes[code]["fault_type"], "damage_mode": codes[code]["damage_mode"],
               "universal_label": codes[code]["universal_label"],
               "speed_mean": float(np.mean(chans["speed"]))}
        for ch, data in chans.items():
            arrays[f"{condition}_{run}__{ch}"] = data.astype(np.float32)
            row[f"n_{ch}"] = len(data)
        rows.append(row)
    out_dir.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(out_dir / f"{code}.npz", **arrays)
    return {"rows": rows, "unreadable": unreadable}


def find_code_dir(extracted: Path, code: str) -> Path:
    """The .rar holds <code>/<files>.mat; tolerate a flat archive too."""
    for d in [extracted / code, *extracted.rglob(code)]:
        if d.is_dir() and any(d.glob("*.mat")):
            return d
    if any(extracted.glob("*.mat")):
        target = extracted.parent / f"{code}.flat" / code
        target.parent.mkdir(parents=True, exist_ok=True)
        extracted.rename(target)
        return target
    raise FileNotFoundError(f"no .mat files for {code} under {extracted}")


def process_code(archive: str, out: str, work: str, src: str) -> dict:
    from safe_extract import extract_rar
    archive_p, out_p, work_p = Path(archive), Path(out), Path(work)
    code = archive_p.stem
    _, prepare_code, *_ = _imports(src)
    extracted = work_p / f"{code}.extract"
    if extracted.exists():
        shutil.rmtree(extracted)
    extract_rar(archive_p, extracted)
    code_dir = find_code_dir(extracted, code)
    v1 = prepare_code(code_dir, out_p / "v1")
    v2 = condense_v2(code_dir, out_p / "v2", src)
    shutil.rmtree(extracted, ignore_errors=True)
    shutil.rmtree(work_p / f"{code}.flat", ignore_errors=True)
    print(f"  {code}: v1 {len(v1['rows'])} runs, v2 {len(v2['rows'])} runs", flush=True)
    return {"code": code, "v1": v1, "v2": v2}


def merge_v1(other: Path, out: Path) -> None:
    import pandas as pd
    mine_path, theirs = out / "v1" / "manifest.csv", pd.read_csv(other / "manifest.csv")
    mine = pd.read_csv(mine_path)
    extra = sorted(set(theirs.code) - set(mine.code))
    for code in extra:
        shutil.copy2(other / f"{code}.npz", out / "v1" / f"{code}.npz")
    merged = pd.concat([mine, theirs[theirs.code.isin(extra)]], ignore_index=True)
    merged = merged.sort_values("code", kind="stable").reset_index(drop=True)
    merged.to_csv(mine_path, index=False)
    print(f"merged {extra} from {other} into {mine_path} ({len(merged)} runs)")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--raw-dir", type=Path)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--work", type=Path)
    ap.add_argument("--src", default=str(Path(__file__).resolve().parents[2] / "src"))
    ap.add_argument("--jobs", type=int, default=4)
    ap.add_argument("--merge-v1", type=Path, help="copy missing codes from this v1 store into OUT/v1")
    args = ap.parse_args()

    if args.merge_v1:
        merge_v1(args.merge_v1, args.out)
        return

    import pandas as pd
    archives = sorted(args.raw_dir.glob("K*.rar"))
    if not archives:
        raise SystemExit(f"no K*.rar under {args.raw_dir}")
    args.work.mkdir(parents=True, exist_ok=True)
    print(f"condensing {len(archives)} codes with {args.jobs} workers", flush=True)
    with ProcessPoolExecutor(max_workers=args.jobs) as pool:
        results = list(pool.map(process_code, [str(a) for a in archives], [str(args.out)] * len(archives),
                                [str(args.work)] * len(archives), [args.src] * len(archives)))

    for store in ("v1", "v2"):
        # codes sorted, files in glob order within a code: the row order
        # prepare_pu.main() writes, so v1's manifest.csv matches the old store
        rows = [r for res in results for r in res[store]["rows"]]
        df = pd.DataFrame(rows)
        if store == "v1":
            df = df[V1_COLUMNS]
        df.to_csv(args.out / store / "manifest.csv", index=False)
        bad = [f for res in results for f in res[store]["unreadable"]]
        print(f"{store}: {len(df)} runs, {df.code.nunique()} codes"
              + (f", {len(bad)} unreadable: {bad[:5]}" if bad else ""))


if __name__ == "__main__":
    main()
