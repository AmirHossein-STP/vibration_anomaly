"""Replay (WP17 gate): real Ottawa 2023 recordings fed through the `bearing`
service reproduce the analysis numbers of notes/17 (own-unit baseline,
cross-fitted, `d_xf`) exactly -- the product IS the method that was measured.

Needs the staged dataset and the committed WP5 results, so it runs on the box
(`remote/box run tests -- python3 -m pytest -q`) and skips elsewhere.
"""
import json
from pathlib import Path

import numpy as np
import pytest

from serving_core.cli import main

DATA = Path("data/ottawa_2023")
WP5 = Path("outputs/wp5/wp5_results.json")
pytestmark = pytest.mark.skipif(not (DATA.is_dir() and WP5.is_file()), reason="needs staged Ottawa 2023 + WP5 results")
BEARINGS = [1, 6, 10]          # inner, outer, outer


def _save(path: Path, x: np.ndarray, rpm: float):
    np.save(path, x)
    path.with_name(path.name + ".meta.json").write_text(json.dumps({"rpm": rpm}))


def test_service_reproduces_wp5_own_unit_numbers(tmp_path, capsys):
    import sys
    sys.path.insert(0, "scripts/analysis")
    from domain_adaptation.data.public_datasets import ottawa2023
    from wp5_cache import to_12k

    data, models = tmp_path / "data", tmp_path / "models"
    run = lambda *a: main(["bearing", "--data-root", str(data), "--models-root", str(models), *a])
    run("init")
    base = data / "bearing"
    (base / "config" / "service.yaml").write_text(
        'fs_hz: 12000\nbearing: {designation: "NSK 6203ZZ"}\ncalibration: {min_windows: 9}\n')
    held = {}
    for b in BEARINGS:
        h = next(ottawa2023(DATA, only=f"H_{b}_0"))
        sig, fs = to_12k(h.signal, h.fs)
        assert fs == 12000
        (base / "train" / f"b{b:02d}").mkdir(parents=True)
        _save(base / "train" / f"b{b:02d}" / "healthy_first5s.npy", sig[:60000], h.rpm)   # windows t0 0..4 s
        (base / "in" / f"b{b:02d}").mkdir(parents=True)
        _save(base / "in" / f"b{b:02d}" / "healthy_last5s.npy", sig[60000:], h.rpm)       # windows t0 5..9 s
        letter = "I" if b <= 5 else "O"
        for st in (1, 2):
            r = next(ottawa2023(DATA, only=f"{letter}_{b}_{st}"))
            _save(base / "in" / f"b{b:02d}" / f"{letter}_{b}_{st}.npy", to_12k(r.signal, r.fs)[0], r.rpm)
        held[b] = letter
    assert run("calibrate") == 0
    out = capsys.readouterr().out
    v = out.split("Candidate model:")[1].split()[0]
    assert run("promote", v, "--yes") == 0
    assert run("infer") == 0
    job = sorted((base / "out").glob("*/infer-*"))[-1]
    rows = json.loads((job / "results.json").read_text())["rows"]
    ref = {f["fold"]: f for f in json.loads(WP5.read_text())["results"]["ottawa_2023:own"]["d_xf"]["folds"]}
    for b in BEARINGS:
        unit = f"b{b:02d}"
        mine = [r for r in rows if r["unit"] == unit]
        h = next(r for r in mine if r["file"] == "healthy_last5s.npy")
        faults = [r for r in mine if r["file"] != "healthy_last5s.npy"]
        recall = sum(r["n_windows_flagged"] for r in faults) / sum(r["n_windows"] for r in faults)
        want = ref[f"ott23_b{b:02d}"]
        assert h["flagged_fraction"] == pytest.approx(want["fa"], abs=1e-4)
        assert recall == pytest.approx(want["recall_race"], abs=1e-9)
        # the analysis caches store features to 7 significant digits (float_format "%.7g");
        # the service works at full precision, so thresholds agree to ~1e-6, verdicts exactly
        assert h["threshold"] == pytest.approx(want["thr"], rel=1e-5)
