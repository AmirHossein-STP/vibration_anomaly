"""The `bearing` service on method v1.0 (notes/22, WP17): many units, each with
its own baseline and its own bearing, through the real CLI. Synthetic signals."""
import csv
import json

import numpy as np
import pytest

from serving_core.cli import main
from serving_core.registry import Registry
from test_serving_core import FS, faulty, healthy

CONFIG = """fs_hz: 12000
rpm: 900
iso20816: {group: 2, foundation: rigid, signal_unit: g}
bearing: {designation: "6203 (Paderborn KAt)"}
units:
  pump_a: {bearing: {designation: "6203 (Paderborn KAt)"}}
  fan_b: {bearing: {designation: null}}
"""


@pytest.fixture
def fleet(tmp_path, capsys):
    data, models = tmp_path / "data", tmp_path / "models"

    def run(*args):
        rc = main(["bearing", "--data-root", str(data), "--models-root", str(models), *args])
        out = capsys.readouterr()
        return rc, out.out, out.err

    run("init")
    base = data / "bearing"
    (base / "config" / "service.yaml").write_text(CONFIG)
    for unit, seed in (("pump_a", 0), ("fan_b", 50)):
        (base / "train" / unit).mkdir(parents=True)
        for i in range(10):
            np.save(base / "train" / unit / f"h{i}.npy", healthy(seed + i))
    rc, out, err = run("calibrate")
    assert rc == 0, err
    v = out.split("Candidate model:")[1].split()[0]
    assert run("promote", v, "--yes")[0] == 0
    return run, base, models / "bearing", v


def _rows(base):
    job = sorted((base / "out").glob("*/infer-*"))[-1]
    return {(r["unit"], r["file"]): r for r in json.loads((job / "results.json").read_text())["rows"]}


def test_each_unit_gets_its_own_baseline_and_detector(fleet):
    run, base, models, v = fleet
    cal = json.loads((models / "versions" / v / "calibration.json").read_text())
    assert cal["method_version"] == "1.0"
    assert cal["units"]["pump_a"]["detector"] == "six_line" and cal["units"]["fan_b"]["detector"] == "geometry_free"
    for unit in ("pump_a", "fan_b"):
        (base / "in" / unit).mkdir(parents=True, exist_ok=True)
        np.save(base / "in" / unit / "new_ok.npy", healthy(900))
        np.save(base / "in" / unit / "new_bad.npy", faulty(901))
    rc, out, err = run("infer")
    assert rc == 0, err
    rows = _rows(base)
    for unit in ("pump_a", "fan_b"):
        ok, bad = rows[(unit, "new_ok.npy")], rows[(unit, "new_bad.npy")]
        assert bad["flagged_fraction"] > ok["flagged_fraction"] + 0.3
        assert bad["verdict"] in ("watch", "faulty")
    assert rows[("pump_a", "new_bad.npy")]["fault_type"] == "outer race"        # injected at BPFO
    assert "geometry unknown" in rows[("fan_b", "new_bad.npy")]["dominant"]       # no typing without geometry
    assert rows[("pump_a", "new_ok.npy")]["iso_zone"] in ("A", "B", "C", "D")     # L0 runs when configured


def test_unknown_unit_is_rejected_not_scored(fleet):
    run, base, *_ = fleet
    (base / "in" / "compressor_c").mkdir(parents=True)
    np.save(base / "in" / "compressor_c" / "x.npy", healthy(7))
    rc, _, _ = run("infer")
    assert rc == 1
    job = sorted((base / "out").glob("*/infer-*"))[-1]
    row = json.loads((job / "results.json").read_text())["rows"][0]
    assert row["status"] == "error" and "no baseline" in row["message"]


def test_trend_cusum_accumulates_per_unit_across_jobs(fleet):
    run, base, *_ = fleet
    (base / "in" / "pump_a").mkdir(parents=True, exist_ok=True)
    for k in range(4):
        np.save(base / "in" / "pump_a" / f"rec{k}.npy", faulty(950 + k))
        assert run("infer", "--archive")[0] == 0
    with (base / "out" / "bearing_trend.csv").open() as f:
        trend = [r for r in csv.DictReader(f) if r["unit"] == "pump_a"]
    c = [float(r["cusum_physics"]) for r in trend]
    assert len(c) == 4 and c == sorted(c) and c[-1] > c[0]
    assert trend[-1]["trend_alarm"] == "True"


def test_model_from_another_method_version_is_refused(fleet):
    run, base, models, v = fleet
    p = models / "versions" / v / "calibration.json"
    d = json.loads(p.read_text()); d["method_version"] = "0.9"; p.write_text(json.dumps(d))
    np.save(base / "in" / "x.npy", healthy(3))
    rc, _, err = run("infer")
    assert rc == 2   # integrity check refuses the modified version before it can be used


def test_missing_speed_is_refused(tmp_path, capsys):
    data, models = tmp_path / "d", tmp_path / "m"
    run = lambda *a: main(["bearing", "--data-root", str(data), "--models-root", str(models), *a])
    run("init")
    (data / "bearing/config/service.yaml").write_text("fs_hz: 12000\n")
    for i in range(10):
        np.save(data / "bearing/train" / f"h{i}.npy", healthy(i))
    assert run("calibrate") == 2
    assert "speed unknown" in capsys.readouterr().err
