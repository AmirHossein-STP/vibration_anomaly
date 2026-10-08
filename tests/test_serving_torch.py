"""Torch-service tests (vibration, hydraulic) on synthetic data. Skipped when
PyTorch is not installed (the local project venv has none; the Docker images do)."""
import csv
import json

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from serving_core.cli import main  # noqa: E402
from serving_core.registry import Registry  # noqa: E402
from serving_core.services.hydraulic_spec import GROUPS, SENSORS  # noqa: E402

FS = 12000


def runner(service, tmp_path, capsys):
    data, models = tmp_path / "data", tmp_path / "models"

    def run(*args):
        rc = main([service, "--data-root", str(data), "--models-root", str(models), *args])
        o = capsys.readouterr()
        return rc, o.out, o.err

    run("init")
    return run, data / service, models / service


def cand(out):
    return out.split("Candidate model:")[1].split()[0]


def noise(seed, seconds):
    r = np.random.default_rng(seed)
    t = np.arange(seconds * FS) / FS
    return 0.5 * np.sin(2 * np.pi * 30 * t) + r.normal(0, 1, len(t))


def test_vibration_train_promote_infer(tmp_path, capsys):
    run, base, models = runner("vibration", tmp_path, capsys)
    (base / "config/service.yaml").write_text("fs_hz: 12000\n")
    for i in range(3):
        np.save(base / "train" / f"h{i}.npy", noise(i, 40))
    rc, out, err = run("train", "--seeds", "1", "--epochs", "2")
    assert rc == 0, err
    v = cand(out)
    assert Registry(models).active() is None and "single seed" in out
    assert run("promote", v, "--yes")[0] == 0
    healthy = noise(50, 10)
    bad = noise(51, 10)
    bad[::700] += 40.0                                      # repetitive strong impacts
    np.save(base / "in" / "healthy.npy", healthy)
    np.save(base / "in" / "impacts.npy", bad)
    rc, out, err = run("infer")
    assert rc == 0, err
    job = sorted((base / "out").glob("*/infer-*"))[-1]
    rows = {r["file"]: r for r in json.loads((job / "results.json").read_text())["rows"]}
    assert rows["impacts.npy"]["score"] > 1.5 * rows["healthy.npy"]["score"]
    assert rows["impacts.npy"]["verdict"] == "faulty" and rows["healthy.npy"]["verdict"] == "healthy"
    # recalibrate keeps the weights, creates a new candidate
    rc, out, err = run("calibrate")
    assert rc == 0, err
    assert (models / "versions" / cand(out) / "model.pt").is_file()


def test_vibration_refuses_too_little_data(tmp_path, capsys):
    run, base, _ = runner("vibration", tmp_path, capsys)
    (base / "config/service.yaml").write_text("fs_hz: 12000\n")
    np.save(base / "train" / "short.npy", noise(1, 8))
    rc, _, err = run("train", "--seeds", "1", "--epochs", "1")
    assert rc == 2 and "not enough healthy data" in err


def cycle(rng, degrade=0.0):
    out = {}
    for g, (n, sensors) in GROUPS.items():
        t = np.linspace(0, 1, n)
        for k, s in enumerate(sensors):
            out[s] = (np.sin(2 * np.pi * (k + 1) * t) + 0.05 * rng.normal(size=n) + 5 * k).astype(np.float32)
            if degrade and g == "medium":                    # a flow-rate shift = "valve" degradation
                out[s] = out[s] + degrade
    return out


def test_hydraulic_train_promote_infer(tmp_path, capsys):
    run, base, models = runner("hydraulic", tmp_path, capsys)
    rng = np.random.default_rng(0)
    rows = []
    for i in range(260):
        deg = i >= 220
        np.savez(base / "train" / f"c{i:04d}.npz", **cycle(rng, 1.5 if deg else 0.0))
        rows.append([f"c{i:04d}.npz", 100, 73 if deg else 100, 0, 130])
    with open(base / "train" / "status.csv", "w", newline="") as f:
        w = csv.writer(f); w.writerow(["file", "cooler", "valve", "pump", "accumulator"]); w.writerows(rows)
    rc, out, err = run("train", "--components", "valve", "--epochs", "3")
    assert rc == 0, err
    assert "labelled check" in out and "single seed" in out
    v = cand(out)
    assert run("promote", v, "--yes")[0] == 0
    np.savez(base / "in" / "ok.npz", **cycle(rng))
    np.savez(base / "in" / "shifted.npz", **cycle(rng, 1.5))
    bad = cycle(rng); bad.pop("PS3")
    np.savez(base / "in" / "missing_sensor.npz", **bad)
    rc, out, err = run("infer")
    assert rc == 1
    job = sorted((base / "out").glob("*/infer-*"))[-1]
    res = {r["file"]: r for r in json.loads((job / "results.json").read_text())["rows"]}
    assert res["shifted.npz"]["score"] > res["ok.npz"]["score"]
    assert res["shifted.npz"]["component"] == "valve" and res["shifted.npz"]["error_medium"] > res["shifted.npz"]["error_slow"]
    assert res["missing_sensor.npz"]["status"] == "error" and "PS3" in res["missing_sensor.npz"]["message"]


def test_hydraulic_needs_status_csv(tmp_path, capsys):
    run, base, _ = runner("hydraulic", tmp_path, capsys)
    rc, _, err = run("train", "--epochs", "1")
    assert rc == 2 and "status.csv" in err
