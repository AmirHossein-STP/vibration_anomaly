"""Tests for the production wrapper. No dataset needed: signals are synthetic.

Covers the shared pieces (registry/promote/rollback integrity, folder contract,
bad-input handling) through the real CLI entry point, using the numpy-only
`bearing` service. Torch services are in test_serving_torch.py.
"""
import json
from pathlib import Path

import numpy as np
import pytest

from serving_core.cli import main
from serving_core.registry import Registry, RegistryError

FS = 12000
RPM = 900.0
GEOM = "bearing: {name: t, n_balls: 8, ball_diameter: 6.75, pitch_diameter: 28.5}\nfs_hz: 12000\nrpm: 900\n"


def healthy(seed, seconds=12):
    r = np.random.default_rng(seed)
    t = np.arange(seconds * FS) / FS
    return 0.3 * np.sin(2 * np.pi * RPM / 60 * t) + r.normal(0, 1, len(t))


def faulty(seed, seconds=12, bpfo_order=3.0526):
    """Impacts at BPFO ringing a 3.2 kHz resonance on top of healthy noise."""
    x = healthy(seed, seconds)
    period = FS / (bpfo_order * RPM / 60)
    ring = np.exp(-np.arange(300) / 40.0) * np.sin(2 * np.pi * 3200 * np.arange(300) / FS)
    k = 0.0
    while int(k) + 300 < len(x):
        x[int(k):int(k) + 300] += 6.0 * ring
        k += period
    return x


@pytest.fixture
def env(tmp_path, capsys):
    data, models = tmp_path / "data", tmp_path / "models"

    def run(*args):
        rc = main(["bearing", "--data-root", str(data), "--models-root", str(models), *args])
        out = capsys.readouterr()
        return rc, out.out, out.err

    run("init")
    (data / "bearing/config/service.yaml").write_text(GEOM)
    return run, data / "bearing", models / "bearing"


def version_from(out):
    return out.split("Candidate model:")[1].split()[0]


def test_full_cycle_calibrate_promote_infer_rollback(env):
    run, base, models = env
    for i in range(10):
        np.save(base / "train" / f"h{i}.npy", healthy(i))
    rc, out, err = run("calibrate")
    assert rc == 0, err
    v1 = version_from(out)
    assert Registry(models).active() is None             # train/calibrate never activate
    np.save(base / "in" / "new_healthy.npy", healthy(100))
    rc, out, err = run("infer")
    assert rc == 2 and "No active model" in err            # inputs but no model
    assert run("promote", v1)[0] == 2                      # no tty and no --yes -> refuses
    assert run("promote", v1, "--yes")[0] == 0
    (base / "in" / "new_healthy.npy").rename(base / "new_healthy_tmp.npy")
    rc, out, err = run("infer")
    assert rc == 0 and "Nothing to do" in out              # empty in/ is not an error
    (base / "new_healthy_tmp.npy").rename(base / "in" / "new_healthy.npy")
    np.save(base / "in" / "new_faulty.npy", faulty(101))
    np.save(base / "in" / "broken.npy", np.array([1.0, np.nan] * 40000))
    rc, out, err = run("infer", "--archive")
    assert rc == 1                                         # finished, one input rejected
    job = sorted((base / "out").glob("*/infer-*"))[-1]
    res = json.loads((job / "results.json").read_text())
    by = {r["file"]: r for r in res["rows"]}
    assert by["broken.npy"]["status"] == "error" and "NaN" in by["broken.npy"]["message"]
    assert by["new_faulty.npy"]["flagged_fraction"] > by["new_healthy.npy"]["flagged_fraction"] + 0.3
    assert by["new_faulty.npy"]["verdict"] in ("watch", "faulty")
    assert (job / "results.csv").is_file() and (job / "manifest.json").is_file() and (job / "run.log").is_file()
    assert (base / "out" / "score_log.csv").is_file()
    assert (base / "in" / "broken.npy").exists()           # failed files stay
    assert not (base / "in" / "new_faulty.npy").exists()   # scored files are archived
    # second candidate, promote, then roll back
    rc, out, _ = run("calibrate"); v2 = version_from(out)
    assert run("promote", v2, "--yes")[0] == 0 and Registry(models).active() == v2
    assert run("rollback", "--yes")[0] == 0 and Registry(models).active() == v1


def test_promote_refuses_tampered_version(env):
    run, base, models = env
    for i in range(10):
        np.save(base / "train" / f"h{i}.npy", healthy(i))
    _, out, _ = run("calibrate"); v = version_from(out)
    cal = models / "versions" / v / "calibration.json"
    d = json.loads(cal.read_text()); d["threshold"] = 99.0; cal.write_text(json.dumps(d))
    rc, out, err = run("promote", v, "--yes")
    assert rc == 2 and "modified" in err
    assert Registry(models).active() is None


def test_calibrate_needs_consistent_geometry_and_enough_data(env):
    run, base, _ = env
    (base / "config" / "service.yaml").write_text("fs_hz: 12000\nrpm: 900\nbearing: {n_balls: 8}\n")
    np.save(base / "train" / "h.npy", healthy(1))
    rc, _, err = run("calibrate")                           # half a geometry is refused
    assert rc == 2 and "together" in err
    (base / "config" / "service.yaml").write_text(GEOM)
    rc, _, err = run("calibrate")                           # one 12 s file = 23 windows < 200
    assert rc == 2 and "healthy windows" in err


def test_unknown_setting_is_rejected(env):
    run, base, _ = env
    (base / "config" / "service.yaml").write_text(GEOM + "typo_setting: 1\n")
    rc, _, err = run("doctor")
    assert rc == 1 or rc == 2
    rc, _, err = run("calibrate")
    assert rc == 2 and "unknown setting" in err


def test_doctor_reports_missing_model_as_warning(env):
    run, *_ = env
    rc, out, _ = run("doctor")
    assert rc == 0 and "[warn] active model" in out


def test_registry_rollback_history():
    import tempfile
    with tempfile.TemporaryDirectory() as d:
        reg = Registry(Path(d))
        assert reg.rollback_target() is None
        with pytest.raises(RegistryError):
            reg.rollback()


def test_current_user_survives_unknown_uid(monkeypatch):
    """Containers run with the host uid, which has no passwd entry."""
    import getpass
    from serving_core.util import current_user

    def boom():
        raise KeyError("getpwuid(): uid not found: 501")

    monkeypatch.setattr(getpass, "getuser", boom)
    assert current_user()
