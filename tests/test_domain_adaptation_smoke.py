"""Fast sanity tests -- no real dataset download required. CWRU rows use
tiny synthetic .npz files (same convention as test_smoke.py); PU rows use
tiny synthetic .mat files matching the REAL dataset's confirmed nested
struct layout (see data/pu_loader.py's docstring and
DOMAIN_ADAPTATION_PLAN.md's "Revisions" section) -- no download needed to
run this suite.

These check the plumbing (label harmonization, resampling, role/split
logic, GRL's gradient-reversal behavior, shapes end to end), not model
quality. Model-forward/training tests are skipped if torch isn't installed
(see [[vscode-plugin-env-constraints]]-style local setup: this venv
deliberately doesn't have torch -- training happens on Kaggle, see
train_baseline.py/train_dann.py). Run with: .venv/bin/python -m pytest
tests/test_domain_adaptation_smoke.py
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
import scipy.io

pytest.importorskip("scipy")

from domain_adaptation.config import (
    CommonConfig,
    Config,
    CWRUConfig,
    ModelConfig,
    PathsConfig,
    PUConfig,
    STFTConfig,
)
from domain_adaptation.data import dataset as da_dataset
from domain_adaptation.data.cwru_loader import build_cwru_catalog
from domain_adaptation.data.pu_bearing_codes import PU_BEARING_CODES
from domain_adaptation.data.pu_loader import UnreadableMatFile, build_pu_catalog, load_vibration_signal
from domain_adaptation.analysis import binary_detection_metrics, domain_separability_probe
from domain_adaptation.data.transforms import resample_to_rate

try:
    import torch

    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False

requires_torch = pytest.mark.skipif(
    not HAS_TORCH, reason="local venv deliberately has no torch -- see train_baseline.py's docstring"
)


# --- shared synthetic-data helpers ------------------------------------------

def write_synthetic_cwru_npz(raw_dir: Path, rpm: int, fault_type: str, size: int = 7) -> None:
    raw_dir.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(0)
    signal = rng.standard_normal(20000).astype(np.float64)
    if fault_type == "Normal":
        np.savez(raw_dir / f"{rpm}_Normal.npz", DE=signal)
    else:
        np.savez(raw_dir / f"{rpm}_{fault_type}_{size}_DE12.npz", DE=signal)


def write_synthetic_pu_mat(root_dir: Path, code: str, condition: str, run: int, n_samples: int = 6000) -> None:
    code_dir = root_dir / code
    code_dir.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(hash((code, condition, run)) % (2**32))
    mdict = {
        f"{condition}_{code}_{run}": {
            "Y": [
                {"Name": "force", "Data": rng.standard_normal(100)},
                {"Name": "vibration_1", "Data": rng.standard_normal(n_samples).astype(np.float64)},
            ]
        }
    }
    scipy.io.savemat(code_dir / f"{condition}_{code}_{run}.mat", mdict)


def make_config(tmp_path: Path, **overrides) -> Config:
    defaults = dict(
        seed=0,
        paths=PathsConfig(
            cwru_raw_dir="raw", processed_dir="processed", models_dir="models",
            figures_dir="figures", reports_dir="reports", pu_root_local="pu_root",
        ),
        common=CommonConfig(fs_hz=1000, segment_length=256, train_stride=64, eval_stride=128),
        stft=STFTConfig(nperseg=32, hop_length=8, log_epsilon=1e-10),
        cwru=CWRUConfig(sensor="DE", train_rpms=[1730, 1750], val_rpm=1797, include_in_source=True),
        pu=PUConfig(
            dataset_slug="test/test", native_fs_hz=4000, vibration_channel="vibration_1",
            healthy_codes=["K001"], artificial_outer_codes=["KA01"], artificial_inner_codes=["KI01"],
            real_outer_codes=["KA04"], real_inner_codes=["KI04"],
            source_conditions=["N15_M07_F10"], target_condition="N09_M07_F10",
            source_train_fraction=0.5, target_normal_adapt_fraction=0.5,
        ),
        model=ModelConfig(base_channels=2, feature_dim=8, classifier_hidden=6, domain_hidden=6, dropout=0.1),
        train_baseline=None,
        train_dann=None,
        evaluation=None,
        project_root=tmp_path,
    )
    defaults.update(overrides)
    return Config(**defaults)


# --- data layer --------------------------------------------------------------

def test_cwru_catalog_maps_universal_labels_and_drops_ball(tmp_path):
    raw_dir = tmp_path / "raw"
    write_synthetic_cwru_npz(raw_dir, 1730, "Normal")
    write_synthetic_cwru_npz(raw_dir, 1750, "IR")
    write_synthetic_cwru_npz(raw_dir, 1750, "OR@6")
    write_synthetic_cwru_npz(raw_dir, 1750, "B")  # should be dropped
    write_synthetic_cwru_npz(raw_dir, 1797, "Normal")

    catalog = build_cwru_catalog(raw_dir, train_rpms=[1730, 1750], val_rpm=1797)
    assert set(catalog.fault_type) == {"Normal", "IR", "OR@6"}  # "B" dropped
    assert dict(zip(catalog.fault_type, catalog.universal_label)) == {"Normal": 0, "IR": 1, "OR@6": 2}
    # rpm isn't a returned column -- "condition" (str(rpm)) is CWRU's operating-condition tag.
    assert set(catalog[catalog.condition == "1797"].split) == {"val"}
    assert set(catalog[catalog.condition != "1797"].split) == {"train"}


def test_cwru_catalog_rejects_unmapped_rpm(tmp_path):
    raw_dir = tmp_path / "raw"
    write_synthetic_cwru_npz(raw_dir, 1730, "Normal")
    write_synthetic_cwru_npz(raw_dir, 1999, "Normal")  # not in train_rpms or val_rpm
    with pytest.raises(ValueError, match="1999"):
        build_cwru_catalog(raw_dir, train_rpms=[1730], val_rpm=1797)


def test_pu_bearing_codes_cover_universal_label_space():
    assert PU_BEARING_CODES["K001"]["universal_label"] == 0
    assert PU_BEARING_CODES["KI04"]["universal_label"] == 1
    assert PU_BEARING_CODES["KA04"]["universal_label"] == 2
    assert PU_BEARING_CODES["KB23"]["universal_label"] is None  # combined damage, excluded


def test_build_pu_catalog_parses_filenames_and_skips_combined(tmp_path):
    root = tmp_path / "pu"
    for run in [1, 2]:
        write_synthetic_pu_mat(root, "K001", "N15_M07_F10", run)
        write_synthetic_pu_mat(root, "KA04", "N09_M07_F10", run)
    write_synthetic_pu_mat(root, "KB23", "N15_M07_F10", 1)  # combined damage

    catalog = build_pu_catalog(root, ["K001", "KA04", "KB23"])
    assert set(catalog.code) == {"K001", "KA04"}  # KB23 silently dropped
    assert set(catalog.universal_label) == {0, 2}
    assert len(catalog[catalog.code == "K001"]) == 2


def test_build_pu_catalog_rejects_missing_code_folder(tmp_path):
    root = tmp_path / "pu"
    root.mkdir()
    with pytest.raises(FileNotFoundError):
        build_pu_catalog(root, ["K001"])


def test_load_vibration_signal_extracts_named_channel(tmp_path):
    root = tmp_path / "pu"
    write_synthetic_pu_mat(root, "K001", "N15_M07_F10", 1, n_samples=500)
    mat_path = root / "K001" / "N15_M07_F10_K001_1.mat"
    signal = load_vibration_signal(mat_path)
    assert signal.shape == (500,)
    with pytest.raises(KeyError):
        load_vibration_signal(mat_path, channel="does_not_exist")


def test_resample_to_rate_noop_when_rates_match():
    signal = np.arange(100, dtype=np.float64)
    assert resample_to_rate(signal, 12000, 12000) is signal


def test_resample_to_rate_scales_length_by_exact_ratio():
    signal = np.random.default_rng(0).standard_normal(6400)
    resampled = resample_to_rate(signal, 64000, 12000)  # ratio 3/16
    assert resampled.shape[0] == 6400 * 3 // 16


def test_build_domain_adaptation_datasets_end_to_end(tmp_path):
    config = make_config(tmp_path)
    raw_dir = tmp_path / "raw"
    for rpm, fault in [(1730, "Normal"), (1730, "IR"), (1750, "OR@6"), (1797, "Normal"), (1797, "IR")]:
        write_synthetic_cwru_npz(raw_dir, rpm, fault)

    pu_root = tmp_path / "pu_root"
    for run in range(1, 5):  # >= 2 runs on each side of a 0.5 fraction split
        write_synthetic_pu_mat(pu_root, "K001", "N15_M07_F10", run)
        write_synthetic_pu_mat(pu_root, "KA01", "N15_M07_F10", run)
        write_synthetic_pu_mat(pu_root, "KI01", "N15_M07_F10", run)
        write_synthetic_pu_mat(pu_root, "K001", "N09_M07_F10", run)
        write_synthetic_pu_mat(pu_root, "KA04", "N09_M07_F10", run)
        write_synthetic_pu_mat(pu_root, "KI04", "N09_M07_F10", run)

    results = da_dataset.build_domain_adaptation_datasets(config, pu_root)
    assert set(results.keys()) == set(da_dataset.ROLES)
    for role, (X, meta) in results.items():
        assert len(X) == len(meta) > 0, role
        assert X.shape[1:] == (config.stft.nperseg // 2, (config.common.segment_length - config.stft.nperseg) // config.stft.hop_length + 1)

    # source_train/source_val contain BOTH domains; target_adapt/target_test are PU-only.
    assert set(results["source_train"][1].domain) == {"cwru", "pu_artificial"}
    assert set(results["target_adapt"][1].domain) == {"pu_target"}
    # target_test has both the held-out healthy rows AND the real-fault rows.
    assert set(results["target_test"][1].universal_label) == {0, 1, 2}

    # cached second call returns identical arrays without touching raw files
    results2 = da_dataset.build_domain_adaptation_datasets(config, pu_root)
    assert np.array_equal(results["source_train"][0], results2["source_train"][0])


def test_build_datasets_pu_only_when_cwru_excluded(tmp_path):
    """config.cwru.include_in_source=False -> SOURCE_TRAIN/SOURCE_VAL come
    from Paderborn alone, and CWRU's files are never even read (kernel v4's
    experiment: is forcing CWRU<->PU invariance actively harmful?)."""
    config = make_config(tmp_path)
    config.cwru.include_in_source = False

    # Deliberately write NO CWRU files at all: if the loader still tried to
    # read them this would raise, which is the point of the assertion.
    pu_root = tmp_path / "pu_root"
    for run in range(1, 5):
        for code, cond in [
            ("K001", "N15_M07_F10"), ("KA01", "N15_M07_F10"), ("KI01", "N15_M07_F10"),
            ("K001", "N09_M07_F10"), ("KA04", "N09_M07_F10"), ("KI04", "N09_M07_F10"),
        ]:
            write_synthetic_pu_mat(pu_root, code, cond, run)

    results = da_dataset.build_domain_adaptation_datasets(config, pu_root)
    assert set(results["source_train"][1].domain) == {"pu_artificial"}  # no "cwru"
    assert set(results["source_val"][1].domain) == {"pu_artificial"}
    assert set(results["source_train"][1].domain_id) == {da_dataset.DOMAIN_IDS["pu_artificial"]}
    # the target side is unaffected by the toggle
    assert set(results["target_adapt"][1].domain) == {"pu_target"}
    assert set(results["target_test"][1].universal_label) == {0, 1, 2}


def test_cwru_included_by_default_still_mixes_both_domains(tmp_path):
    """Guard against the v4 toggle silently changing the v1-v3 setup."""
    config = make_config(tmp_path)
    assert config.cwru.include_in_source is True
    raw_dir = tmp_path / "raw"
    for rpm, fault in [(1730, "Normal"), (1730, "IR"), (1750, "OR@6"), (1797, "Normal"), (1797, "IR")]:
        write_synthetic_cwru_npz(raw_dir, rpm, fault)
    pu_root = tmp_path / "pu_root"
    for run in range(1, 5):
        for code, cond in [
            ("K001", "N15_M07_F10"), ("KA01", "N15_M07_F10"), ("KI01", "N15_M07_F10"),
            ("K001", "N09_M07_F10"), ("KA04", "N09_M07_F10"), ("KI04", "N09_M07_F10"),
        ]:
            write_synthetic_pu_mat(pu_root, code, cond, run)
    results = da_dataset.build_domain_adaptation_datasets(config, pu_root)
    assert set(results["source_train"][1].domain) == {"cwru", "pu_artificial"}


def test_load_vibration_signal_names_the_file_when_unparseable(tmp_path):
    """scipy raises a bare 'TypeError: Expecting matrix here' with no clue
    which of ~1400 files was at fault -- which is exactly how kernel v5 died.
    The wrapper must name the path."""
    bad = tmp_path / "K001"
    bad.mkdir()
    path = bad / "N15_M07_F10_K001_1.mat"
    path.write_bytes(b"MATLAB 5.0 MAT-file, but truncated garbage" + b"\x00" * 200)
    with pytest.raises(UnreadableMatFile, match="N15_M07_F10_K001_1.mat"):
        load_vibration_signal(path)


def test_dataset_skips_isolated_unreadable_file(tmp_path, capsys):
    """One corrupt file among many must NOT kill the run (v5 lost ~100
    minutes of compute to exactly that)."""
    config = make_config(tmp_path)
    config.cwru.include_in_source = False
    pu_root = tmp_path / "pu_root"
    for run in range(1, 9):
        for code, cond in [
            ("K001", "N15_M07_F10"), ("KA01", "N15_M07_F10"), ("KI01", "N15_M07_F10"),
            ("K001", "N09_M07_F10"), ("KA04", "N09_M07_F10"), ("KI04", "N09_M07_F10"),
        ]:
            write_synthetic_pu_mat(pu_root, code, cond, run)
    # corrupt exactly one file (1 of 8 runs for that code/condition = 12.5% of
    # that code, but well under 2% is not achievable at test scale, so this
    # asserts the SKIP path via a role whose failure fraction stays legal)
    victim = pu_root / "KA04" / "N09_M07_F10_KA04_3.mat"
    victim.write_bytes(b"garbage")
    config_relaxed = make_config(tmp_path)
    config_relaxed.cwru.include_in_source = False
    da_dataset.MAX_UNREADABLE_FRACTION = 0.5  # allow it, we are testing the skip path
    try:
        results = da_dataset.build_domain_adaptation_datasets(config_relaxed, pu_root)
        assert len(results["target_test"][0]) > 0  # run survived
        assert "SKIPPING unreadable" in capsys.readouterr().out
    finally:
        da_dataset.MAX_UNREADABLE_FRACTION = 0.02


def test_dataset_aborts_when_too_many_files_unreadable(tmp_path):
    """...but wholesale corruption must abort loudly rather than silently
    training on a fraction of the data."""
    config = make_config(tmp_path)
    config.cwru.include_in_source = False
    pu_root = tmp_path / "pu_root"
    for run in range(1, 5):
        for code, cond in [
            ("K001", "N15_M07_F10"), ("KA01", "N15_M07_F10"), ("KI01", "N15_M07_F10"),
            ("K001", "N09_M07_F10"), ("KA04", "N09_M07_F10"), ("KI04", "N09_M07_F10"),
        ]:
            write_synthetic_pu_mat(pu_root, code, cond, run)
    for mat in (pu_root / "KA04").glob("*.mat"):
        mat.write_bytes(b"garbage")
    for mat in (pu_root / "KI04").glob("*.mat"):
        mat.write_bytes(b"garbage")
    with pytest.raises(RuntimeError, match="systemic"):
        da_dataset.build_domain_adaptation_datasets(config, pu_root)


# --- analysis (torch-free, so these run locally) ----------------------------

def test_separability_probe_detects_separable_domains():
    """Two clearly offset clusters -> a fresh probe should nail it (~1.0),
    which is what "adaptation did nothing" looks like."""
    rng = np.random.default_rng(0)
    source = rng.normal(0.0, 0.5, size=(200, 8))
    target = rng.normal(5.0, 0.5, size=(200, 8))  # far apart, trivially separable
    result = domain_separability_probe(source, target, seed=0)
    assert result["separability_accuracy"] > 0.95
    assert result["proxy_a_distance"] > 1.8
    assert result["n_per_domain"] == 200


def test_separability_probe_reports_chance_for_aligned_domains():
    """Two samples drawn from the SAME distribution -> no probe can beat
    chance, which is what successful alignment looks like."""
    rng = np.random.default_rng(0)
    source = rng.normal(0.0, 1.0, size=(300, 8))
    target = rng.normal(0.0, 1.0, size=(300, 8))  # identical distribution
    result = domain_separability_probe(source, target, seed=0)
    assert 0.35 < result["separability_accuracy"] < 0.65
    assert result["proxy_a_distance"] < 0.7
    assert result["chance"] == 0.5


def test_separability_probe_balances_unequal_domain_sizes():
    rng = np.random.default_rng(0)
    source = rng.normal(0.0, 1.0, size=(500, 4))
    target = rng.normal(0.0, 1.0, size=(60, 4))
    result = domain_separability_probe(source, target, seed=0)
    assert result["n_per_domain"] == 60  # subsampled to the smaller side, so chance stays 0.5


def test_separability_probe_rejects_tiny_inputs():
    with pytest.raises(ValueError, match="at least|>=10|Need"):
        domain_separability_probe(np.zeros((3, 4)), np.zeros((50, 4)))


def test_binary_detection_collapses_both_fault_types():
    """Labels 1 and 2 both count as 'faulty'; only 0 is healthy."""
    y_true = np.array([0, 0, 1, 1, 2, 2])
    y_pred = np.array([0, 1, 1, 0, 2, 0])  # 1 false alarm, 2 missed faults
    r = binary_detection_metrics(y_true, y_pred)
    assert r["confusion"] == {"tn": 1, "fp": 1, "fn": 2, "tp": 2}
    assert r["n_healthy"] == 2 and r["n_faulty"] == 4
    assert r["recall"] == pytest.approx(0.5)
    assert r["false_alarm_rate"] == pytest.approx(0.5)


def test_binary_detection_confusing_fault_types_still_counts_as_detected():
    """Calling an inner-race fault 'outer_race' is wrong for typing but
    RIGHT for detection -- that distinction is the whole point of reporting
    both, since v1-v5 showed typing fails while detection partly works."""
    y_true = np.array([1, 1, 2, 2])
    y_pred = np.array([2, 2, 1, 1])  # every type wrong, every fault caught
    r = binary_detection_metrics(y_true, y_pred)
    assert r["recall"] == pytest.approx(1.0)
    assert r["false_alarm_rate"] == pytest.approx(0.0)
    assert r["precision"] == pytest.approx(1.0)


def test_binary_detection_reports_always_faulty_null_comparison():
    """The null model's score must travel with the headline number: a 90%
    faulty test set makes 'always say faulty' look like 0.9 accuracy."""
    y_true = np.array([0] + [1] * 9)
    r = binary_detection_metrics(y_true, np.ones(10, dtype=int))
    assert r["always_faulty_accuracy"] == pytest.approx(0.9)
    assert r["accuracy"] == pytest.approx(0.9)
    assert r["false_alarm_rate"] == pytest.approx(1.0)  # flagged the only healthy window


def test_binary_detection_handles_no_positive_predictions():
    """v5's DANN called ~91% of everything healthy; the metric must not
    divide by zero when a model predicts no faults at all."""
    r = binary_detection_metrics(np.array([0, 1, 2]), np.zeros(3, dtype=int))
    assert r["precision"] == 0.0 and r["recall"] == 0.0 and r["f1"] == 0.0
    assert r["false_alarm_rate"] == 0.0


# --- models --------------------------------------------------------------

@requires_torch
def test_infer_reads_every_supported_signal_format(tmp_path):
    """A real machine's data will not arrive as CWRU .npz -- .npy, .npz and
    .csv all have to work, since that is what the deployment path accepts."""
    from domain_adaptation.infer import load_signal_file

    rng = np.random.default_rng(0)
    sig = rng.standard_normal(5000)
    np.save(tmp_path / "a.npy", sig)
    np.savez(tmp_path / "b.npz", vib=sig)
    (tmp_path / "c.csv").write_text("\n".join(str(v) for v in sig))

    assert load_signal_file(tmp_path / "a.npy").shape == (5000,)
    assert load_signal_file(tmp_path / "b.npz").shape == (5000,)          # single array: key inferred
    assert load_signal_file(tmp_path / "b.npz", "vib").shape == (5000,)   # or named
    assert np.allclose(load_signal_file(tmp_path / "c.csv"), sig, atol=1e-6)


@requires_torch
def test_infer_rejects_ambiguous_npz_and_unknown_formats(tmp_path):
    from domain_adaptation.infer import load_signal_file

    np.savez(tmp_path / "two.npz", a=np.zeros(10), b=np.zeros(10))
    with pytest.raises(ValueError, match="signal-key"):
        load_signal_file(tmp_path / "two.npz")
    (tmp_path / "x.wav").write_bytes(b"nope")
    with pytest.raises(ValueError, match="unsupported format"):
        load_signal_file(tmp_path / "x.wav")


@requires_torch
def test_infer_explains_itself_when_the_signal_is_too_short(tmp_path):
    """A too-short recording is the most likely first-contact failure for a
    new user, so it must say what to do, not raise an opaque shape error."""
    from domain_adaptation.infer import score_signal

    config = make_config(tmp_path)
    with pytest.raises(ValueError, match="too short|Record a longer"):
        score_signal(np.zeros(50), fs_hz=1000, config=config, model=None, normalizer=None, device="cpu")


@requires_torch
def test_gradient_reversal_layer_flips_gradient_sign():
    from domain_adaptation.models.grl import GradientReversalLayer

    x = torch.tensor([2.0, 3.0], requires_grad=True)
    grl = GradientReversalLayer(lambda_=0.5)
    y = grl(x)
    assert torch.allclose(y, x)  # forward pass is identity

    y.sum().backward()
    assert torch.allclose(x.grad, torch.tensor([-0.5, -0.5]))  # backward pass negates and scales


@requires_torch  # grl.py imports torch unconditionally even though this function itself is pure math
def test_grl_lambda_schedule_endpoints():
    from domain_adaptation.models.grl import grl_lambda_schedule

    assert grl_lambda_schedule(0.0) == pytest.approx(0.0, abs=1e-6)
    assert 0.9 < grl_lambda_schedule(1.0) < 1.0


@requires_torch
def test_feature_classifier_forward_shapes(tmp_path):
    from domain_adaptation.models.dann import N_CLASSES, FeatureClassifier

    config = make_config(tmp_path)
    model = FeatureClassifier(input_shape=(16, 16), config=config)
    x = torch.randn(4, 1, 16, 16)
    features, logits = model(x)
    assert features.shape == (4, config.model.feature_dim)
    assert logits.shape == (4, N_CLASSES)


@requires_torch
def test_dann_model_forward_shapes(tmp_path):
    from domain_adaptation.models.dann import N_CLASSES, N_DOMAINS, DANNModel

    config = make_config(tmp_path)
    model = DANNModel(input_shape=(16, 16), config=config)
    x = torch.randn(4, 1, 16, 16)
    features, class_logits, domain_logits = model(x)
    assert features.shape == (4, config.model.feature_dim)
    assert class_logits.shape == (4, N_CLASSES)
    assert domain_logits.shape == (4, N_DOMAINS)


@requires_torch
def test_healthy_anchor_loss_zero_for_identical_healthy_features():
    from domain_adaptation.models.dann import healthy_anchor_loss

    features = torch.randn(6, 4)
    labels = torch.tensor([0, 0, 1, 1, 2, 2])
    loss = healthy_anchor_loss(features, labels, features[labels == 0])
    assert torch.allclose(loss, torch.tensor(0.0), atol=1e-5)


@requires_torch
def test_healthy_anchor_loss_returns_zero_with_no_healthy_source_rows():
    from domain_adaptation.models.dann import healthy_anchor_loss

    features = torch.randn(4, 4)
    labels = torch.tensor([1, 1, 2, 2])  # no label==0 rows
    target_features = torch.randn(3, 4)
    loss = healthy_anchor_loss(features, labels, target_features)
    assert torch.allclose(loss, torch.tensor(0.0))


@requires_torch
def test_tiny_dann_training_step_reduces_classification_loss(tmp_path):
    """One gradient step through the full DANN forward (classifier + GRL'd
    discriminator + healthy-anchor loss) should not blow up / NaN."""
    from domain_adaptation.models.dann import DANNModel, healthy_anchor_loss

    torch.manual_seed(0)
    config = make_config(tmp_path)
    model = DANNModel(input_shape=(16, 16), config=config)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-2)
    cls_criterion = torch.nn.CrossEntropyLoss()
    domain_criterion = torch.nn.CrossEntropyLoss()

    x_s = torch.randn(6, 1, 16, 16)
    label_s = torch.tensor([0, 0, 1, 1, 2, 2])
    domain_s = torch.tensor([0, 1, 0, 1, 0, 1])
    x_t = torch.randn(4, 1, 16, 16)
    domain_t = torch.full((4,), 2)

    losses = []
    for _ in range(5):
        model.set_grl_lambda(1.0)
        features_s, class_logits, domain_logits_s = model(x_s)
        features_t, _, domain_logits_t = model(x_t)
        cls_loss = cls_criterion(class_logits, label_s)
        domain_loss = domain_criterion(torch.cat([domain_logits_s, domain_logits_t]), torch.cat([domain_s, domain_t]))
        anchor_loss = healthy_anchor_loss(features_s, label_s, features_t)
        loss = cls_loss + domain_loss + anchor_loss

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        losses.append(cls_loss.item())

    assert all(np.isfinite(losses))
    assert losses[-1] < losses[0]
