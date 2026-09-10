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
from domain_adaptation.data.pu_loader import build_pu_catalog, load_vibration_signal
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
        cwru=CWRUConfig(sensor="DE", train_rpms=[1730, 1750], val_rpm=1797),
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


# --- models --------------------------------------------------------------

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
