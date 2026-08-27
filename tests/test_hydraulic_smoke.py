"""Fast sanity tests using synthetic data -- no real dataset download
required. These check the plumbing (shapes line up end to end,
split/validation logic is correct, per-component `is_healthy` semantics
are right -- see notes/09_hydraulic_notes.md's per-component redesign),
not model quality.

Run with: .venv/bin/python -m pytest tests/test_hydraulic_smoke.py
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
import torch

from hydraulic_fusion_ad.config import (
    Config,
    DataConfig,
    EvaluationConfig,
    ModelConfig,
    PathsConfig,
    SensorGroupConfig,
    SplitConfig,
    TrainConfig,
)
from hydraulic_fusion_ad.data.catalog import build_catalog
from hydraulic_fusion_ad.data.dataset import build_raw_datasets
from hydraulic_fusion_ad.features.normalize import MultiBranchNormalizer
from hydraulic_fusion_ad.models.fusion_autoencoder import (
    FusionAutoencoder,
    combined_mse,
    conv_layer_plan,
    per_branch_mse,
)

# Small sensor-group shapes, chosen so conv_layer_plan() (min_final_length=8,
# see models/fusion_autoencoder.py) still produces >=1 Conv1d layer for the
# medium/fast branches while staying cheap to run in a test.
SENSOR_GROUPS = {
    "slow": SensorGroupConfig(rate_hz=1, samples_per_cycle=4, sensors=["T1", "T2"]),
    "medium": SensorGroupConfig(rate_hz=10, samples_per_cycle=32, sensors=["F1"]),
    "fast": SensorGroupConfig(rate_hz=100, samples_per_cycle=64, sensors=["P1"]),
}
PROFILE_COLUMNS = ["cooler_condition", "valve_condition", "pump_leakage", "accumulator_pressure", "stable_flag"]
HEALTHY_VALUES = {"cooler_condition": 100, "valve_condition": 100, "pump_leakage": 0, "accumulator_pressure": 130}
VALID_VALUES = {
    "cooler_condition": [3, 20, 100],
    "valve_condition": [73, 80, 90, 100],
    "pump_leakage": [0, 1, 2],
    "accumulator_pressure": [90, 100, 115, 130],
    "stable_flag": [0, 1],
}


def make_config(
    tmp_path: Path,
    n_cycles: int,
    train_fraction: float = 0.5,
    val_fraction: float = 0.34,
    target_component: str = "cooler_condition",
    block_size: int = 1,
) -> Config:
    return Config(
        seed=0,
        paths=PathsConfig(raw_dir="raw", processed_dir="processed", models_dir="models", figures_dir="figures", reports_dir="reports"),
        data=DataConfig(
            n_cycles_expected=n_cycles,
            target_component=target_component,
            profile_columns=PROFILE_COLUMNS,
            healthy_values=HEALTHY_VALUES,
            valid_values=VALID_VALUES,
            sensor_groups=SENSOR_GROUPS,
        ),
        # block_size=1 by default (test datasets are tiny): equivalent to a
        # pure per-cycle random shuffle, same round()-based train/val/test
        # COUNT math as any other block_size, so existing count assertions
        # below don't need to change -- see data/dataset.py's
        # _block_shuffle_split for why block_size>1 matters on real data.
        split=SplitConfig(train_fraction=train_fraction, val_fraction=val_fraction, block_size=block_size),
        model=ModelConfig(
            slow_hidden_dim=6, slow_branch_latent=3,
            medium_base_channels=2, medium_branch_latent=4,
            fast_base_channels=2, fast_branch_latent=5,
            fused_latent_dim=6,
            branch_weights={"slow": 1.0, "medium": 1.0, "fast": 1.0},
        ),
        train=TrainConfig(batch_size=4, epochs=1, learning_rate=1e-2, weight_decay=0.0, early_stopping_patience=1, device="cpu"),
        evaluation=EvaluationConfig(threshold_percentile=99.0),
        project_root=tmp_path,
    )


def write_synthetic_raw(raw_dir: Path, rows: list[dict]) -> None:
    """rows: one dict per cycle, keys = PROFILE_COLUMNS. Sensor values are
    random per cycle -- only profile.txt's values matter for these tests.
    """
    raw_dir.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(0)
    n = len(rows)

    profile = np.array([[r[c] for c in PROFILE_COLUMNS] for r in rows])
    np.savetxt(raw_dir / "profile.txt", profile, fmt="%d", delimiter="\t")

    for group in SENSOR_GROUPS.values():
        for sensor in group.sensors:
            arr = rng.standard_normal((n, group.samples_per_cycle)).astype(np.float32)
            np.savetxt(raw_dir / f"{sensor}.txt", arr, delimiter="\t")


def healthy_row(**overrides) -> dict:
    row = {**HEALTHY_VALUES, "stable_flag": 0}
    row.update(overrides)
    return row


def test_build_catalog_flags_healthy_and_degraded(tmp_path):
    # target_component="cooler_condition" (make_config's default): is_healthy
    # must track ONLY cooler_condition -- row 2's pump/valve degradation
    # shouldn't flip it, since this detector doesn't consider those columns
    # at all (see notes/09_hydraulic_notes.md's per-component redesign).
    config = make_config(tmp_path, n_cycles=4)
    rows = [
        healthy_row(),
        healthy_row(cooler_condition=20),
        healthy_row(pump_leakage=2, valve_condition=90),  # two OTHER components off
        healthy_row(),
    ]
    write_synthetic_raw(tmp_path / "raw", rows)

    group_arrays, meta = build_catalog(tmp_path / "raw", config)

    assert set(group_arrays.keys()) == {"slow", "medium", "fast"}
    assert group_arrays["slow"].shape == (4, 2, 4)
    assert group_arrays["medium"].shape == (4, 1, 32)
    assert group_arrays["fast"].shape == (4, 1, 64)

    # is_healthy: target (cooler_condition) only -- row 2 stays healthy despite
    # its OTHER degradations, since target_component="cooler_condition" here.
    assert meta["is_healthy"].tolist() == [True, False, True, True]
    # degraded_component: diagnostic across ALL FOUR columns, independent of target.
    assert meta.loc[1, "degraded_component"] == "cooler_condition"
    assert meta.loc[2, "degraded_component"] == "valve_condition,pump_leakage"


def test_build_catalog_rejects_unexpected_profile_value(tmp_path):
    config = make_config(tmp_path, n_cycles=1)
    write_synthetic_raw(tmp_path / "raw", [healthy_row(cooler_condition=42)])  # 42 not in VALID_VALUES
    with pytest.raises(ValueError, match="cooler_condition"):
        build_catalog(tmp_path / "raw", config)


def test_split_is_disjoint_and_covers_all_cycles(tmp_path):
    # target_component="pump_leakage": the last 4 rows' pump_leakage=1
    # override must be what drives is_healthy False here. block_size=1 ->
    # per-cycle shuffle, same train/val/test COUNT math as any block_size
    # (see make_config's comment) -- this test checks partition counts and
    # disjointness, not which specific cycles land where.
    config = make_config(tmp_path, n_cycles=10, train_fraction=0.5, val_fraction=0.3, target_component="pump_leakage")
    rows = [healthy_row() for _ in range(6)] + [healthy_row(pump_leakage=1) for _ in range(4)]
    write_synthetic_raw(tmp_path / "raw", rows)

    train_arrays, val_arrays, val_meta, test_arrays, test_meta = build_raw_datasets(config)

    n_train = len(train_arrays["slow"])
    n_val = len(val_arrays["slow"])
    n_test = len(test_arrays["slow"])
    assert n_train == 3  # 50% of 6 healthy cycles
    assert n_val == 2  # 30% of 6 (rounded)
    assert n_test == 6 - n_train - n_val + 4  # remaining healthy + all 4 degraded
    assert val_meta["is_healthy"].all()
    assert test_meta["is_healthy"].sum() == 6 - n_train - n_val
    # train/val/test-healthy cycle_idx sets are pairwise disjoint and drawn
    # only from the 6 healthy cycles (the property block-shuffling must
    # preserve, whatever the specific shuffle order is).
    healthy_test_idx = set(test_meta.loc[test_meta.is_healthy, "cycle_idx"])
    val_idx_set = set(val_meta["cycle_idx"])
    assert len(healthy_test_idx & val_idx_set) == 0  # disjoint
    assert healthy_test_idx | val_idx_set <= set(range(6))  # both drawn only from the 6 healthy cycles
    # cached second call returns identical arrays without touching raw files
    train2, *_ = build_raw_datasets(config)
    assert np.array_equal(train2["slow"], train_arrays["slow"])


def test_normalizer_roundtrip_is_standardized():
    rng = np.random.default_rng(0)
    arrays = {
        "slow": rng.normal(5.0, 2.0, size=(20, 2, 4)).astype(np.float32),
        "medium": rng.normal(-3.0, 0.5, size=(20, 1, 32)).astype(np.float32),
        "fast": rng.normal(100.0, 10.0, size=(20, 1, 64)).astype(np.float32),
    }
    normalizer = MultiBranchNormalizer().fit(arrays)
    normed = normalizer.transform(arrays)
    for group, arr in normed.items():
        assert abs(arr.mean()) < 0.1, group
        assert abs(arr.std() - 1.0) < 0.1, group


def test_normalizer_raises_before_fit():
    with pytest.raises(RuntimeError):
        MultiBranchNormalizer().transform({"slow": np.zeros((1, 2, 4), dtype=np.float32)})


def test_conv_layer_plan_halves_cleanly():
    n_layers, final_length = conv_layer_plan(64, max_layers=4, min_final_length=8)
    assert final_length * (2**n_layers) == 64
    assert n_layers >= 1


def test_fusion_autoencoder_forward_preserves_shapes(tmp_path):
    config = make_config(tmp_path, n_cycles=1)
    model = FusionAutoencoder(config)
    x = {
        "slow": torch.randn(3, 2, 4),
        "medium": torch.randn(3, 1, 32),
        "fast": torch.randn(3, 1, 64),
    }
    recon = model(x)
    for group in x:
        assert recon[group].shape == x[group].shape


def test_per_branch_and_combined_mse_zero_for_identical_input():
    x = {"slow": torch.randn(4, 2, 4), "medium": torch.randn(4, 1, 32)}
    branch_errors = per_branch_mse(x, x)
    for group in x:
        assert torch.allclose(branch_errors[group], torch.zeros(4))
    combined = combined_mse(branch_errors, {"slow": 1.0, "medium": 2.0})
    assert torch.allclose(combined, torch.zeros(4))


def test_tiny_training_step_reduces_loss(tmp_path):
    """One gradient step on a trivial batch should not blow up / NaN."""
    torch.manual_seed(0)
    config = make_config(tmp_path, n_cycles=1)
    model = FusionAutoencoder(config)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-2)
    x = {
        "slow": torch.randn(4, 2, 4),
        "medium": torch.randn(4, 1, 32),
        "fast": torch.randn(4, 1, 64),
    }

    losses = []
    for _ in range(5):
        recon = model(x)
        branch_errors = per_branch_mse(x, recon)
        loss = combined_mse(branch_errors, config.model.branch_weights).mean()
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        losses.append(loss.item())

    assert all(np.isfinite(losses))
    assert losses[-1] < losses[0]
