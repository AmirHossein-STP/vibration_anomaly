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


def _order_locked_signal(fs, rpm, fault_order, seconds=4.0, seed=0):
    """A bearing-like signal whose defect sits at a fixed ORDER, so its
    frequency in Hz moves with speed -- exactly the real situation."""
    rng = np.random.default_rng(seed)
    t = np.arange(int(fs * seconds)) / fs
    shaft_hz = rpm / 60.0
    sig = 0.05 * rng.standard_normal(len(t))
    for harmonic in (1, 2, 3):           # defect tone plus two harmonics
        sig += np.sin(2 * np.pi * fault_order * harmonic * shaft_hz * t) / harmonic
    return sig


def test_order_representation_aligns_two_speeds_that_time_representation_does_not():
    """The decisive check for the whole order-tracking idea, at the level the
    model actually sees: spectrograms.

    The same defect at the same ORDER is recorded at 1500rpm and 900rpm. In
    the time representation its Hz differ by 1500/900, so the spectrograms
    disagree. After angular resampling the speed divides out and they should
    line up.
    """
    from domain_adaptation.data.order_tracking import angular_resample
    from domain_adaptation.data.transforms import resample_to_rate
    from vibration_ad.features.spectrogram import windows_to_log_spectrograms
    from vibration_ad.data.windows import make_windows

    class STFT:
        nperseg, hop_length, log_epsilon = 256, 64, 1e-10
        noverlap = 256 - 64

    fs, fault_order, seg = 64000, 5.4, 4096

    def mean_spectrogram(rpm, mode):
        sig = _order_locked_signal(fs, rpm, fault_order)
        sig = (angular_resample(sig, fs, rpm, 256) if mode == "order"
               else resample_to_rate(sig, fs, 12000))
        w = make_windows(sig, seg, seg // 2)
        return windows_to_log_spectrograms(w, 12000, STFT).mean(axis=0)

    for mode, expectation in [("time", "disagree"), ("order", "agree")]:
        a, b = mean_spectrogram(1500, mode), mean_spectrogram(900, mode)
        corr = np.corrcoef(a.ravel(), b.ravel())[0, 1]
        if mode == "time":
            time_corr = corr
        else:
            order_corr = corr

    # Order tracking must bring the two speeds substantially closer together.
    assert order_corr > time_corr, (
        f"order tracking failed to help: time={time_corr:.3f} order={order_corr:.3f}"
    )
    assert order_corr > 0.9, f"speeds still not aligned in order domain: {order_corr:.3f}"


def test_dataset_builds_in_order_representation(tmp_path):
    """The switch has to work end to end, not just on bare signals."""
    config = make_config(tmp_path)
    config.common.representation = "order"
    config.common.samples_per_rev = 64        # small, to suit the tiny test signals
    config.common.segment_length = 128
    config.common.train_stride, config.common.eval_stride = 32, 64
    config.stft.nperseg, config.stft.hop_length = 32, 8
    config.cwru.include_in_source = False
    config.pu.native_fs_hz = 4000

    pu_root = tmp_path / "pu_root"
    for run in range(1, 5):
        for code, cond in [("K001", "N15_M07_F10"), ("KA01", "N15_M07_F10"), ("KI01", "N15_M07_F10"),
                           ("K001", "N09_M07_F10"), ("KA04", "N09_M07_F10"), ("KI04", "N09_M07_F10")]:
            write_synthetic_pu_mat(pu_root, code, cond, run, n_samples=8000)

    results = da_dataset.build_domain_adaptation_datasets(config, pu_root)
    for role in da_dataset.ROLES:
        X, meta = results[role]
        assert len(X) == len(meta) > 0, role


def test_unknown_representation_is_rejected(tmp_path):
    config = make_config(tmp_path)
    config.common.representation = "wavelet"
    with pytest.raises(ValueError, match="must be 'time' or 'order'"):
        da_dataset._to_common_representation(np.zeros(1000), 12000, 1500, config)


# --- condensed PU store (what actually gets moved to the GPU server) --------

def _build_prepared_store(tmp_path, codes_conditions, runs=(1, 2, 3, 4), n_samples=64000):
    """Synthetic raw PU -> prepare_pu -> condensed store, returning both roots."""
    from domain_adaptation.data.prepare_pu import prepare_code
    import pandas as pd

    raw = tmp_path / "raw_pu"
    for code, cond in codes_conditions:
        for run in runs:
            write_synthetic_pu_mat(raw, code, cond, run, n_samples=n_samples)
    prepared = tmp_path / "prepared_pu"
    rows = []
    for code in sorted({c for c, _ in codes_conditions}):
        rows += prepare_code(raw / code, prepared, fs_out=25600)["rows"]
    pd.DataFrame(rows).to_csv(prepared / "manifest.csv", index=False)
    return raw, prepared


def test_prepared_store_is_detected_and_raw_is_not(tmp_path):
    from domain_adaptation.data.pu_loader import is_prepared_store

    raw, prepared = _build_prepared_store(tmp_path, [("K001", "N15_M07_F10")])
    assert is_prepared_store(prepared)      # has manifest.csv
    assert not is_prepared_store(raw)       # bare bearing-code folders
    # Auto-detection is what lets one --pu-root serve both forms.


def test_prepared_store_round_trips_signals_at_the_reduced_rate(tmp_path):
    from domain_adaptation.data.pu_loader import PreparedPUStore

    _, prepared = _build_prepared_store(tmp_path, [("K001", "N15_M07_F10")], runs=(1,), n_samples=64000)
    store = PreparedPUStore(prepared)
    sig = store.load("K001", "N15_M07_F10", 1)
    # 64000 samples at 64kHz -> 25600 at 25.6kHz: exactly the 2/5 ratio.
    assert len(sig) == 25600
    assert np.isfinite(sig).all()


def test_prepared_catalog_matches_the_raw_catalog_schema(tmp_path):
    from domain_adaptation.data.pu_loader import PreparedPUStore, build_pu_catalog

    raw, prepared = _build_prepared_store(
        tmp_path, [("K001", "N15_M07_F10"), ("KA04", "N09_M07_F10")]
    )
    raw_cat = build_pu_catalog(raw, ["K001", "KA04"])
    prep_cat = PreparedPUStore(prepared).catalog(["K001", "KA04"])
    assert list(raw_cat.columns) == list(prep_cat.columns)      # identical schema
    assert len(raw_cat) == len(prep_cat)
    assert set(raw_cat.code) == set(prep_cat.code)
    assert set(raw_cat.universal_label) == set(prep_cat.universal_label)
    assert set(prep_cat.fs_native_hz) == {25600}                # already downsampled
    assert set(raw_cat.fs_native_hz) == {64000}


def test_prepared_store_missing_code_fails_loudly(tmp_path):
    from domain_adaptation.data.pu_loader import PreparedPUStore

    _, prepared = _build_prepared_store(tmp_path, [("K001", "N15_M07_F10")])
    with pytest.raises(FileNotFoundError, match="KA04"):
        PreparedPUStore(prepared).catalog(["K001", "KA04"])


def test_dataset_build_is_equivalent_through_either_storage_form(tmp_path):
    """The point of the condensed store: identical downstream results.

    Values are close rather than bit-identical because the raw path
    resamples 64k -> 12k in one polyphase step while the prepared path goes
    64k -> 25.6k -> 12k in two. The overall ratio is the same 3/16, and the
    25.6 kHz intermediate sits well above twice the final 6 kHz Nyquist, so
    no content the final signal needs is lost -- only the anti-alias filter
    shaping differs slightly.
    """
    codes_conditions = [(c, cond) for c in ("K001", "KA01", "KI01") for cond in ("N15_M07_F10",)]
    codes_conditions += [(c, "N09_M07_F10") for c in ("K001", "KA04", "KI04")]
    raw, prepared = _build_prepared_store(tmp_path, codes_conditions, n_samples=64000)

    results = {}
    for name, root in [("raw", raw), ("prepared", prepared)]:
        config = make_config(tmp_path / name)
        config.cwru.include_in_source = False
        config.pu.native_fs_hz = 64000
        config.common.fs_hz = 12000
        config.common.segment_length, config.common.train_stride, config.common.eval_stride = 4096, 512, 1024
        config.stft.nperseg, config.stft.hop_length = 256, 32
        results[name] = da_dataset.build_domain_adaptation_datasets(config, root)

    for role in da_dataset.ROLES:
        X_raw, m_raw = results["raw"][role]
        X_prep, m_prep = results["prepared"][role]
        assert X_raw.shape == X_prep.shape, f"{role}: shape differs"
        assert list(m_raw.universal_label) == list(m_prep.universal_label), f"{role}: labels differ"
        assert list(m_raw.code) == list(m_prep.code), f"{role}: provenance differs"
        # Same content, marginally different filter shaping.
        assert np.corrcoef(X_raw.ravel(), X_prep.ravel())[0, 1] > 0.98, f"{role}: content diverged"


# --- order tracking (the cross-speed fix) -----------------------------------

def _peak_order(signal, samples_per_rev=256):
    """Order of the dominant spectral component of an angular-resampled signal."""
    spec = np.abs(np.fft.rfft(signal * np.hanning(len(signal))))
    freqs = np.fft.rfftfreq(len(signal))          # cycles per SAMPLE
    return float(freqs[np.argmax(spec[1:]) + 1] * samples_per_rev)  # -> per REVOLUTION


def test_angular_resample_puts_a_fault_at_the_same_ORDER_at_any_speed():
    """The whole reason order tracking should fix what adversarial alignment
    could not. A bearing fault sits at a fixed number of events per
    REVOLUTION, so its frequency in Hz scales with shaft speed -- which is a
    multiplicative stretch of the spectrogram's frequency axis, something a
    CNN has no tolerance for. After angular resampling the speed divides out
    and the fault lands in the same bin at both speeds.
    """
    from domain_adaptation.data.order_tracking import angular_resample

    fs, fault_order, duration = 64000, 5.4, 4.0   # 5.4 orders ~ a typical BPFO
    t = np.arange(int(fs * duration)) / fs
    peaks_hz, peaks_order = {}, {}
    for rpm in (900, 1500):                        # PU's N09 and N15
        shaft_hz = rpm / 60.0
        sig = np.sin(2 * np.pi * fault_order * shaft_hz * t)
        peaks_hz[rpm] = fault_order * shaft_hz
        peaks_order[rpm] = _peak_order(angular_resample(sig, fs, rpm))

    # In Hz the same defect appears in completely different places...
    assert peaks_hz[900] == pytest.approx(81.0)
    assert peaks_hz[1500] == pytest.approx(135.0)
    # ...but in the order domain both land on the true fault order.
    assert peaks_order[900] == pytest.approx(fault_order, abs=0.05)
    assert peaks_order[1500] == pytest.approx(fault_order, abs=0.05)
    assert peaks_order[900] == pytest.approx(peaks_order[1500], abs=0.05)


def test_angular_resample_length_tracks_revolutions_not_seconds():
    """A fixed-length window covers a fixed number of REVOLUTIONS at any
    speed -- the invariance the representation is bought for. Four seconds
    at 1500rpm is 100 revolutions; at 900rpm it is 60."""
    from domain_adaptation.data.order_tracking import angular_resample, revolutions_per_window

    four_seconds = np.zeros(64000 * 4)
    assert len(angular_resample(four_seconds, 64000, 1500)) == 100 * 256
    assert len(angular_resample(four_seconds, 64000, 900)) == 60 * 256
    assert revolutions_per_window(8384) == pytest.approx(32.75)


def test_angular_resample_rejects_nonpositive_rpm():
    from domain_adaptation.data.order_tracking import angular_resample

    with pytest.raises(ValueError, match="rpm must be positive"):
        angular_resample(np.zeros(100), 12000, 0)


def test_order_axis_spans_zero_to_nyquist_order():
    from domain_adaptation.data.order_tracking import order_axis

    axis = order_axis(256, samples_per_rev=256)
    assert axis[0] == 0.0
    assert axis[-1] == pytest.approx(128.0)   # Nyquist order = samples_per_rev / 2
    assert len(axis) == 129


def test_angular_resample_on_real_cwru_signal_recovers_injected_order():
    """Same check but on a REAL recording already in data/raw/ (no download):
    inject a known order into genuine CWRU vibration and recover it."""
    from domain_adaptation.data.order_tracking import angular_resample

    raw = Path("data/raw/1797_Normal.npz")
    if not raw.exists():
        pytest.skip("CWRU data/raw not present")
    with np.load(raw) as d:
        sig = d["DE"].astype(np.float64).ravel()[:120000]

    rpm, fs, injected = 1797, 12000, 7.0
    t = np.arange(len(sig)) / fs
    # Inject well above the real signal's amplitude so the peak is unambiguous.
    probe = sig + 20 * sig.std() * np.sin(2 * np.pi * injected * (rpm / 60.0) * t)
    assert _peak_order(angular_resample(probe, fs, rpm)) == pytest.approx(injected, abs=0.1)


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
