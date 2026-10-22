"""L0 ISO 20816 zones (WP19)."""
import numpy as np
import pytest

from domain_adaptation.iso20816 import G, velocity_rms_mm_s, zone


def test_velocity_of_a_pure_tone_is_exact():
    fs, f0, v_peak = 12000.0, 50.0, 4.0e-3            # 4 mm/s peak velocity at 50 Hz
    t = np.arange(int(fs * 4)) / fs
    acc_ms2 = v_peak * 2 * np.pi * f0 * np.cos(2 * np.pi * f0 * t)
    v = velocity_rms_mm_s(acc_ms2 / G, fs, unit="g", rpm=1500)
    assert v == pytest.approx(4.0 / np.sqrt(2), rel=0.02)


def test_out_of_band_content_is_ignored():
    fs = 12000.0
    t = np.arange(int(fs * 2)) / fs
    acc = 50.0 * np.sin(2 * np.pi * 3000 * t)            # 3 kHz: above the 1 kHz ISO band
    assert velocity_rms_mm_s(acc, fs, unit="m/s2", rpm=1500) < 0.05


def test_zone_boundaries():
    assert zone(1.0, 2, "rigid") == "A" and zone(2.0, 2, "rigid") == "B"
    assert zone(4.0, 2, "rigid") == "C" and zone(5.0, 2, "rigid") == "D"
    assert zone(8.0, 1, "flexible") == "C"
