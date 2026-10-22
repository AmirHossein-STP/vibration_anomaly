"""Loaders for the WP5 public datasets (MASTER_PLAN.md WP5, notes/17):
Ottawa 2018, Ottawa 2023, MFPT and XJTU-SY. CWRU and Paderborn keep their
own loaders.

Every loader yields `Recording`s: one continuous vibration signal at its
native rate with its provenance, label and shaft speed. Where speed varies
within a recording (Ottawa 2018), it comes as an encoder phase instead of a
scalar rpm. Datasets are read from the staged copies (`remote/box stage`,
data/<name>), treated as untrusted, and never executed.

Labels: 0 healthy, 1 inner race, 2 outer race (the universal labels of
cwru_loader / pu_bearing_codes), and further 3 ball, 4 cage, 5 combined.

Geometry sources (notes/17 §3):
- Ottawa 2018: ER16K, 9 balls, 0.3125/1.516 in (Huang & Baddour 2018: BPFO 3.57, BPFI 5.43).
- Ottawa 2023: NSK 6203ZZ / FAFNIR 203KD, 8 balls, 6.77/28.5 mm (Sehri et al. 2023, Table 2).
- MFPT: the BPFO/BPFI shipped in every file, 81.125/118.875 Hz at 25 Hz shaft speed.
- XJTU-SY: LDK UER204, 8 balls, 7.92 mm, pitch (29.30+39.80)/2 = 34.55 mm
  (Lei et al. tutorial, Table 1; its BPFO at 35 Hz is 107.91 Hz).
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd
import scipy.io

from domain_adaptation.features.envelope import BearingGeometry

HEALTHY, INNER, OUTER, BALL, CAGE, COMBINED = 0, 1, 2, 3, 4, 5
LABEL_NAMES = {0: "healthy", 1: "inner", 2: "outer", 3: "ball", 4: "cage", 5: "combined"}

OTTAWA18_ER16K = BearingGeometry("ER16K", n_balls=9, ball_diameter=0.3125, pitch_diameter=1.516)
OTTAWA23_6203 = BearingGeometry("6203ZZ / 203KD", n_balls=8, ball_diameter=6.77, pitch_diameter=28.5)
# MFPT ships its fault frequencies, not dimensions. The roller and pitch
# diameters of the MFPT documentation (0.235 / 1.245 in, 8 rollers, 0 deg)
# reproduce them: BPFO 3.245, BPFI 4.755 orders at 25 Hz = 81.1 / 118.9 Hz.
MFPT_BEARING = BearingGeometry("MFPT NICE", n_balls=8, ball_diameter=0.235, pitch_diameter=1.245)
XJTU_UER204 = BearingGeometry("LDK UER204", n_balls=8, ball_diameter=7.92, pitch_diameter=34.55)


@dataclass
class Recording:
    dataset: str
    unit: str            # the physical bearing (or, where one bearing per state, the state's bearing)
    condition: str       # operating condition / speed profile
    recording: str       # file or minute id, unique within the unit
    label: int
    state: str           # healthy / developing / faulty / run (XJTU)
    fs: float
    signal: np.ndarray
    rpm: float | None = None
    # time-varying speed: shaft phase in revolutions at the signal's sample times
    phase_revs: np.ndarray | None = None
    meta: dict = field(default_factory=dict)


# --- Ottawa 2018: time-varying speed, encoder phase ------------------------

OTT18_RE = re.compile(r"^(?P<h>[HIOBC])-(?P<profile>[ABCD])-(?P<trial>\d)\.mat$")
OTT18_LABEL = {"H": HEALTHY, "I": INNER, "O": OUTER, "B": BALL, "C": COMBINED}
OTT18_PROFILES = {"A": "increasing", "B": "decreasing", "C": "inc_then_dec", "D": "dec_then_inc"}


def encoder_phase(encoder: np.ndarray, fs: float, cpr: int = 1024) -> np.ndarray:
    """Shaft phase (revolutions) at every sample, from an incremental encoder.

    Rising edges at the mid-level are counted: edge k is phase k/cpr.
    Between edges the phase is interpolated linearly. Before the first and
    after the last edge it is extrapolated at the local rate.
    """
    lo, hi = np.percentile(encoder, [5, 95])
    b = encoder > (lo + hi) / 2
    edges = np.flatnonzero(~b[:-1] & b[1:]) + 1
    if len(edges) < 2 * cpr:
        raise ValueError(f"only {len(edges)} encoder edges -- not a {cpr}-CPR encoder signal?")
    t = np.arange(len(encoder), dtype=np.float64)
    return np.interp(t, edges, np.arange(len(edges)) / cpr, left=np.nan, right=np.nan)


def ottawa2018(root: Path, only: str | None = None):
    for p in sorted(Path(root).glob("*.mat")):
        if only is not None and p.stem != only:
            continue
        m = OTT18_RE.match(p.name)
        if not m:
            continue
        d = scipy.io.loadmat(p)
        vib = np.asarray(d["Channel_1"], dtype=np.float64).ravel()
        phase = encoder_phase(np.asarray(d["Channel_2"], dtype=np.float64).ravel(), 200000.0)
        ok = np.isfinite(phase)               # trim to the encoder-covered span
        h = m["h"]
        yield Recording("ottawa_2018", unit=f"ott18_{h}", condition=OTT18_PROFILES[m["profile"]],
                        recording=p.stem, label=OTT18_LABEL[h], state="healthy" if h == "H" else "faulty",
                        fs=200000.0, signal=vib[ok], phase_revs=phase[ok],
                        meta={"mean_shaft_hz": float((phase[ok][-1] - phase[ok][0]) / (ok.sum() / 200000.0))})


# --- Ottawa 2023: constant speed, healthy -> developing -> faulty per bearing

OTT23_RE = re.compile(r"^(?P<h>[HIOBC])_(?P<bearing>\d+)_(?P<state>[012])\.mat$")
OTT23_LABEL = {"H": HEALTHY, "I": INNER, "O": OUTER, "B": BALL, "C": CAGE}
OTT23_STATE = {"0": "healthy", "1": "developing", "2": "faulty"}


def ottawa2023(root: Path, only: str | None = None):
    for p in sorted(Path(root).glob("*.mat")):
        if only is not None and p.stem != only:
            continue
        m = OTT23_RE.match(p.name)
        if not m:
            continue
        a = np.asarray(scipy.io.loadmat(p)[p.stem], dtype=np.float64)
        # columns (Sehri et al. 2023 §2.1): accelerometer, microphone, speed
        # [rpm], load [N], temperature. Speed and load are stored once, as
        # the only nonzero entry of their column.
        rpm, load = float(a[:, 2].max()), float(a[:, 3].max())
        if not 1000 < rpm < 2500:
            raise ValueError(f"{p.name}: implausible speed {rpm} rpm in column 3")
        bearing = int(m["bearing"])
        yield Recording("ottawa_2023", unit=f"ott23_b{bearing:02d}", condition=f"{load:.0f}N",
                        recording=p.stem, label=OTT23_LABEL[m["h"]], state=OTT23_STATE[m["state"]],
                        fs=42000.0, signal=a[:, 0], rpm=rpm,
                        meta={"bearing": bearing, "load_N": load,
                              "fault_of_bearing": LABEL_NAMES[OTT23_LABEL[m["h"]]] if m["h"] != "H" else None})


# --- MFPT ---------------------------------------------------------------------

def mfpt(root: Path, only: str | None = None):
    for p in sorted(Path(root).rglob("*.mat")):
        if only is not None and p.stem != only:
            continue
        name = p.stem
        if name.startswith("baseline"):
            label = HEALTHY
        elif name.startswith("OuterRace"):
            label = OUTER
        elif name.startswith("InnerRace"):
            label = INNER
        else:
            continue
        b = scipy.io.loadmat(p, simplify_cells=True)["bearing"]
        rpm = 60.0 * float(b["rate"])
        yield Recording("mfpt", unit=name, condition=f"{float(b['load']):.0f}lbs", recording=name,
                        label=label, state="healthy" if label == HEALTHY else "faulty",
                        fs=float(b["sr"]), signal=np.asarray(b["gs"], dtype=np.float64).ravel(), rpm=rpm,
                        meta={"load_lbs": float(b["load"])})


# --- XJTU-SY: run to failure, one 1.28 s record per minute --------------------

# Failed element per bearing (Lei et al. tutorial, Table 2).
XJTU_FAILED = {
    "Bearing1_1": "outer", "Bearing1_2": "outer", "Bearing1_3": "outer", "Bearing1_4": "cage",
    "Bearing1_5": "inner+outer",
    "Bearing2_1": "inner", "Bearing2_2": "outer", "Bearing2_3": "cage", "Bearing2_4": "outer",
    "Bearing2_5": "outer",
    "Bearing3_1": "outer", "Bearing3_2": "inner+ball+cage+outer", "Bearing3_3": "inner", "Bearing3_4": "inner",
    "Bearing3_5": "outer",
}


def xjtu_bearings(root: Path) -> pd.DataFrame:
    df = pd.read_csv(Path(root) / "manifest.csv")
    df["failed"] = df.bearing.map(XJTU_FAILED)
    df["path"] = [str(Path(root) / f) for f in df.file]
    return df


def xjtu_records(root: Path, bearing: str, channel: int = 0):
    """Minute records of one bearing: channel 0 horizontal, 1 vertical."""
    row = xjtu_bearings(root).set_index("bearing").loc[bearing]
    arr = np.load(row.path, mmap_mode="r")
    for minute in range(arr.shape[0]):
        yield Recording("xjtu_sy", unit=bearing, condition=row.condition, recording=f"{minute + 1:04d}",
                        label=-1, state="run", fs=float(row.fs_hz),
                        signal=np.asarray(arr[minute, :, channel], dtype=np.float64), rpm=float(row.rpm),
                        meta={"minute": minute + 1, "n_minutes": int(row.n_minutes), "failed": row.failed,
                              "channel": "horizontal" if channel == 0 else "vertical"})
