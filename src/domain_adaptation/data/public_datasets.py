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

import json
import re
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd
import scipy.io

from domain_adaptation.features.envelope import BearingGeometry

HEALTHY, INNER, OUTER, BALL, CAGE, COMBINED, INNER_BALL, OUTER_BALL = 0, 1, 2, 3, 4, 5, 6, 7
LABEL_NAMES = {0: "healthy", 1: "inner", 2: "outer", 3: "ball", 4: "cage", 5: "combined", 6: "inner+ball",
               7: "outer+ball"}
RACE_INVOLVING = (1, 2, 5, 6, 7)

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


# --- HUST (Thuan & Hong 2023): 5 bearing types, constant speed per file ------

HUST_RE = re.compile(r"^(?P<h>N|IO|IB|OB|I|O|B)(?P<type>[4-8])0(?P<load>[024])\.mat$")
HUST_LABEL = {"N": HEALTHY, "I": INNER, "O": OUTER, "B": BALL, "IO": COMBINED, "IB": INNER_BALL, "OB": OUTER_BALL}


def hust(root: Path, only: str | None = None):
    """`data` = 10 s at 51.2 kHz (vertical accelerometer on the bearing), `fs` =
    measured shaft frequency in Hz. Unit = bearing type (one physical bearing per
    type and state)."""
    for p in sorted(Path(root).glob("*.mat")):
        if only is not None and p.stem != only:
            continue
        m = HUST_RE.match(p.name)
        if not m:
            continue
        d = scipy.io.loadmat(p)
        shaft_hz = float(np.asarray(d["fs"]).ravel()[0])
        if not 5 < shaft_hz < 100:
            raise ValueError(f"{p.name}: implausible shaft speed {shaft_hz} Hz")
        btype = f"620{m['type']}"
        yield Recording("hust", unit=btype, condition=f"{int(m['load']) * 100}W", recording=p.stem,
                        label=HUST_LABEL[m["h"]], state="healthy" if m["h"] == "N" else "faulty",
                        fs=51200.0, signal=np.asarray(d["data"], dtype=np.float64).ravel(), rpm=60.0 * shaft_hz,
                        meta={"bearing": f"KG {btype} (HUST)", "load_W": int(m["load"]) * 100})


# --- Paderborn LDM run-to-failure (time-varying load and speed) --------------
# Condensed store (remote/datasets/condense_ldm.py): <root>/<B>/vibration_12k.npy
# int16 (records, 2, 19200) on the source volt grid, meta.json, <B>_operatingConditions.csv.

LDM_DEFECTS = {   # description PDF, Table 3 (found after dismantling)
    "B01": "IR,OR,B", "B02": "IR", "B03": "IR", "B04": "IR,B", "B05": "B", "B06": "IR,B", "B07": "IR",
    "B08": "IR,B", "B09": "OR,B", "B10": "OR,B", "B11": "IR,OR,B", "B12": "IR,B", "B13": "IR,B", "B14": "B",
    "B15": "IR,B", "B16": "IR,OR,B", "B17": "B",
}
LDM_OPCOLS = {"setDynLoad / N": "set_dyn_N", "peak_dynLoad / N": "peak_dyn_N", "setStatLoad / N": "set_stat_N",
              "meanAbs_statLoad / N": "stat_N", "setSpeed / rpm": "set_rpm", "meanAbs_speed / rpm": "speed_rpm"}


def ldm_bearings(root: Path) -> list[str]:
    return sorted(p.name for p in Path(root).iterdir() if (p / "vibration_12k.npy").is_file())


def ldm_records(root: Path, bearing: str, channel: int = 0):
    """One Recording per record: 1.6 s at 12 kHz in g, rpm = the record's measured mean speed."""
    bdir = Path(root) / bearing
    meta = json.loads((bdir / "meta.json").read_text())
    codes = np.load(bdir / "vibration_12k.npy", mmap_mode="r", allow_pickle=False)
    op = pd.read_csv(bdir / f"{bearing}_operatingConditions.csv").rename(columns=LDM_OPCOLS)
    if len(op) != codes.shape[0]:
        raise ValueError(f"{bearing}: {len(op)} operating-condition rows for {codes.shape[0]} records")
    step, offset, gpv = meta["step_volts"][channel], meta["offset_volts"][channel], meta["g_per_volt"]
    for k in range(codes.shape[0]):
        row = op.iloc[k]
        sig = (offset + step * np.asarray(codes[k, channel], dtype=np.float64)) * gpv
        yield Recording("paderborn_ldm", unit=bearing, condition="varying", recording=f"{k + 1:05d}", label=-1,
                        state="run", fs=12000.0, signal=sig, rpm=float(row.speed_rpm),
                        meta={"minute": k + 1, "channel": meta["channels"][channel],
                              "extra": {"record": k + 1, "time": row.Time, "speed_rpm": float(row.speed_rpm),
                                        "set_rpm": float(row.set_rpm), "stat_N": float(row.stat_N),
                                        "set_stat_N": float(row.set_stat_N), "peak_dyn_N": float(row.peak_dyn_N),
                                        "set_dyn_N": float(row.set_dyn_N), "defect": LDM_DEFECTS.get(bearing, "")}})
