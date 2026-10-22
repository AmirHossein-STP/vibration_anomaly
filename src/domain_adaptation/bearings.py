"""Bearing geometry for many bearings (MASTER_PLAN Phase 2 WP13, notes/23).

Three ways to describe a unit's bearing, all giving the fault orders the
physics features read (`bpfo`, `bpfi`, `ftf`, `bsf`, in orders of shaft speed):

    lookup("Rexnord ZA-2115")                 catalog designation
    from_dims(n_balls, d, D, contact_deg)     explicit geometry
    from_factors(bpfo=..., bpfi=...)          manufacturer fault factors (orders)

The catalog holds only bearings verified against a cited source; each entry's
published fault frequencies are checked in tests/test_bearings.py.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from domain_adaptation.features.envelope import BearingGeometry


@dataclass(frozen=True)
class FaultOrders:
    """Fault frequencies as multiples of shaft speed, when only factors are known.
    Duck-types BearingGeometry for envelope_order_features (bpfo, bpfi, ftf)."""

    name: str
    bpfo: float
    bpfi: float
    ftf: float = float("nan")
    bsf: float = float("nan")


def from_dims(n_balls: int, ball_diameter: float, pitch_diameter: float, contact_angle_deg: float = 0.0,
              name: str = "custom") -> BearingGeometry:
    if not (n_balls >= 3 and 0 < ball_diameter < pitch_diameter):
        raise ValueError(f"implausible geometry n={n_balls}, d={ball_diameter}, D={pitch_diameter}")
    return BearingGeometry(name, n_balls, ball_diameter, pitch_diameter, contact_angle_deg)


def from_factors(bpfo: float, bpfi: float, ftf: float | None = None, bsf: float | None = None,
                 name: str = "factors") -> FaultOrders:
    if not (0 < bpfo < bpfi):
        raise ValueError(f"expected 0 < BPFO < BPFI in orders, got {bpfo}, {bpfi}")
    return FaultOrders(name, float(bpfo), float(bpfi), float("nan") if ftf is None else float(ftf),
                       float("nan") if bsf is None else float(bsf))


def bsf_of(g: BearingGeometry) -> float:
    r = g.ball_diameter / g.pitch_diameter * np.cos(np.radians(g.contact_angle_deg))
    return g.pitch_diameter / (2 * g.ball_diameter) * (1 - r ** 2)


IN = 25.4
CATALOG: dict[str, tuple[BearingGeometry | FaultOrders, str]] = {
    "SKF 6205-2RS JEM": (BearingGeometry("SKF 6205-2RS JEM", 9, 0.3126, 1.537),
                         "CWRU bearing data center (drive end); BPFO 3.5848, BPFI 5.4152"),
    "6203 (Paderborn KAt)": (BearingGeometry("6203", 8, 6.75, 28.5), "Lessmeier et al. 2016; BPFO 3.053, BPFI 4.947"),
    "ER16K": (BearingGeometry("ER16K", 9, 0.3125, 1.516), "Huang & Baddour 2018; BPFO 3.57, BPFI 5.43"),
    "NSK 6203ZZ": (BearingGeometry("NSK 6203ZZ", 8, 6.77, 28.5), "Sehri et al. 2023, Table 2"),
    "FAFNIR 203KD": (BearingGeometry("FAFNIR 203KD", 8, 6.77, 28.5), "Sehri et al. 2023, Table 2"),
    "LDK UER204": (BearingGeometry("LDK UER204", 8, 7.92, (29.30 + 39.80) / 2),
                   "Lei et al., XJTU-SY tutorial, Table 1; BPFO 107.91 Hz at 35 Hz"),
    "MFPT test bearing": (FaultOrders("MFPT test bearing", 81.125 / 25, 118.875 / 25, 14.8375 / 25, 63.91 / 25),
                          "MFPT files: BPFO 81.125, BPFI 118.875, FTF 14.8375, BSF 63.91 Hz at 25 Hz"),
    "Rexnord ZA-2115": (BearingGeometry("Rexnord ZA-2115", 16, 0.331, 2.815, 15.17),
                        "Qiu et al. 2006 (IMS); BPFO 236.4, BPFI 296.9 Hz at 2000 rpm"),
    "61806-2RS": (BearingGeometry("61806-2RS", 19, 3.18, 35.5), "Paderborn LDM description, Table 1"),
}
# HUST (Thuan & Hong 2023): balls and ball diameter from the paper; pitch
# diameter is NOT published -- (bore + OD) / 2 is the standard approximation.
for _n, (_bore, _od, _dw, _z) in {"6204": (20, 47, 7.6, 8), "6205": (25, 52, 7.8, 9), "6206": (30, 62, 9.0, 9),
                                    "6207": (35, 72, 11.0, 9), "6208": (40, 80, 12.0, 9)}.items():
    CATALOG[f"KG {_n} (HUST)"] = (BearingGeometry(f"KG {_n}", _z, _dw, (_bore + _od) / 2),
                                  "Thuan & Hong 2023 (balls, Dw); pitch = (bore+OD)/2 approximation")


def lookup(designation: str) -> BearingGeometry | FaultOrders:
    if designation not in CATALOG:
        raise KeyError(f"bearing {designation!r} not in the catalog; give geometry (from_dims) or fault "
                       f"factors (from_factors). Known: {sorted(CATALOG)}")
    return CATALOG[designation][0]
