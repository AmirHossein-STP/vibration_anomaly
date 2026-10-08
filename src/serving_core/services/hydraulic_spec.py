"""Hydraulic cycle format (no torch import, so tools can use it)."""
GROUPS = {
    "slow": (60, ["TS1", "TS2", "TS3", "TS4", "VS1", "CE", "CP", "SE"]),
    "medium": (600, ["FS1", "FS2"]),
    "fast": (6000, ["PS1", "PS2", "PS3", "PS4", "PS5", "PS6", "EPS1"]),
}
SENSORS = [s for _, ss in GROUPS.values() for s in ss]
HEALTHY = {"cooler": 100, "valve": 100, "pump": 0, "accumulator": 130}
NOTES = {
    "cooler": "promising but unproven: score correlated with campaign time (r=0.88) in the study data",
    "valve": "usable only with a calibrated threshold; moderate separation in the study (ROC-AUC 0.89)",
    "accumulator": "usable only with a calibrated threshold; moderate separation in the study (ROC-AUC 0.87)",
    "pump": "EXPERIMENTAL: weak in the study (ROC-AUC 0.66); do not act on it alone",
}
DEFAULT_COMPONENTS = ["cooler", "valve", "accumulator"]
