"""Canonical bearing-code -> fault metadata for the Paderborn University /
KAt-DataCenter bearing dataset (Lessmeier et al. 2016, "Condition Monitoring
of Bearing Damage in Electromechanical Drive Systems").

Verified against the real dataset on 2026-09-10 (see
DOMAIN_ADAPTATION_PLAN.md's "Revisions" section): downloaded 3 real files
off the attached Kaggle mirror (`dippatel03/paderborn-db`) and paged
through its full file listing via the Kaggle API to confirm every code
below exists, with 20 runs under each of the dataset's 4 operating
conditions.

`universal_label` maps into the SAME 3-class space CWRU uses (see
cwru_loader.UNIVERSAL_LABELS): 0=healthy, 1=inner_race, 2=outer_race.
Combined-damage codes (KB*, both races damaged on one physical bearing)
get universal_label=None -- they don't fit a single-fault-type label and
are excluded from this project's classification task; kept here only for
reference/completeness.
"""
from __future__ import annotations

HEALTHY = ["K001", "K002", "K003", "K004", "K005", "K006"]
ARTIFICIAL_OUTER = ["KA01", "KA03", "KA05", "KA06", "KA07", "KA08", "KA09"]
ARTIFICIAL_INNER = ["KI01", "KI03", "KI05", "KI07", "KI08"]
REAL_OUTER = ["KA04", "KA15", "KA16", "KA22", "KA30"]
REAL_INNER = ["KI04", "KI14", "KI16", "KI17", "KI18", "KI21"]
REAL_COMBINED = ["KB23", "KB24", "KB27"]


def _build_table() -> dict[str, dict]:
    table: dict[str, dict] = {}
    for code in HEALTHY:
        table[code] = {"fault_type": "healthy", "damage_mode": "none", "universal_label": 0}
    for code in ARTIFICIAL_OUTER:
        table[code] = {"fault_type": "outer_race", "damage_mode": "artificial", "universal_label": 2}
    for code in ARTIFICIAL_INNER:
        table[code] = {"fault_type": "inner_race", "damage_mode": "artificial", "universal_label": 1}
    for code in REAL_OUTER:
        table[code] = {"fault_type": "outer_race", "damage_mode": "real", "universal_label": 2}
    for code in REAL_INNER:
        table[code] = {"fault_type": "inner_race", "damage_mode": "real", "universal_label": 1}
    for code in REAL_COMBINED:
        table[code] = {"fault_type": "combined", "damage_mode": "real", "universal_label": None}
    return table


PU_BEARING_CODES = _build_table()
