"""CWRU catalog, remapped onto the universal 3-class label space.

Reuses vibration_ad.data.catalog.build_catalog() for the actual filename
parsing (rpm / fault_type / fault_size_mils) -- no need to reparse CWRU's
naming convention a second time, see vibration_ad's notes/01_dataset_notes.md
for that logic. This module only adds what's specific to the
domain-adaptation task: mapping CWRU's 5 fault labels onto the universal
3-class space shared with Paderborn, and tagging each file with a
train/val split role.

Universal label space (shared with pu_loader.py):
    0 = healthy, 1 = inner race fault, 2 = outer race fault.
Ball ("B") faults are DROPPED here -- Paderborn's real-damage bearings are
only ever inner- or outer-race (see pu_bearing_codes.py), so there's no
cross-domain ball-fault class to align against; keeping CWRU's ball faults
would just be source-only noise the classifier has nowhere to transfer.
See DOMAIN_ADAPTATION_PLAN.md's "Revisions" section. CWRU's own
`vibration_ad` experiment is unaffected -- it still uses all fault types.
"""
from __future__ import annotations

from pathlib import Path

import pandas as pd

from vibration_ad.data.catalog import build_catalog

UNIVERSAL_LABELS = {"healthy": 0, "inner_race": 1, "outer_race": 2}

_FAULT_TYPE_TO_UNIVERSAL = {
    "Normal": "healthy",
    "IR": "inner_race",
    "OR@3": "outer_race",
    "OR@6": "outer_race",
    "OR@12": "outer_race",
    # "B" (ball) has no entry on purpose -> dropped by the isin() filter below.
}


def build_cwru_catalog(raw_dir: Path, train_rpms: list[int], val_rpm: int) -> pd.DataFrame:
    """One row per CWRU file with a universal-label mapping (ball faults
    dropped), split 'train'/'val' by RPM family -- same anti-leakage
    reasoning as vibration_ad's own RPM holdout: a random split would mix
    near-duplicate overlapping windows from the same long recording across
    train and val.

    Returns columns: filepath, domain, condition, fs_native_hz, fault_type,
    universal_label, split.
    """
    catalog = build_catalog(Path(raw_dir))
    catalog = catalog[catalog.fault_type.isin(_FAULT_TYPE_TO_UNIVERSAL)].copy()
    if catalog.empty:
        raise RuntimeError(f"No CWRU files with a universal-label mapping found in {raw_dir}")

    catalog["universal_label"] = catalog.fault_type.map(_FAULT_TYPE_TO_UNIVERSAL).map(UNIVERSAL_LABELS)
    catalog["domain"] = "cwru"
    catalog["fs_native_hz"] = 12000
    catalog["condition"] = catalog.rpm.astype(str)

    known_rpms = set(train_rpms) | {val_rpm}
    present_rpms = set(catalog.rpm.unique().tolist())
    unknown = present_rpms - known_rpms
    if unknown:
        raise ValueError(
            f"CWRU catalog has RPMs not covered by train_rpms/val_rpm: {sorted(unknown)}"
        )
    if val_rpm not in present_rpms:
        raise ValueError(f"val_rpm={val_rpm} not present in CWRU catalog RPMs {sorted(present_rpms)}")

    catalog["split"] = catalog.rpm.apply(lambda r: "val" if r == val_rpm else "train")
    cols = ["filepath", "domain", "condition", "fs_native_hz", "fault_type", "universal_label", "split"]
    return catalog[cols].reset_index(drop=True)
