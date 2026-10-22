"""WP12 gate: method v1.0 (src/domain_adaptation/method.py) reproduces the
committed Phase 1 numbers exactly (notes/22 §parity).

Re-runs wp5_score.py and wp7_decision.py (both now call method.py) and
compares every summary number with the committed result files at git HEAD
(outputs/wp5/wp5_results.json, outputs/wp7/wp7_decision_results.json).

Usage (box): python scripts/analysis/wp12_parity.py
"""
from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

FILES = {"outputs/wp5/wp5_results.json": "scripts/analysis/wp5_score.py",
         "outputs/wp7/wp7_decision_results.json": "scripts/analysis/wp7_decision.py"}
HEADLINE = [("wp5", ["results", "ottawa_2018", "d_xf", "summary", "auc_race"]),
            ("wp5", ["results", "ottawa_2023:own", "d_xf", "summary", "auc_race"]),
            ("wp5", ["results", "mfpt", "d_xf", "summary", "auc_race"]),
            ("wp5", ["results", "pu:N09_M07_F10", "d_xf", "summary", "auc_race"]),
            ("wp7", ["horizontal", "rms_cusum", "detected"]),
            ("wp7", ["horizontal", "phys_cusum", "detected"])]


def numbers(obj, path=()):
    if isinstance(obj, dict):
        for k, v in obj.items():
            yield from numbers(v, path + (str(k),))
    elif isinstance(obj, list):
        for i, v in enumerate(obj):
            yield from numbers(v, path + (str(i),))
    elif isinstance(obj, (int, float)) and not isinstance(obj, bool):
        yield path, float(obj)


def main():
    committed = {}
    for f in FILES:
        r = subprocess.run(["git", "show", f"HEAD:{f}"], capture_output=True, text=True)
        # on the box there is no .git: its current files ARE the committed ones (they were pulled from it)
        committed[f] = json.loads(r.stdout if r.returncode == 0 else Path(f).read_text())
        print(f"reference {f}: {'git HEAD' if r.returncode == 0 else 'snapshot before re-run'}")
    for script in FILES.values():
        subprocess.run([sys.executable, script], check=True, capture_output=True)
    bad, total = [], 0
    for f in FILES:
        new = dict(numbers(json.loads(Path(f).read_text())))
        for path, v in numbers(committed[f]):
            total += 1
            w = new.get(path)
            if w is None or not (abs(w - v) <= 1e-12 or (v != v and w != w)):
                bad.append((f, "/".join(path), v, w))
    print(f"compared {total} numbers; {len(bad)} differ")
    for b in bad[:20]:
        print("  DIFF", b)
    for src, path in HEADLINE:
        d = json.loads(Path("outputs/wp5/wp5_results.json" if src == "wp5" else "outputs/wp7/wp7_decision_results.json").read_text())
        for k in path:
            d = d[k]
        print(f"  {src} {'/'.join(path)} = {d}")
    print("WP12 PARITY:", "PASS" if not bad else "FAIL")
    sys.exit(0 if not bad else 1)


if __name__ == "__main__":
    main()
