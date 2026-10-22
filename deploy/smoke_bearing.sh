#!/usr/bin/env bash
# End-to-end check of the bearing image (WP17 gate): build, then a full
# calibrate -> promote -> infer -> rollback cycle on synthetic signals for two
# units, inside the container. Run from the repo root on a machine with Docker:
#   bash deploy/smoke_bearing.sh
set -euo pipefail
TAG=${TAG:-smoke}
DOCKER=${DOCKER:-docker}     # e.g. DOCKER="sudo docker" where the user is not in the docker group
$DOCKER build -q -f deploy/Dockerfile --target bearing -t ad-bearing:$TAG . >/dev/null
W=$(mktemp -d); mkdir -p "$W/data" "$W/models"; chmod -R 777 "$W"
run() { $DOCKER run --rm -u "$(id -u):$(id -g)" -v "$W/data:/data" -v "$W/models:/models" ad-bearing:$TAG "$@"; }
run init >/dev/null
PYTHONPATH=src python3 - "$W/data/bearing" <<'PY'
import sys, numpy as np
from pathlib import Path
b = Path(sys.argv[1]); FS, RPM = 12000, 900
def healthy(s): r = np.random.default_rng(s); t = np.arange(12 * FS) / FS; return 0.3 * np.sin(2 * np.pi * RPM / 60 * t) + r.normal(0, 1, len(t))
def faulty(s):
    x = healthy(s); ring = np.exp(-np.arange(300) / 40) * np.sin(2 * np.pi * 3200 * np.arange(300) / FS); k = 0.0
    while int(k) + 300 < len(x): x[int(k):int(k) + 300] += 6 * ring; k += FS / (3.0526 * RPM / 60)
    return x
(b / "config/service.yaml").write_text('fs_hz: 12000\nrpm: 900\nbearing: {designation: "6203 (Paderborn KAt)"}\nunits: {fan_b: {bearing: {designation: null}}}\n')
for u, s in (("pump_a", 0), ("fan_b", 50)):
    (b / "train" / u).mkdir(parents=True); (b / "in" / u).mkdir(parents=True)
    for i in range(10): np.save(b / "train" / u / f"h{i}.npy", healthy(s + i))
    np.save(b / "in" / u / "ok.npy", healthy(900)); np.save(b / "in" / u / "bad.npy", faulty(901))
PY
V1=$(run calibrate | grep "Candidate model:" | awk '{print $3}')
run promote "$V1" --yes >/dev/null
run infer >/dev/null
V2=$(run calibrate | grep "Candidate model:" | awk '{print $3}')
run promote "$V2" --yes >/dev/null && run rollback --yes >/dev/null
RES=$(ls -d "$W"/data/bearing/out/*/infer-*/results.csv | tail -1)
cat "$RES" | cut -d, -f1-8
python3 - "$RES" <<'PY'
import csv, sys
rows = {(r["unit"], r["file"]): r for r in csv.DictReader(open(sys.argv[1]))}
for u in ("pump_a", "fan_b"):
    assert rows[(u, "bad.npy")]["verdict"] in ("watch", "faulty") and rows[(u, "ok.npy")]["verdict"] == "healthy", u
assert rows[("pump_a", "bad.npy")]["fault_type"] == "outer race"
print("SMOKE OK: image builds; calibrate -> promote -> infer -> rollback for 2 units")
PY
rm -rf "$W"
