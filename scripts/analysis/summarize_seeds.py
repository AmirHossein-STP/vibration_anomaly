import json, pathlib, statistics as st
rows = []
for seed in (1, 2, 3, 4, 5):
    p = pathlib.Path(f"outputs/da_reports_seed{seed}/evaluation_metrics.json")
    if not p.exists():
        print(f"seed {seed}: MISSING"); continue
    m = json.loads(p.read_text())["baseline"]
    b = m["binary_detection"]
    rows.append((seed, b["recall"], b["precision"], b["false_alarm_rate"], m["accuracy"]))

print(f"{'seed':>5}{'recall':>10}{'precision':>11}{'false alarm':>13}{'3cls acc':>10}")
for s, r, p_, f, a in rows:
    print(f"{s:>5}{r:>10.3f}{p_:>11.3f}{f:>13.1%}{a:>10.3f}")
if len(rows) > 1:
    rec = [r[1] for r in rows]; acc = [r[4] for r in rows]; fa = [r[3] for r in rows]
    print(f"\n{'mean':>5}{st.mean(rec):>10.3f}{st.mean([r[2] for r in rows]):>11.3f}"
          f"{st.mean(fa):>13.1%}{st.mean(acc):>10.3f}")
    print(f"{'std':>5}{st.stdev(rec):>10.3f}{st.stdev([r[2] for r in rows]):>11.3f}"
          f"{st.stdev(fa):>13.1%}{st.stdev(acc):>10.3f}")
    print(f"{'min':>5}{min(rec):>10.3f}")
    print(f"{'max':>5}{max(rec):>10.3f}")
    print(f"\nrecall range across seeds: {max(rec)-min(rec):.3f}")
    print(f"kernel v8 reported: 0.761   |   Phase 0 GPU run: 0.424")
