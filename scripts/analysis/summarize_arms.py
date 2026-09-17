import json, pathlib, statistics as st
# Recall and false-alarm both fell for order. That is the signature of a more
# CONSERVATIVE operating point, which would be a threshold difference rather
# than worse discrimination. F1 and balanced accuracy weigh both directions,
# so they separate the two explanations.
print(f"{'rep':>6}{'n':>4}{'F1':>16}{'balanced acc':>18}")
for rep in ("time", "order"):
    f1s, bals = [], []
    for seed in range(1, 9):
        p = pathlib.Path(f"outputs/da_reports_armA_{rep}_s{seed}/evaluation_metrics.json")
        if not p.exists(): continue
        b = json.loads(p.read_text())["baseline"]["binary_detection"]
        f1s.append(b["f1"])
        bals.append((b["recall"] + (1 - b["false_alarm_rate"])) / 2)
    print(f"{rep:>6}{len(f1s):>4}   {st.mean(f1s):.3f} +- {st.stdev(f1s):.3f}    {st.mean(bals):.3f} +- {st.stdev(bals):.3f}")
