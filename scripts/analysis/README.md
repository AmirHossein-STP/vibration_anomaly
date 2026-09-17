# Analysis scripts for the GPU-server runs

These read the per-run artifacts a sweep leaves in
`outputs/da_reports_<tag>/` (`evaluation_metrics.json`,
`target_test_predictions.csv`) and aggregate ACROSS runs. They live here
rather than in `src/` because they answer questions about a set of
experiments, not about a single model.

They exist because of one finding: **run-to-run variance on this task is
larger than most effects being compared.** Six identical-config runs
differing only in seed gave target recall 0.424 ... 0.974. Any number
quoted from a single run is uninterpretable, so every claim in
`GPU_SERVER_PLAN.md` and the README's arm A table comes from one of these.

| script | question it answers |
|---|---|
| `summarize_seeds.py` | mean +- sd of the headline metrics over seeds of one config |
| `summarize_arms.py` | F1 and balanced accuracy per arm -- used to separate "worse discrimination" from "more conservative threshold" when recall AND false alarms move together |
| `per_unit.py` | per-bearing-unit recall/false-alarm spread, plus per-recording detection. The deployment-relevant error bar: a deployed system sees one machine, not eight seeds |
| `ensemble.py` | majority vote over seeds -- the only label-free lever available, since target recall cannot be selected for without target fault labels |

Run them from the repo root with the outputs present, e.g.

```bash
python scripts/analysis/summarize_arms.py
python scripts/analysis/per_unit.py "outputs/da_reports_armA_time_s*/target_test_predictions.csv"
```
