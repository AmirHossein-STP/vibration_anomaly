"""Does averaging over seeds fix the variance? Needs no fault labels -- only
the models' own predictions -- so it is legal under the zero-shot rules."""
import pandas as pd, numpy as np, pathlib, sys
sys.path.insert(0, "src")
from domain_adaptation.analysis import binary_detection_metrics

preds, truth = [], None
for seed in (1, 2, 3, 4, 5):
    p = pathlib.Path(f"outputs/da_reports_seed{seed}/target_test_predictions.csv")
    df = pd.read_csv(p)
    if truth is None:
        truth = df.universal_label.to_numpy()
    else:
        assert np.array_equal(truth, df.universal_label.to_numpy()), "row order differs between seeds"
    preds.append(df.baseline_pred.to_numpy())
P = np.vstack(preds)

# Majority vote over the 3 classes, ties broken toward the lower class id.
vote = np.array([np.bincount(P[:, i], minlength=3).argmax() for i in range(P.shape[1])])

print(f"{'model':<28}{'recall':>9}{'precision':>11}{'false alarm':>13}")
indiv = []
for k, seed in enumerate((1, 2, 3, 4, 5)):
    b = binary_detection_metrics(truth, P[k])
    indiv.append(b)
    print(f"{'seed ' + str(seed):<28}{b['recall']:>9.3f}{b['precision']:>11.3f}{b['false_alarm_rate']:>12.1%}")
b = binary_detection_metrics(truth, vote)
print(f"{'-'*61}")
print(f"{'ENSEMBLE (majority of 5)':<28}{b['recall']:>9.3f}{b['precision']:>11.3f}{b['false_alarm_rate']:>12.1%}")
print(f"{'mean of individuals':<28}{np.mean([x['recall'] for x in indiv]):>9.3f}"
      f"{np.mean([x['precision'] for x in indiv]):>11.3f}"
      f"{np.mean([x['false_alarm_rate'] for x in indiv]):>12.1%}")
print(f"{'best individual':<28}{max(x['recall'] for x in indiv):>9.3f}")
print(f"{'worst individual':<28}{min(x['recall'] for x in indiv):>9.3f}")

# How often do the 5 models actually agree? Disagreement is the variance made visible.
agree = (P == P[0]).all(axis=0).mean()
print(f"\nall 5 models agree on {agree:.1%} of windows")
print(f"unanimous-faulty windows: {((P != 0).all(axis=0)).mean():.1%} | unanimous-healthy: {((P == 0).all(axis=0)).mean():.1%}")
