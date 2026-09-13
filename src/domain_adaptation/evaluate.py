"""Phase 4: the actual zero-shot check. Runs the Phase 2 baseline and (if
trained) the Phase 3 DANN model on TARGET_TEST -- the held-out operating
condition's real-damage bearings, whose fault labels neither model has
ever seen -- and compares them.

TARGET_TEST mixes held-out healthy rows (label=0) with every real-damage
bearing's rows (label=1 inner_race, 2 outer_race, see data/dataset.py) --
so this is a genuine 3-class evaluation, not just "does it flag faults."

Also produces the feature-space visualization DOMAIN_ADAPTATION_PLAN.md's
Phase 4 asks for: a 2D projection (t-SNE) of the feature extractor's
embeddings across SOURCE_VAL + TARGET_TEST, colored by domain (do the
domains overlap?) and by fault class (do target real faults land inside
the source fault clusters?) -- using the DANN feature extractor if it
exists, else the baseline's, so there's always something to look at even
before Phase 3 has been run.

Usage:
    python -m domain_adaptation.evaluate --config configs/domain_adaptation.yaml [--pu-root DIR]
"""
from __future__ import annotations

import argparse
import json

import numpy as np
import torch
from sklearn.manifold import TSNE
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score
from torch.utils.data import DataLoader

from vibration_ad.features.spectrogram import SpectrogramNormalizer, expected_shape

from domain_adaptation import viz
from domain_adaptation.analysis import binary_detection_metrics, domain_separability_probe
from domain_adaptation.config import Config, load_config
from domain_adaptation.data.dataset import build_domain_adaptation_datasets
from domain_adaptation.data.pu_loader import find_pu_root
from domain_adaptation.data.torch_dataset import LabeledDomainDataset
from domain_adaptation.models.dann import DANNModel, FeatureClassifier
from domain_adaptation.models.feature_extractor import select_device


@torch.no_grad()
def run_inference(model, X_norm: np.ndarray, domain_ids: np.ndarray, labels: np.ndarray, device, batch_size: int = 64):
    loader = DataLoader(LabeledDomainDataset(X_norm, domain_ids, labels), batch_size=batch_size, shuffle=False)
    model.eval()
    all_features, all_preds, all_labels = [], [], []
    for X, _domain_id, label in loader:
        X = X.to(device)
        outputs = model(X)  # (features, class_logits[, domain_logits])
        features, class_logits = outputs[0], outputs[1]
        all_features.append(features.cpu().numpy())
        all_preds.append(class_logits.argmax(dim=1).cpu().numpy())
        all_labels.append(label.numpy())
    return np.concatenate(all_features), np.concatenate(all_preds), np.concatenate(all_labels)


def compute_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    cm = confusion_matrix(y_true, y_pred, labels=[0, 1, 2])
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "macro_f1": float(f1_score(y_true, y_pred, average="macro", labels=[0, 1, 2], zero_division=0)),
        "confusion_matrix": cm.tolist(),
        "n": int(len(y_true)),
        # Reported alongside the 3-class figures, not instead of them: across
        # kernels v1-v5 fault TYPING never transferred while fault DETECTION
        # partly did, and detection is the question the deployment actually
        # poses. See analysis.binary_detection_metrics.
        "binary_detection": binary_detection_metrics(y_true, y_pred),
    }


def _print_summary(name: str, metrics: dict) -> None:
    b = metrics["binary_detection"]
    print(f"  {name}: 3-class accuracy={metrics['accuracy']:.3f} macro_f1={metrics['macro_f1']:.3f}")
    print(
        f"  {name}: BINARY healthy-vs-faulty -- precision={b['precision']:.3f} "
        f"recall={b['recall']:.3f} f1={b['f1']:.3f} false_alarm_rate={b['false_alarm_rate']:.1%} "
        f"(always-faulty accuracy would be {b['always_faulty_accuracy']:.3f})"
    )
    print(json.dumps(metrics, indent=2))


def evaluate(config: Config, pu_root) -> dict:
    models_dir = config.resolve(config.paths.models_dir)
    reports_dir = config.resolve(config.paths.reports_dir)
    figures_dir = config.resolve(config.paths.figures_dir)
    for d in (reports_dir, figures_dir):
        d.mkdir(parents=True, exist_ok=True)

    baseline_ckpt = models_dir / "baseline_best.pt"
    dann_ckpt = models_dir / "dann_best.pt"
    if not baseline_ckpt.exists():
        raise FileNotFoundError(f"{baseline_ckpt} not found -- run train_baseline.py first.")

    normalizer = SpectrogramNormalizer.load(models_dir / "normalizer.npz")
    expected = expected_shape(config.common, config.stft)
    device = select_device(config.train_baseline.device)

    print("Loading cached datasets...")
    datasets = build_domain_adaptation_datasets(config, pu_root)
    test_X, test_meta = datasets["target_test"]
    val_X, val_meta = datasets["source_val"]
    test_X_norm = normalizer.transform(test_X)
    val_X_norm = normalizer.transform(val_X)

    results: dict[str, dict] = {}
    feature_source = None  # which model's features feed the t-SNE plot: prefer DANN, else baseline

    print("Evaluating baseline (no adaptation) on TARGET_TEST...")
    baseline = FeatureClassifier(expected, config).to(device)
    baseline.load_state_dict(torch.load(baseline_ckpt, map_location=device))
    test_domain_ids = test_meta.domain_id.to_numpy()
    test_labels = test_meta.universal_label.to_numpy()
    baseline_features, baseline_preds, _ = run_inference(baseline, test_X_norm, test_domain_ids, test_labels, device)
    results["baseline"] = compute_metrics(test_labels, baseline_preds)
    feature_source = ("baseline", baseline)
    _print_summary("baseline", results["baseline"])

    if dann_ckpt.exists():
        print("Evaluating DANN (adapted) on TARGET_TEST...")
        dann = DANNModel(expected, config).to(device)
        dann.load_state_dict(torch.load(dann_ckpt, map_location=device))
        dann_features, dann_preds, _ = run_inference(dann, test_X_norm, test_domain_ids, test_labels, device)
        results["dann"] = compute_metrics(test_labels, dann_preds)
        feature_source = ("dann", dann)
        _print_summary("dann", results["dann"])
    else:
        print(f"{dann_ckpt} not found -- skipping DANN evaluation (run train_dann.py to add the comparison).")

    # --- save metrics + per-window predictions ---
    with open(reports_dir / "evaluation_metrics.json", "w") as f:
        json.dump(results, f, indent=2)
    test_meta = test_meta.copy()
    test_meta["baseline_pred"] = baseline_preds
    if "dann" in results:
        test_meta["dann_pred"] = dann_preds
    test_meta.to_csv(reports_dir / "target_test_predictions.csv", index=False)

    # --- figures ---
    viz.plot_confusion_matrix_multiclass(
        np.array(results["baseline"]["confusion_matrix"]),
        figures_dir / "confusion_matrix_baseline.png",
        "Baseline (no adaptation) -- TARGET_TEST",
    )
    if "dann" in results:
        viz.plot_confusion_matrix_multiclass(
            np.array(results["dann"]["confusion_matrix"]),
            figures_dir / "confusion_matrix_dann.png",
            "DANN (adapted) -- TARGET_TEST",
        )
    viz.plot_baseline_vs_dann_bars(results["baseline"], results.get("dann"), figures_dir / "baseline_vs_dann.png")

    print("Computing feature-space embeddings (SOURCE_VAL + TARGET_TEST)...")
    model_name, model = feature_source
    val_domain_ids = val_meta.domain_id.to_numpy()
    val_labels = val_meta.universal_label.to_numpy()
    rng = np.random.default_rng(config.seed)
    val_sample = rng.choice(len(val_X_norm), size=min(2000, len(val_X_norm)), replace=False)

    # --- Did adaptation actually align the domains? ---
    # Train a FRESH probe on each model's frozen features (see
    # analysis.domain_separability_probe). This is the check v1 lacked: its
    # domain_loss sat at chance the whole run, which looked like alignment
    # but -- per the t-SNE plot -- wasn't. The probe can't be fooled that
    # way, and comparing baseline vs. DANN says whether adaptation moved
    # the needle AT ALL, independent of any target label.
    for probe_name, probe_model, probe_test_features in [
        ("baseline", baseline, baseline_features),
        ("dann", dann if "dann" in results else None, dann_features if "dann" in results else None),
    ]:
        if probe_model is None:
            continue
        probe_val_features, _, _ = run_inference(
            probe_model, val_X_norm[val_sample], val_domain_ids[val_sample], val_labels[val_sample], device
        )
        probe_result = domain_separability_probe(probe_val_features, probe_test_features, seed=config.seed)
        results[probe_name]["domain_separability"] = probe_result
        print(
            f"  {probe_name}: a fresh probe separates source from target with "
            f"{probe_result['separability_accuracy']:.3f} accuracy "
            f"(0.5 = aligned/indistinguishable, 1.0 = not aligned at all; "
            f"proxy A-distance {probe_result['proxy_a_distance']:.2f})"
        )
        if probe_name == model_name:
            val_features = probe_val_features

    # Re-save metrics now that the probe results are attached.
    with open(reports_dir / "evaluation_metrics.json", "w") as f:
        json.dump(results, f, indent=2)

    print("Fitting t-SNE...")
    test_features = dann_features if model_name == "dann" else baseline_features

    combined_features = np.concatenate([val_features, test_features], axis=0)
    combined_domain_ids = np.concatenate([val_domain_ids[val_sample], test_domain_ids])
    combined_labels = np.concatenate([val_labels[val_sample], test_labels])

    tsne = TSNE(n_components=2, perplexity=config.evaluation.tsne_perplexity, random_state=config.seed, init="pca")
    coords = tsne.fit_transform(combined_features)
    viz.plot_feature_space(
        coords, combined_domain_ids, combined_labels,
        figures_dir / f"feature_space_tsne_{model_name}.png",
        method_name=f"t-SNE ({model_name} features)",
    )

    print(f"\nSaved metrics/plots to {reports_dir} and {figures_dir}")
    return results


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/domain_adaptation.yaml")
    parser.add_argument("--pu-root", default=None, help="Root dir of the PU dataset (default: auto-detect /kaggle/input, else config's pu_root_local).")
    args = parser.parse_args()

    config = load_config(args.config)
    pu_root = args.pu_root or find_pu_root(config.resolve(config.paths.pu_root_local))
    evaluate(config, pu_root)


if __name__ == "__main__":
    main()
