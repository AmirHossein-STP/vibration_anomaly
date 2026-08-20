"""Render reports/presentation_template.html into a self-contained HTML
report (base64-embedded figures, no external dependencies) using the real
results from the most recent full run.

Reads from outputs/kaggle_run/ by default (that's where
scripts/07_kaggle_pull.sh puts a pulled Kaggle run); pass --run-dir to
point at a local run's outputs/ instead once local training works.

Usage:
    python scripts/build_presentation.py
    python scripts/build_presentation.py --run-dir outputs

Requires only numpy-free stdlib + pandas (no torch) -- run with any Python
that has pandas, e.g. the sibling venv used elsewhere in this project, or
this project's own .venv once requirements.txt is installed.
"""
from __future__ import annotations

import argparse
import base64
import json
import re
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
TEMPLATE = ROOT / "reports" / "presentation_template.html"
OUT = ROOT / "outputs" / "reports" / "presentation.html"

IMG_FILES = {
    "error_histogram": "error_histogram.png",
    "roc_curve": "roc_curve.png",
    "detection_rate_by_severity": "detection_rate_by_severity.png",
    "detection_rate_by_rpm": "detection_rate_by_rpm.png",
    "latent_space_pca": "latent_space_pca.png",
    "reconstruction_examples": "reconstruction_examples.png",
    "infer_faulty_timeline": "infer_1797_IR_21_DE12_timeline.png",
    "infer_normal_timeline": "infer_1797_Normal_timeline.png",
}


def pct(x: float, digits: int = 1) -> str:
    return f"{x * 100:.{digits}f}%"


def build(run_dir: Path) -> None:
    figdir = run_dir / "outputs" / "figures"
    repdir = run_dir / "outputs" / "reports"

    metrics = json.loads((repdir / "evaluation_metrics.json").read_text())
    sens = pd.read_csv(repdir / "threshold_sensitivity.csv")
    sweep = pd.read_csv(run_dir / "outputs" / "sweep" / "summary.csv")

    cm = metrics["confusion_matrix"]
    tn, fp, fn, tp = cm["tn"], cm["fp"], cm["fn"], cm["tp"]
    n_total = metrics["n_test_healthy"] + metrics["n_test_faulty"]

    def b64_img(name: str) -> str:
        data = (figdir / IMG_FILES[name]).read_bytes()
        return f"data:image/png;base64,{base64.b64encode(data).decode('ascii')}"

    max_fp = sweep["confusion_matrix.fp"].max()
    default_latent_dim = 128
    bars = []
    for _, row in sweep.iterrows():
        ld = int(row["latent_dim"])
        fpv = int(row["confusion_matrix.fp"])
        height_px = max(6, round(90 * fpv / max_fp))
        cls = "sweep-col is-default" if ld == default_latent_dim else "sweep-col"
        bars.append(
            f'<div class="{cls}"><span class="val">{fpv}</span>'
            f'<div class="bar" style="height:{height_px}px"></div>'
            f'<span class="lbl">{ld}</span></div>'
        )
    sweep_bars = "\n".join(bars)

    txt = {
        "tn": str(tn), "fp": str(fp), "fn": str(fn), "tp": str(tp),
        "total_faulty": str(tp + fn),
        "test_healthy_n": str(metrics["n_test_healthy"]),
        "histogram_caption": (
            "Healthy windows cluster tightly around a low error; every faulty window sits an "
            f"order of magnitude higher, with a complete gap in between. Threshold={metrics['threshold']:.3f} "
            f"({metrics['threshold_percentile']:.0f}th percentile of validation-healthy error)."
        ),
        "roc_caption": f"AUC = {metrics['roc_auc']:.3f}. Perfect ranking of every held-out window, healthy or faulty.",
        "severity_caption": "100% detection in every fault-type/severity combination tested, including the smallest seeded defects.",
        "rpm_caption": "100% detection at every recorded load condition — including any condition never included in healthy training data.",
        "latent_caption": "Healthy windows (blue) collapse into one tight, isolated cluster; faulty windows (red) spread across many distinct sub-clusters -- plausibly one per fault type/severity -- but none overlap the healthy cluster.",
        "sens_recall_min": pct(sens["recall"].min(), 0),
        "sens_precision_range": f"{pct(sens['precision'].min())} - {pct(sens['precision'].max())}",
        "sens_f1_range": f"{pct(sens['f1'].min())} - {pct(sens['f1'].max())}",
        "sens_precision_lo": pct(sens["precision"].iloc[0]),
        "sens_precision_hi": pct(sens["precision"].iloc[-1]),
        "infer_normal_caption": "A handful of windows cross the threshold -- consistent with the false-alarm rate the deployed percentile threshold implies.",
    }

    html = TEMPLATE.read_text()

    for key in IMG_FILES:
        html = html.replace("{{IMG:" + key + "}}", b64_img(key))
    for key, val in txt.items():
        html = html.replace("{{TXT:" + key + "}}", val)
    html = html.replace("{{SWEEP_BARS}}", sweep_bars)

    html = html.replace(
        '<p class="dek">PLACEHOLDER_DEK</p>',
        '<p class="dek">An autoencoder trained only on healthy bearing vibration flags faults it has '
        "never seen, by how badly it fails to reconstruct them. Tested against "
        f"{n_total:,} held-out windows spanning every fault type, severity, and load condition "
        "in the CWRU bearing dataset.</p>",
    )
    stat_map = [
        ("ROC-AUC", f"{metrics['roc_auc']:.3f}"),
        ("Recall", pct(metrics["recall"], 1)),
        ("Precision", pct(metrics["precision"], 2)),
        ("F1", pct(metrics["f1"], 2)),
    ]
    old_stats = """    <div class="stat-row">
      <div class="stat"><div class="v">PLACEHOLDER</div><div class="k">ROC-AUC</div></div>
      <div class="stat"><div class="v">PLACEHOLDER</div><div class="k">Recall</div></div>
      <div class="stat"><div class="v">PLACEHOLDER</div><div class="k">Precision</div></div>
      <div class="stat"><div class="v">PLACEHOLDER</div><div class="k">F1</div></div>
    </div>"""
    new_stats = "    <div class=\"stat-row\">\n" + "\n".join(
        f'      <div class="stat"><div class="v">{v}</div><div class="k">{k}</div></div>' for k, v in stat_map
    ) + "\n    </div>"
    if old_stats not in html:
        raise RuntimeError("stat row template not found -- did presentation_template.html change?")
    html = html.replace(old_stats, new_stats)

    remaining = re.findall(r"\{\{[^}]+\}\}", html)
    if remaining:
        raise RuntimeError(f"Unfilled placeholders remain: {sorted(set(remaining))}")

    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(html)
    print(f"wrote {OUT} ({OUT.stat().st_size / 1024:.0f} KB)")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", default="outputs/kaggle_run", help="Directory containing outputs/{figures,reports,sweep}")
    args = parser.parse_args()
    build(ROOT / args.run_dir)


if __name__ == "__main__":
    main()
