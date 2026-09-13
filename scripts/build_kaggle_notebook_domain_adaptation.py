"""Regenerate notebooks/domain_adaptation/kaggle_train_domain_adaptation.ipynb
from the current source tree. Same snapshot-not-live-synced pattern as
build_kaggle_notebook.py / build_kaggle_notebook_hydraulic.py -- re-run +
re-push (scripts/19_domain_adaptation_kaggle_push.sh) any time
src/domain_adaptation/, configs/domain_adaptation.yaml, or the
vibration_ad modules it imports change.

Third, independent experiment in this repo -- see DOMAIN_ADAPTATION_PLAN.md.
Embeds BOTH `domain_adaptation/` (its own package) AND the specific
`vibration_ad/` modules it imports directly (windowing/STFT/normalizer --
see DOMAIN_ADAPTATION_PLAN.md's "Revisions" section for why that's reused
rather than reimplemented). Two attached Kaggle input datasets (via
kernel-metadata.json's dataset_sources), nothing downloaded/uploaded at
runtime:
  - `esraakhaled299/cwru-data` (CWRU, same dataset the first experiment
    uses) -- needs the same .mat -> .npz conversion as
    build_kaggle_notebook.py's cell 3, copied here rather than imported
    since it's notebook-cell code, not a package function.
  - `dippatel03/paderborn-db` (Paderborn/KAt-DataCenter) -- no conversion
    needed, read directly via domain_adaptation.data.pu_loader.

Stdlib only -- run with any Python 3: `python3 scripts/build_kaggle_notebook_domain_adaptation.py`
"""
import base64
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "notebooks" / "domain_adaptation" / "kaggle_train_domain_adaptation.ipynb"

VIBRATION_AD_SRC_FILES = [
    "src/vibration_ad/__init__.py",
    "src/vibration_ad/config.py",
    "src/vibration_ad/data/__init__.py",
    "src/vibration_ad/data/catalog.py",
    "src/vibration_ad/data/dataset.py",
    "src/vibration_ad/data/windows.py",
    "src/vibration_ad/features/__init__.py",
    "src/vibration_ad/features/spectrogram.py",
]
DOMAIN_ADAPTATION_SRC_FILES = [
    "src/domain_adaptation/__init__.py",
    "src/domain_adaptation/config.py",
    "src/domain_adaptation/data/__init__.py",
    "src/domain_adaptation/data/pu_bearing_codes.py",
    "src/domain_adaptation/data/cwru_loader.py",
    "src/domain_adaptation/data/pu_loader.py",
    "src/domain_adaptation/data/transforms.py",
    "src/domain_adaptation/data/dataset.py",
    "src/domain_adaptation/data/torch_dataset.py",
    "src/domain_adaptation/models/__init__.py",
    "src/domain_adaptation/models/grl.py",
    "src/domain_adaptation/models/feature_extractor.py",
    "src/domain_adaptation/models/heads.py",
    "src/domain_adaptation/models/dann.py",
    "src/domain_adaptation/train_baseline.py",
    "src/domain_adaptation/train_dann.py",
    "src/domain_adaptation/evaluate.py",
    "src/domain_adaptation/analysis.py",
    "src/domain_adaptation/viz.py",
]
SRC_FILES = VIBRATION_AD_SRC_FILES + DOMAIN_ADAPTATION_SRC_FILES
CONFIG_FILES = ["configs/domain_adaptation.yaml"]


def md(*lines):
    return {"cell_type": "markdown", "metadata": {}, "source": [l + "\n" for l in lines[:-1]] + ([lines[-1]] if lines else [])}


def code(*lines):
    return {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": [l + "\n" for l in lines[:-1]] + ([lines[-1]] if lines else []),
    }


def encode_files(relpaths: list[str]) -> str:
    entries = []
    for rel in relpaths:
        content = (ROOT / rel).read_bytes()
        b64 = base64.b64encode(content).decode("ascii")
        entries.append(f"    {rel!r}: (\n        {b64!r}\n    ),")
    return "\n".join(entries)


def build_notebook() -> dict:
    cells = []

    cells.append(md(
        "# Zero-Shot Fault Diagnosis via Domain Adaptation -- Kaggle test run (CPU, offline)",
        "",
        "Self-contained snapshot of `src/domain_adaptation/` (plus the "
        "specific `vibration_ad/` modules it imports directly -- regenerated "
        "by `scripts/build_kaggle_notebook_domain_adaptation.py`, not "
        "live-synced). Third, independent experiment in this project -- see "
        "`/DOMAIN_ADAPTATION_PLAN.md` for the full plan and its "
        "'Revisions & Decisions' section for what changed from the original "
        "proposal and why.",
        "",
        "Trains a supervised source-only baseline (Phase 2), then a DANN "
        "model (Phase 3: domain discriminator + gradient reversal + a "
        "healthy-anchor alignment loss), then evaluates both on a target "
        "operating condition's REAL bearing faults that neither model ever "
        "saw during training -- the zero-shot check. No Internet needed -- "
        "both `esraakhaled299/cwru-data` and `dippatel03/paderborn-db` are "
        "attached input datasets, readable regardless of the kernel's "
        "Internet setting (this account isn't phone-verified, see "
        "vibration_ad's `notes/07_dvc_and_kaggle.md`). CPU only.",
    ))

    cells.append(md("## 1. Reconstruct the `vibration_ad` + `domain_adaptation` packages + config on disk"))
    cells.append(code(
        "import base64, pathlib",
        "",
        "WORKDIR = pathlib.Path('/kaggle/working')",
        "SRC_FILES = {",
        encode_files(SRC_FILES),
        "}",
        "for relpath, b64 in SRC_FILES.items():",
        "    p = WORKDIR / relpath",
        "    p.parent.mkdir(parents=True, exist_ok=True)",
        "    p.write_bytes(base64.b64decode(b64))",
        "print(f'wrote {len(SRC_FILES)} source files under {WORKDIR}/src')",
    ))
    cells.append(code(
        "CONFIG_FILES = {",
        encode_files(CONFIG_FILES),
        "}",
        "for relpath, b64 in CONFIG_FILES.items():",
        "    p = WORKDIR / relpath",
        "    p.parent.mkdir(parents=True, exist_ok=True)",
        "    p.write_bytes(base64.b64decode(b64))",
        "print(f'wrote {len(CONFIG_FILES)} config files')",
    ))

    cells.append(md(
        "## 2. Environment setup + confirm PyTorch",
        "Same subprocess-PYTHONPATH pattern as the other two notebooks in "
        "this project -- `!python` on the PATH isn't guaranteed to be this "
        "kernel's interpreter.",
    ))
    cells.append(code(
        "import sys, os, subprocess",
        "sys.path.insert(0, str(WORKDIR / 'src'))",
        "os.environ['PYTHONPATH'] = str(WORKDIR / 'src')",
        "os.chdir(WORKDIR)",
        "PY = sys.executable",
        "import torch",
        "print('python:', PY)",
        "print('torch:', torch.__version__)",
        "print('cuda available:', torch.cuda.is_available(), '(CPU-only run, expected)')",
        "",
        "def run(*args):",
        "    subprocess.run([PY, *args], check=True)",
    ))

    cells.append(md(
        "## 3. Convert the CWRU dataset from the attached input (.mat -> .npz)",
        "Identical conversion to the CWRU vibration MVP notebook -- same "
        "`esraakhaled299/cwru-data` dataset, same `{rpm}_{fault}_{size}_DE12.npz` "
        "/ `{rpm}_Normal.npz` naming `vibration_ad.data.catalog` expects. See "
        "vibration_ad's `notes/07_dvc_and_kaggle.md` for why this dataset/format "
        "(not a runtime download) and how the mount path was confirmed.",
    ))
    cells.append(code(
        "import scipy.io, re",
        "import numpy as np",
        "",
        "_candidates = list(pathlib.Path('/kaggle/input').rglob('CWRU_organized'))",
        "if not _candidates:",
        "    print('CWRU_organized not found under /kaggle/input. Contents:')",
        "    for p in sorted(pathlib.Path('/kaggle/input').rglob('*')):",
        "        print(' ', p)",
        "    raise FileNotFoundError(",
        "        \"cwru-data dataset not found under /kaggle/input -- is \"",
        "        \"'esraakhaled299/cwru-data' attached to this kernel?\"",
        "    )",
        "MAT_ROOT = _candidates[0]",
        "print(f'MAT_ROOT = {MAT_ROOT}')",
        "RAW_DIR = WORKDIR / 'data' / 'raw'",
        "RAW_DIR.mkdir(parents=True, exist_ok=True)",
        "LOAD_TO_RPM = {0: 1797, 1: 1772, 2: 1750, 3: 1730}",
        "",
        "BALL_IR_RE = re.compile(r'^(B|IR)(\\d{3})_(\\d)\\.mat$')",
        "OR_RE = re.compile(r'^OR(\\d{3})@(\\d+)_(\\d)\\.mat$')",
        "NORMAL_RE = re.compile(r'^Normal_(\\d)\\.mat$')",
        "",
        "def de_signal(mat_path):",
        "    d = scipy.io.loadmat(str(mat_path))",
        "    key = next(k for k in d if k.endswith('_DE_time'))",
        "    return d[key].ravel()",
        "",
        "n = 0",
        "for mat_path in sorted((MAT_ROOT / '12k_DE').rglob('*.mat')):",
        "    fname = mat_path.name",
        "    m = BALL_IR_RE.match(fname)",
        "    if m:",
        "        fault, size, load = m.group(1), int(m.group(2)), int(m.group(3))",
        "        out_name = f'{LOAD_TO_RPM[load]}_{fault}_{size}_DE12.npz'",
        "    else:",
        "        m = OR_RE.match(fname)",
        "        if not m:",
        "            print(f'skip (unrecognized): {mat_path}')",
        "            continue",
        "        size, pos, load = int(m.group(1)), m.group(2), int(m.group(3))",
        "        out_name = f'{LOAD_TO_RPM[load]}_OR@{pos}_{size}_DE12.npz'",
        "    np.savez(RAW_DIR / out_name, DE=de_signal(mat_path))",
        "    n += 1",
        "",
        "for mat_path in sorted((MAT_ROOT / 'Normal').rglob('*.mat')):",
        "    m = NORMAL_RE.match(mat_path.name)",
        "    if not m:",
        "        print(f'skip (unrecognized): {mat_path}')",
        "        continue",
        "    rpm = LOAD_TO_RPM[int(m.group(1))]",
        "    np.savez(RAW_DIR / f'{rpm}_Normal.npz', DE=de_signal(mat_path))",
        "    n += 1",
        "",
        "print(f'converted {n} files into {RAW_DIR}')",
        "assert n == 64, f'expected 64 CWRU files, got {n}'",
    ))

    cells.append(md(
        "## 4. Locate the attached Paderborn (PU) dataset",
        "No conversion needed -- read directly via "
        "`domain_adaptation.data.pu_loader`. `find_pu_root()` searches "
        "`/kaggle/input` for a `K001/` subfolder rather than assuming a "
        "fixed mount path (same reasoning as the CWRU notebook's "
        "`CWRU_organized` search) -- see DOMAIN_ADAPTATION_PLAN.md's "
        "'Revisions' section for how this dataset's structure was verified.",
    ))
    cells.append(code(
        "from domain_adaptation.data.pu_loader import find_pu_root",
        "",
        "PU_ROOT = find_pu_root()",
        "print(f'PU_ROOT = {PU_ROOT}')",
        "print('Bearing-code folders found:', sorted(p.name for p in PU_ROOT.iterdir() if p.is_dir())[:10], '...')",
    ))

    cells.append(md(
        "## 5. Phase 2: supervised source-only baseline (no adaptation)",
        "Trains on SOURCE_TRAIN (CWRU + PU_Artificial, fully labeled), "
        "early-stops on SOURCE_VAL. Fits + saves the shared normalizer.",
    ))
    cells.append(code(
        "run('-m', 'domain_adaptation.train_baseline', '--config', 'configs/domain_adaptation.yaml', '--pu-root', str(PU_ROOT))"
    ))

    cells.append(md(
        "## 6. Phase 3: DANN adaptation",
        "Adds the domain discriminator (behind a Gradient Reversal Layer) "
        "and the healthy-anchor loss, adapting on TARGET_ADAPT -- the "
        "held-out condition's HEALTHY runs only. No fault example, labeled "
        "or not, from the target domain appears anywhere in this step.",
    ))
    cells.append(code(
        "run('-m', 'domain_adaptation.train_dann', '--config', 'configs/domain_adaptation.yaml', '--pu-root', str(PU_ROOT))"
    ))

    cells.append(md(
        "## 7. Phase 4: the zero-shot check",
        "Evaluates both models on TARGET_TEST -- the held-out condition's "
        "REAL damage bearings, whose labels neither model has seen -- and "
        "compares baseline vs. DANN. Also renders a t-SNE feature-space plot.",
    ))
    cells.append(code(
        "run('-m', 'domain_adaptation.evaluate', '--config', 'configs/domain_adaptation.yaml', '--pu-root', str(PU_ROOT))"
    ))

    cells.append(md("## 8. Results"))
    cells.append(code(
        "import json",
        "metrics = json.loads((WORKDIR / 'outputs/da_reports/evaluation_metrics.json').read_text())",
        "for name, m in metrics.items():",
        "    print(f\"{name:10s}  accuracy={m['accuracy']:.3f}  macro_f1={m['macro_f1']:.3f}  n={m['n']}\")",
        "    print('  confusion_matrix (rows=true, cols=pred, order=[healthy, inner_race, outer_race]):')",
        "    for row in m['confusion_matrix']:",
        "        print('   ', row)",
        "print()",
        "print(json.dumps(metrics, indent=2))",
    ))
    cells.append(code(
        "from IPython.display import Image, display",
        "figdir = WORKDIR / 'outputs' / 'da_figures'",
        "for name in sorted(p.name for p in figdir.glob('*.png')):",
        "    print(name)",
        "    display(Image(filename=str(figdir / name)))",
    ))

    return {
        "cells": cells,
        "metadata": {
            "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
            "language_info": {"name": "python", "version": "3.10"},
            "accelerator": "GPU",
        },
        "nbformat": 4,
        "nbformat_minor": 5,
    }


if __name__ == "__main__":
    notebook = build_notebook()
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(notebook, indent=1))
    print(f"wrote {OUT} ({OUT.stat().st_size} bytes, {len(notebook['cells'])} cells)")
