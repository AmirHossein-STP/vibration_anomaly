"""Build the Kaggle notebook that condenses Paderborn into a ~1 GB bundle.

Runs where the 9.7 GB dataset already sits and compute is free, so the GPU
server -- which bills by the minute -- only ever downloads the small result.
See src/domain_adaptation/data/prepare_pu.py for what is kept and why.

Output lands in /kaggle/working/prepared_pu/ as one .npz per bearing code
plus a manifest, deliberately split per code rather than one big archive:
`kaggle kernels output` has stalled on large single files in this project
before, and per-code granularity makes a stalled transfer resumable.

Stdlib only: `python3 scripts/build_kaggle_notebook_prepare_pu.py`
"""
import base64
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "notebooks" / "prepare_pu" / "kaggle_prepare_pu.ipynb"

SRC_FILES = [
    "src/domain_adaptation/__init__.py",
    "src/domain_adaptation/data/__init__.py",
    "src/domain_adaptation/data/pu_bearing_codes.py",
    "src/domain_adaptation/data/pu_loader.py",
    "src/domain_adaptation/data/transforms.py",
    "src/domain_adaptation/data/prepare_pu.py",
]


def md(*lines):
    return {"cell_type": "markdown", "metadata": {}, "source": [l + "\n" for l in lines[:-1]] + ([lines[-1]] if lines else [])}


def code(*lines):
    return {"cell_type": "code", "execution_count": None, "metadata": {}, "outputs": [],
            "source": [l + "\n" for l in lines[:-1]] + ([lines[-1]] if lines else [])}


def encode_files(relpaths):
    return "\n".join(
        f"    {rel!r}: (\n        {base64.b64encode((ROOT / rel).read_bytes()).decode('ascii')!r}\n    )," for rel in relpaths
    )


def build_notebook() -> dict:
    cells = [
        md(
            "# Condense Paderborn for transfer to the GPU server",
            "",
            "The raw dataset is 9.7 GB of which this project uses one channel "
            "(`vibration_1`) out of seven, at a rate it immediately downsamples. "
            "This keeps only that channel, resampled 64 kHz -> 25.6 kHz as "
            "float32, for the WHOLE dataset (every bearing code and operating "
            "condition, not just what today's config reads -- the marginal cost "
            "is small and it means no future config change forces another "
            "download at server rates).",
            "",
            "Result is ~1 GB instead of 9.7 GB. It also removes ~72 s of "
            "`scipy.io.loadmat` from every training run, which matters once a "
            "hyperparameter sweep repeats it dozens of times.",
        ),
        md("## 1. Reconstruct the loader code"),
        code(
            "import base64, pathlib",
            "WORKDIR = pathlib.Path('/kaggle/working')",
            "SRC_FILES = {", encode_files(SRC_FILES), "}",
            "for relpath, b64 in SRC_FILES.items():",
            "    p = WORKDIR / relpath",
            "    p.parent.mkdir(parents=True, exist_ok=True)",
            "    p.write_bytes(base64.b64decode(b64))",
            "print(f'wrote {len(SRC_FILES)} source files')",
        ),
        md("## 2. Locate the attached dataset"),
        code(
            "import sys, os",
            "sys.path.insert(0, str(WORKDIR / 'src'))",
            "os.environ['PYTHONPATH'] = str(WORKDIR / 'src')",
            "os.chdir(WORKDIR)",
            "from domain_adaptation.data.pu_loader import find_pu_root",
            "PU_ROOT = find_pu_root()",
            "codes = sorted(d.name for d in PU_ROOT.iterdir() if d.is_dir())",
            "print(f'PU_ROOT = {PU_ROOT}')",
            "print(f'{len(codes)} bearing codes: {codes}')",
        ),
        md(
            "## 3. Condense",
            "Every code, every condition, every run. Unreadable source files "
            "are named and skipped rather than aborting -- this mirror has at "
            "least one (`KA08/N15_M01_F10_KA08_2.mat`), and it previously "
            "killed a whole run.",
        ),
        code(
            "import subprocess, sys",
            "subprocess.run([sys.executable, '-m', 'domain_adaptation.data.prepare_pu',",
            "                '--pu-root', str(PU_ROOT), '--out', 'prepared_pu'], check=True)",
        ),
        md("## 4. Confirm what will be transferred"),
        code(
            "import pandas as pd",
            "out = WORKDIR / 'prepared_pu'",
            "files = sorted(out.glob('*.npz'))",
            "total_mb = sum(f.stat().st_size for f in files) / 1024**2",
            "manifest = pd.read_csv(out / 'manifest.csv')",
            "print(f'{len(files)} archives, {total_mb:.0f} MB total '",
            "      f'(raw dataset is ~9,700 MB -> {9700/max(total_mb,1):.0f}x smaller)')",
            "print(f'{len(manifest)} runs, {manifest.code.nunique()} codes, '",
            "      f'{manifest.condition.nunique()} conditions, {manifest.fs_hz.iloc[0]} Hz')",
            "print()",
            "print(manifest.groupby(['damage_mode','fault_type']).size().to_string())",
        ),
    ]
    return {
        "cells": cells,
        "metadata": {"kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
                     "language_info": {"name": "python", "version": "3.10"}},
        "nbformat": 4, "nbformat_minor": 5,
    }


if __name__ == "__main__":
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(build_notebook(), indent=1))
    print(f"wrote {OUT} ({OUT.stat().st_size} bytes)")
