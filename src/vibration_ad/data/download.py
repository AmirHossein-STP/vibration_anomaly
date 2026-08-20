"""Download the CWRU bearing .npz files we need from the srigas/CWRU_Bearing_NumPy
GitHub mirror.

Deliberately stdlib-only (urllib), no `requests` dependency: this is the one
part of the pipeline someone might want to run before setting up the full
venv, so it shouldn't need anything beyond a base Python 3 interpreter.

The exact set of files is listed in data/raw/MANIFEST.txt (one repo-relative
path per line) rather than discovered dynamically, so a run is reproducible
even if the upstream repo changes later. Only the 12 kHz Drive-End fault
files (*_DE12.npz) and the Normal baseline files are listed -- see
notes/01_dataset_notes.md for why.

Usage:
    python -m vibration_ad.data.download --config configs/default.yaml
"""
from __future__ import annotations

import argparse
import shutil
import sys
import urllib.error
import urllib.request
from pathlib import Path

from vibration_ad.config import load_config

RAW_BASE_URL = "https://raw.githubusercontent.com/srigas/CWRU_Bearing_NumPy/main/"
REQUEST_TIMEOUT_S = 15
# If this many downloads in a row fail, stop instead of grinding through the
# rest of the manifest at the same (usually ~30s OS resolver timeout) cost
# per file -- a systemic problem (no internet, DNS broken, GitHub blocked)
# isn't going to start succeeding on file N+1 just because it failed on N.
MAX_CONSECUTIVE_FAILURES = 3


def download_manifest(manifest_path: Path, dest_dir: Path) -> list[str]:
    """Returns the list of repo-relative paths that failed to download."""
    dest_dir.mkdir(parents=True, exist_ok=True)
    lines = [
        line.strip()
        for line in manifest_path.read_text().splitlines()
        if line.strip()
    ]

    failed: list[str] = []
    consecutive_failures = 0
    for i, repo_relpath in enumerate(lines, 1):
        fname = repo_relpath.rsplit("/", 1)[-1]
        dest = dest_dir / fname
        if dest.exists():
            print(f"[{i}/{len(lines)}] skip (exists): {fname}")
            consecutive_failures = 0
            continue

        url = RAW_BASE_URL + urllib.request.quote(repo_relpath)
        print(f"[{i}/{len(lines)}] downloading: {fname}")
        tmp = dest.with_suffix(dest.suffix + ".part")
        try:
            with urllib.request.urlopen(url, timeout=REQUEST_TIMEOUT_S) as response, open(tmp, "wb") as f:
                shutil.copyfileobj(response, f)
            tmp.rename(dest)
            consecutive_failures = 0
        except (urllib.error.URLError, TimeoutError) as e:
            print(f"  FAILED: {repo_relpath} ({e})", file=sys.stderr)
            tmp.unlink(missing_ok=True)
            failed.append(repo_relpath)
            consecutive_failures += 1
            if consecutive_failures >= MAX_CONSECUTIVE_FAILURES:
                print(
                    f"Aborting after {consecutive_failures} consecutive failures "
                    "-- likely no internet access rather than a per-file issue "
                    "(on Kaggle: check the notebook's Internet toggle AND that "
                    "the account is phone-verified, which gates internet/GPU "
                    "access regardless of that toggle).",
                    file=sys.stderr,
                )
                failed.extend(lines[i:])  # remaining, un-attempted files also "failed"
                break

    return failed


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/default.yaml")
    args = parser.parse_args()

    config = load_config(args.config)
    raw_dir = config.resolve(config.paths.raw_dir)
    manifest_path = raw_dir / "MANIFEST.txt"

    if not manifest_path.exists():
        raise FileNotFoundError(
            f"Manifest not found at {manifest_path}. It should be checked "
            "into the repo alongside this script."
        )

    failed = download_manifest(manifest_path, raw_dir)
    n_files = len(list(raw_dir.glob("*.npz")))
    print(f"Done. {n_files} .npz files present in {raw_dir}")

    if failed:
        raise RuntimeError(
            f"{len(failed)} file(s) failed to download, e.g. {failed[0]!r}. "
            "Not continuing with a partial/empty dataset -- see stderr above "
            "for the underlying error."
        )


if __name__ == "__main__":
    main()
