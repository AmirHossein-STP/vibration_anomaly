#!/usr/bin/env bash
# CURRENTLY BLOCKED, kept for reference -- see notes/07_dvc_and_kaggle.md.
# Tried this as a way to avoid needing kernel Internet access at all (Kaggle
# input datasets are readable even in a no-internet kernel). Uploading from
# THIS machine fails with a Google Cloud Storage 403 "not available in your
# location" on every file, regardless of size/type -- reproduced down to a
# 24-byte text file, so it's not about the CWRU data. Looks tied to this
# machine's outbound network path (public IP geolocates outside the user's
# actual location; several VPN client configs are installed). Would need
# that resolved, or an upload attempt from a different network/browser
# session, before this script is usable. kaggle_train.ipynb currently
# downloads from GitHub inside the kernel instead (needs Internet ON +
# phone verification on the Kaggle account -- also documented there).
#
# Upload data/raw/ (the 64 CWRU .npz files + MANIFEST.txt) as a private
# Kaggle Dataset, so the kaggle_train.ipynb kernel can read it from
# /kaggle/input/ instead of downloading from GitHub -- Kaggle input
# datasets work even when a kernel has no internet access (unlike a
# download inside the kernel), which matters on an account without phone
# verification (see notes/07_dvc_and_kaggle.md).
#
# First run uses `datasets create`; if the dataset already exists, re-run
# with VERSION_MSG set to push an updated version instead, e.g.:
#   VERSION_MSG="refreshed data" bash scripts/08_kaggle_dataset_push.sh
set -euo pipefail
DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
KAGGLE="$DIR/.venv/bin/kaggle"

STAGE="$(mktemp -d)"
trap 'rm -rf "$STAGE"' EXIT

cp "$DIR"/data/raw/*.npz "$STAGE/"
cp "$DIR/data/raw/MANIFEST.txt" "$STAGE/"
cp "$DIR/kaggle_dataset/dataset-metadata.json" "$STAGE/"

if [ -n "${VERSION_MSG:-}" ]; then
  "$KAGGLE" datasets version -p "$STAGE" -m "$VERSION_MSG" -r zip
else
  "$KAGGLE" datasets create -p "$STAGE" -r zip
fi
