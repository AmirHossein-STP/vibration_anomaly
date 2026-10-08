#!/usr/bin/env bash
# Build the three images, tag them with the git commit, smoke-test each (init + doctor), push.
# (The pytest suite is NOT run here: run `pytest tests` first, with torch installed, see DEPLOYMENT_PLAN.md section 8.)
#   REGISTRY=registry.example.com/team deploy/release.sh          # build + test + push
#   deploy/release.sh --no-push                                    # build + test only
set -euo pipefail
cd "$(dirname "$0")/.."
if git rev-parse --git-dir >/dev/null 2>&1; then
  TAG="$(git rev-parse --short HEAD)"
  [ -z "$(git status --porcelain src deploy tests)" ] || { echo "uncommitted changes in src/deploy/tests: commit first (the tag must identify the code)"; exit 2; }
else
  # unpacked hand-off archive without .git: the version comes from you, e.g. TAG=0.1.0 deploy/release.sh
  TAG="${TAG:?no .git here: run as  TAG=<version> deploy/release.sh}"
fi
REGISTRY="${REGISTRY:-local}"
for svc in bearing vibration hydraulic; do
  img="$REGISTRY/ad-$svc:$TAG"
  docker build -f deploy/Dockerfile --target "$svc" --build-arg SERVICE_IMAGE_TAG="$TAG" -t "$img" .
  # smoke test the image itself: the CLI starts and its own doctor runs on an empty volume
  tmp="$(mktemp -d)"; chmod 777 "$tmp"
  docker run --rm -e SERVING_DATA_ROOT=/t/data -e SERVING_MODELS_ROOT=/t/models -v "$tmp:/t" "$img" init >/dev/null
  docker run --rm -e SERVING_DATA_ROOT=/t/data -e SERVING_MODELS_ROOT=/t/models -v "$tmp:/t" "$img" doctor
  rm -rf "$tmp"
  [ "${1:-}" = "--no-push" ] || docker push "$img"
done
echo "released tag $TAG"
