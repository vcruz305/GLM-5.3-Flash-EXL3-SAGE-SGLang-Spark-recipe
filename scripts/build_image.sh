#!/usr/bin/env bash
# Build the serving image from the pins in scripts/env.sh.
# usage: bash scripts/build_image.sh            (about 15-25 min on a Spark; most of it is the exllamav3 nvcc build)
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=env.sh
. "$HERE/env.sh"
DOCKER="${DOCKER:-docker}"
[[ "$(uname -m)" == aarch64 ]] || echo "WARNING: this image targets arm64 (DGX Spark); host is $(uname -m)" >&2
echo "building $IMAGE"
echo "  base        $BASE_IMAGE"
echo "  exllamav3   $EXLLAMAV3_REPO @ $EXLLAMAV3_REF"
echo "  sglang-exl3 $SGLANG_EXL3_REPO @ $SGLANG_EXL3_REF"
$DOCKER build \
  --build-arg BASE_IMAGE="$BASE_IMAGE" \
  --build-arg EXLLAMAV3_REPO="$EXLLAMAV3_REPO" --build-arg EXLLAMAV3_REF="$EXLLAMAV3_REF" \
  --build-arg SGLANG_EXL3_REPO="$SGLANG_EXL3_REPO" --build-arg SGLANG_EXL3_REF="$SGLANG_EXL3_REF" \
  --build-arg MAX_JOBS="${MAX_JOBS:-3}" \
  -t "$IMAGE" "$HERE/../docker"
echo "built $IMAGE"
