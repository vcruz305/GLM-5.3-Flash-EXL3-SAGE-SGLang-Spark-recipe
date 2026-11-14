#!/usr/bin/env bash
# Download the target pack and the DFlash2 draft at their pinned revisions, verify every file
# against SHA256SUMS, then build the SGLang config view (scripts/make_view.py).
# usage: bash scripts/download_and_verify.sh
# Needs: the `hf` CLI (pip install -U "huggingface_hub[cli]" in a venv or with pipx) and about 101 GB free.
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$HERE/.."
# shellcheck source=env.sh
. "$HERE/env.sh"
command -v hf >/dev/null || { echo "missing 'hf' CLI: pipx install 'huggingface_hub[cli]'" >&2; exit 2; }
mkdir -p "$MODELS_DIR"

need_gb=105
free_gb=$(df -BG --output=avail "$MODELS_DIR" | tail -1 | tr -dc 0-9)
if (( free_gb < need_gb )); then
  echo "WARNING: $MODELS_DIR has ${free_gb} GB free; the pack + draft need about 101 GB (+ margin)." >&2
fi

hf download "$MODEL_REPO" --revision "$MODEL_REV" --local-dir "$TARGET_DIR"
hf download "$DRAFT_REPO" --revision "$DRAFT_REV" --local-dir "$DRAFT_DIR"

verify() {  # $1 = prefix in SHA256SUMS, $2 = dir
  local prefix="$1" dir="$2"
  ( cd "$dir" && grep "  $prefix/" "$ROOT/SHA256SUMS" | sed "s#  $prefix/#  #" | sha256sum -c --quiet ) \
    && echo "OK: $dir matches SHA256SUMS ($prefix)"
}
verify target "$TARGET_DIR"
verify draft "$DRAFT_DIR"

python3 "$HERE/make_view.py" "$TARGET_DIR" "$TARGET_VIEW"
echo "done. target view: $TARGET_VIEW  draft: $DRAFT_DIR"
