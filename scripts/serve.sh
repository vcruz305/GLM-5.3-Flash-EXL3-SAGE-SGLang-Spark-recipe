#!/usr/bin/env bash
# Start GLM-5.3-Flash EXL3 + DFlash2 on one DGX Spark with the measured flag set.
#
# usage: bash scripts/serve.sh                     # PROFILE=single (default): the measured fast path
#        PROFILE=concurrent bash scripts/serve.sh  # 4 running requests (see README "Concurrency")
#        DRY_RUN=1 bash scripts/serve.sh           # print the docker command and exit
#
# Speculative decoding (DFlash2, 4 draft tokens) is on in every profile.
# The server is ready when the log prints "The server is fired up and ready to roll!".
# Send one short warm-up request after that: the first requests after a cold start are slow
# (see README "Known limits").
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=env.sh
. "$HERE/env.sh"
export DOCKER="${DOCKER:-docker}"   # set DOCKER="sudo docker" if your user is not in the docker group
PROFILE="${PROFILE:-single}"

# ---- measured kernel switches (exllamav3 / sglang-exl3) ----
: "${SGLANG_EXL3_MOE_COOPMK:=1}"    # decode: routed experts through the CoopMK cooperative kernel (<= 8 rows)
: "${EXL3_INT8_GEMV:=2}"            # INT8 GEMV for single-row dense EXL3 linears ...
: "${EXL3_INT8_GEMV_MAX_K:=2}"      # ... only on 2-bit (K2) tensors
: "${SGLANG_EXL3_MGEMM:=0}"         # off: measured 3-5 % slower on this pack
: "${SGLANG_EXL3_MOE_KERNEL:=exllamav3}"
: "${EXL3_GEMV:=1}"

case "$PROFILE" in
  single)
    : "${EXL3_FUSED_MOE:=1}"        # prefill: routed experts through the fused exl3_moe kernel
    : "${CHUNK:=4096}"
    : "${MAX_RUNNING:=1}"
    : "${MEM_FRAC:=0.84}"
    : "${MAMBA_CACHE:=8}"
    : "${CG_BS:=1}"
    ;;
  concurrent)
    : "${EXL3_FUSED_MOE:=0}"        # as measured (see README); 1 is unmeasured at 4 running requests
    : "${CHUNK:=256}"
    : "${MAX_RUNNING:=4}"
    : "${MEM_FRAC:=0.90}"
    : "${MAMBA_CACHE:=20}"
    : "${CG_BS:=1 2 4}"
    ;;
  *) echo "unknown PROFILE=$PROFILE (single|concurrent)" >&2; exit 2 ;;
esac
: "${NUM_DRAFT:=4}"
: "${CONTEXT:=8192}"
read -r -a CG_LIST <<<"$CG_BS"
CG_MAX="${CG_LIST[-1]}"

for d in "$TARGET_VIEW" "$TARGET_DIR" "$DRAFT_DIR"; do
  [[ -d "$d" || "${DRY_RUN:-0}" == 1 ]] || { echo "missing $d: run scripts/download_and_verify.sh" >&2; exit 2; }
done
# shellcheck disable=SC2206  # DOCKER may be "sudo docker"
CMD=($DOCKER run -d --name "$CONTAINER" --gpus all --ipc host --oom-score-adj 1000
  -p "127.0.0.1:$PORT:30000"
  -v "$TARGET_VIEW:/models/target:ro" -v "$TARGET_DIR:$TARGET_DIR:ro" -v "$DRAFT_DIR:/models/draft:ro"
  -v "$TUNE_CACHE:/tune_cache:rw" -e EXLLAMAV3_TUNE_CACHE=/tune_cache
  -e SGLANG_EXL3_MOE_COOPMK="$SGLANG_EXL3_MOE_COOPMK" -e EXL3_FUSED_MOE="$EXL3_FUSED_MOE"
  -e EXL3_INT8_GEMV="$EXL3_INT8_GEMV" -e EXL3_INT8_GEMV_MAX_K="$EXL3_INT8_GEMV_MAX_K"
  -e SGLANG_EXL3_MGEMM="$SGLANG_EXL3_MGEMM" -e SGLANG_EXL3_MOE_KERNEL="$SGLANG_EXL3_MOE_KERNEL"
  -e EXL3_GEMV="$EXL3_GEMV" -e SGLANG_EXL3_MTP_FP8=0
  -e CAMPAIGN_ALLOW_CUDA_TRITON_DSA=1 -e DF1_SKIP_DFLASH_SAMPLING_PREWARM=1
  -e OMP_NUM_THREADS=4
  --entrypoint python3 "$IMAGE" -m sglang.launch_server
  --model-path /models/target --served-model-name glm-5.3-flash --host 0.0.0.0 --port 30000
  --trust-remote-code --language-only --load-format exl3_mmap --quantization exl3 --dtype bfloat16
  --tp-size 1 --context-length "$CONTEXT" --chunked-prefill-size "$CHUNK" --max-running-requests "$MAX_RUNNING"
  --mem-fraction-static "$MEM_FRAC" --max-mamba-cache-size "$MAMBA_CACHE" --mamba-ssm-dtype bfloat16
  --attention-backend dsa --kv-cache-dtype bfloat16 --dsa-prefill-backend triton --dsa-decode-backend triton
  --sampling-backend pytorch --disable-flashinfer-autotune --disable-shared-experts-fusion
  --cuda-graph-max-bs-decode "$CG_MAX" --cuda-graph-bs-decode "${CG_LIST[@]}" --disable-prefill-cuda-graph
  --skip-server-warmup --enable-metrics
  --speculative-algorithm DFLASH --speculative-draft-model-path /models/draft
  --speculative-num-draft-tokens "$NUM_DRAFT"
  --speculative-draft-model-quantization unquant --speculative-draft-attention-backend triton)

echo "PROFILE=$PROFILE IMAGE=$IMAGE CHUNK=$CHUNK MAX_RUNNING=$MAX_RUNNING MEM_FRAC=$MEM_FRAC MAMBA_CACHE=$MAMBA_CACHE CG_BS='$CG_BS' NUM_DRAFT=$NUM_DRAFT COOPMK=$SGLANG_EXL3_MOE_COOPMK FUSED=$EXL3_FUSED_MOE INT8=$EXL3_INT8_GEMV/K$EXL3_INT8_GEMV_MAX_K"
if [[ "${DRY_RUN:-0}" == 1 ]]; then printf '%q ' "${CMD[@]}"; echo; exit 0; fi
mkdir -p "$TUNE_CACHE"

if $DOCKER ps -a --format '{{.Names}}' | grep -qx "$CONTAINER"; then
  echo "container $CONTAINER already exists; stop it first: docker rm -f $CONTAINER" >&2; exit 2
fi
"${CMD[@]}"
if [[ "${GUARD:-1}" == 1 ]]; then
  setsid nohup bash "$HERE/guard.sh" "$CONTAINER" >/dev/null 2>&1 < /dev/null &
  echo "memory guard started (stops $CONTAINER if MemAvailable < ${GUARD_FLOOR_GIB:-4} GiB)"
fi
echo "started $CONTAINER; follow: docker logs -f $CONTAINER   (ready after about 5 min)"
