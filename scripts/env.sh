# shellcheck shell=bash
# Pins for this recipe. Every ref below is an immutable commit or digest.
# Source this file; scripts read these values and accept overrides from the environment.

# SGLang base image: lmsysorg/sglang:v0.5.22, multi-arch index digest.
# The arm64 manifest under it is sha256:06441be3de19a816ddbac917535e920cbfc7fc43171abc08417394f43634ffb6
# (CUDA 13.0, torch 2.14.1+cu130, triton 3.8.0, sglang commit baa9befa).
: "${BASE_IMAGE:=lmsysorg/sglang:v0.5.22@sha256:e3a4f06001a29542fc4a8638536d9ffc5eb9c9942b08603c09b19a56eae4a1ae}"

# exllamav3 runtime (CoopMK + fused routed-expert kernels). Fork commit whose sources match the
# measured build file for file.
: "${EXLLAMAV3_REPO:=https://github.com/vcruz305/exllamav3.git}"
: "${EXLLAMAV3_REF:=475461e9a1a288e7c10eb472056239d3fb78b625}"

# SGLang EXL3 plugin (master after PR #5).
: "${SGLANG_EXL3_REPO:=https://github.com/vcruz305/sglang-exl3.git}"
: "${SGLANG_EXL3_REF:=ad9c2a06733bd5237c49f5bff1c644ac3652bb7f}"

# Image tag built by scripts/build_image.sh.
: "${IMAGE:=glm53-flash-exl3-sglang:v0.5.22-r1}"

# Target pack and DFlash2 draft (Hugging Face).
: "${MODEL_REPO:=r0b0tlab/GLM-5.3-Flash-EXL3-SAGE-2.25bpw}"
: "${MODEL_REV:=b3194b87af5665bd8b1dd6316c99a48607fc137d}"
: "${DRAFT_REPO:=Enntity/GLM-5.3-Flash-DFlash2-enntity}"
: "${DRAFT_REV:=e60c8f86bba18e47bb23927588df6ce31248b569}"

# Where models live on the host.
: "${MODELS_DIR:=$HOME/models}"
: "${TARGET_DIR:=$MODELS_DIR/GLM-5.3-Flash-EXL3-SAGE-2.25bpw}"
: "${TARGET_VIEW:=$MODELS_DIR/GLM-5.3-Flash-EXL3-SAGE-2.25bpw-sglang-view}"
: "${DRAFT_DIR:=$MODELS_DIR/GLM-5.3-Flash-DFlash2-enntity}"

# Persistent exllamav3 kernel tune cache (first requests are slow until it is filled).
: "${TUNE_CACHE:=$HOME/.cache/glm53-exl3-tune}"

: "${PORT:=30000}"
: "${CONTAINER:=glm53-flash-sglang}"
