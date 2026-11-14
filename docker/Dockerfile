# GLM-5.3-Flash EXL3 on SGLang 0.5.22 for one DGX Spark (GB10, sm_121, arm64).
# Build with scripts/build_image.sh, which passes the pins from scripts/env.sh.
ARG BASE_IMAGE
FROM ${BASE_IMAGE}

ARG EXLLAMAV3_REPO
ARG EXLLAMAV3_REF
ARG SGLANG_EXL3_REPO
ARG SGLANG_EXL3_REF
ARG MAX_JOBS=3

# 1. exllamav3 runtime, built for GB10 against the image's torch.
#    MAX_JOBS=3 keeps the nvcc peak inside a 128 GB unified-memory box.
RUN set -eux; \
    git clone --filter=blob:none "${EXLLAMAV3_REPO}" /opt/src/exllamav3; \
    cd /opt/src/exllamav3; git checkout --detach "${EXLLAMAV3_REF}"; \
    TORCH_CUDA_ARCH_LIST=12.1a MAX_JOBS=${MAX_JOBS} CUDA_HOME=/usr/local/cuda \
      python3 -m pip install --no-build-isolation --no-deps .; \
    cd /; python3 -c "import exllamav3, exllamav3_ext as e; print('exllamav3', exllamav3.__version__); assert hasattr(e, 'CoopMK'), 'CoopMK missing from exllamav3_ext'"

# 2. sglang-exl3 plugin, pure Python on PYTHONPATH (the measured runs did not build its own
#    sglang_exl3_c extension). The dist-info only registers the SGLang plugin entry point.
RUN set -eux; \
    git clone --filter=blob:none "${SGLANG_EXL3_REPO}" /opt/sglang-exl3; \
    cd /opt/sglang-exl3; git checkout --detach "${SGLANG_EXL3_REF}"; rm -rf .git; \
    mkdir -p /opt/sglang-exl3-dist/sglang_exl3-0.2.1.dist-info; \
    printf 'Metadata-Version: 2.1\nName: sglang-exl3\nVersion: 0.2.1\n' > /opt/sglang-exl3-dist/sglang_exl3-0.2.1.dist-info/METADATA; \
    printf '[sglang.srt.plugins]\nexl3 = sglang_exl3:register\n' > /opt/sglang-exl3-dist/sglang_exl3-0.2.1.dist-info/entry_points.txt

# 3. SGLang v0.5.22 patches for GLM-5.3-Flash EXL3 + DFlash2 on GB10 (see docker/patches/README.md).
COPY patches/sglang-v0.5.22/ /opt/patches/sglang-v0.5.22/
RUN set -eux; cd /sgl-workspace/sglang; \
    for p in /opt/patches/sglang-v0.5.22/*.patch; do git apply -p1 --verbose "$p"; done

# 4. Fail-closed flashinfer stub: SGLang imports flashinfer at module level; every call raises,
#    and the probed optional names are absent so SGLang picks its torch/triton fallbacks.
COPY flashinfer-stub/ /opt/flashinfer-stub/

ENV PYTHONPATH=/opt/flashinfer-stub:/opt/sglang-exl3/src:/opt/sglang-exl3-dist \
    SGLANG_PLUGINS=exl3 \
    SGLANG_EXL3_REGISTER=1 \
    SGLANG_IS_FLASHINFER_AVAILABLE=false

LABEL org.opencontainers.image.title="glm53-flash-exl3-sglang" \
      org.opencontainers.image.base.name="${BASE_IMAGE}" \
      io.github.vcruz305.exllamav3.ref="${EXLLAMAV3_REF}" \
      io.github.vcruz305.sglang-exl3.ref="${SGLANG_EXL3_REF}"
