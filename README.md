# GLM-5.3-Flash EXL3 on SGLang — one DGX Spark

Serve [`zai-org/GLM-5.3-Flash`](https://huggingface.co/zai-org/GLM-5.3-Flash) (320B total, about 18B active, MoE)
on one DGX Spark (GB10, 128 GB unified memory) with **SGLang v0.5.22**, the EXL3 pack
[`r0b0tlab/GLM-5.3-Flash-EXL3-SAGE-2.25bpw`](https://huggingface.co/r0b0tlab/GLM-5.3-Flash-EXL3-SAGE-2.25bpw)
and the DFlash2 draft [`Enntity/GLM-5.3-Flash-DFlash2-enntity`](https://huggingface.co/Enntity/GLM-5.3-Flash-DFlash2-enntity).
Speculative decoding is on in every profile.

| Sparks | Engine | Model | Status |
|---|---|---|---|
| 1 | SGLang v0.5.22 + [sglang-exl3](https://github.com/vcruz305/sglang-exl3) + [exllamav3 (fork)](https://github.com/vcruz305/exllamav3) | 2.25 bpw EXL3 (SAGE) + DFlash2 | Measured 2026-11-13. Replay through these scripts: **TODO(confirm)** |

## Measured speed (one GB10, single stream, DFlash2 with 4 draft tokens, temperature 0)

These numbers come from the campaign that produced this recipe (2026-11-13), on the same pack, draft,
SGLang source, patches, plugin code and exllamav3 sources this repo pins. They were taken before the
image was rebuilt from these scripts; reproducing them through `scripts/` is **TODO(confirm)**.
Raw receipts are in [`results/`](results/).

**Decode** — `bench/decode_probe.py`, 256 new tokens per prompt, `(n-1)/(T256-T1)`:

| Prompt | Pass 1 tok/s | Pass 2 tok/s | Accepted tokens per verify |
|---|---|---|---|
| prose | 20.59 | 20.89 | 1.75 |
| code | 31.22 | 29.12 | 2.56 |
| json | 26.18 | 25.74 | 2.15 |
| math | 36.39 | 33.50 | 2.98 |
| **mean** | **28.60** | **27.31** | |

Earlier boots of the same decode path measured 28.5–32.2 (pass 1) and 26.6–28.5 (pass 2); pass 2 is
2–5 tok/s slower than pass 1 in every boot (cause not found).

**Prefill** — `bench/prefill_probe.py`, streamed `/v1/completions`, prompt tok/s = prompt tokens / TTFT,
chunked prefill 4096. Each length has two different prompts; the first prompt at each new length is slower
(first use of that shape), so both are shown:

| Prompt tokens (2nd prompt) | Prefill tok/s | TTFT s |
|---|---|---|
| 536 | 281.7 | 1.90 |
| 1,138 | 403.6 | 2.82 |
| 2,053 | 509.6 | 4.03 |
| 4,027 | 560.4 | 7.19 |
| 7,131 | 566.5 | 12.59 |

| Prompt tokens (1st prompt) | Prefill tok/s | TTFT s |
|---|---|---|
| 1,117 | 108.2 | 10.33 |
| 2,141 | 152.4 | 14.05 |
| 4,018 | 551.1 | 7.29 |
| 7,192 | 535.0 | 13.44 |

The very first request after READY took 97 s (528 tokens); see Known limits.

**Concurrency** (`PROFILE=concurrent`, measured 2026-11-12, `bench/concurrency_probe.py`, 256 tokens per request,
steady pass):

| Running requests | Aggregate tok/s | Per-request tok/s (mean) |
|---|---|---|
| 1 | 24.6 | 25.3 |
| 2 | 34.4 | 19.5 |
| 4 | 39.0 | 11.9 |

A separate eager boot with 8 running requests measured 39.9 aggregate. The concurrency boots used an
unmerged row-chunking variant of the CoopMK dispatch; an A/B on the same boot setup without it measured
37.5 vs 38.2 tok/s at 4 requests. This repo's plugin pin does not include it.

## Quick start

```bash
git clone https://github.com/vcruz305/GLM-5.3-Flash-EXL3-SAGE-SGLang-Spark-recipe
cd GLM-5.3-Flash-EXL3-SAGE-SGLang-Spark-recipe
bash scripts/download_and_verify.sh   # ~101 GB, verified against SHA256SUMS, builds the config view
bash scripts/build_image.sh           # ~15-25 min (exllamav3 CUDA build)
bash scripts/serve.sh                 # PROFILE=single; READY after about 5 min
bash scripts/smoke.sh                 # waits for READY, warm-up request, one /v1 chat request
```

If your user is not in the `docker` group, prefix the build/serve/smoke commands with `DOCKER="sudo docker"`.

OpenAI-compatible API on `http://127.0.0.1:30000/v1`, model name `glm-5.3-flash`. Stop with
`docker rm -f glm53-flash-sglang`. Restart once after the very first boot before measuring anything.

Benchmarks:

```bash
python3 bench/prefill_probe.py warm
python3 bench/prefill_probe.py run prefill_c4096
python3 bench/decode_probe.py dec_p1 && python3 bench/decode_probe.py dec_p2
```

## Measured flag set (`PROFILE=single`)

| Setting | Value | Why |
|---|---|---|
| `--speculative-algorithm DFLASH`, `--speculative-num-draft-tokens` | 4 | 2, 3, 6 and 8 draft tokens measured equal or slower |
| `--speculative-draft-model-quantization` | `unquant` | the bf16 draft otherwise inherits `--quantization exl3` |
| `--speculative-draft-attention-backend` | `triton` | the default resolves to flashinfer, which is stubbed out |
| `SGLANG_EXL3_MOE_COOPMK` | 1 | decode batches of ≤ 8 rows run each MoE layer's routed experts as one cooperative kernel; decode rose from about 12 to about 30 tok/s with it (together with the move to the exllamav3 1.6.1 kernels) |
| `EXL3_FUSED_MOE` | 1 | prefill-sized batches run the fused `exl3_moe` kernel instead of a per-expert Python loop; prefill 414–445 → 510–567 tok/s at 2–7k tokens, decode unchanged |
| `EXL3_INT8_GEMV` / `EXL3_INT8_GEMV_MAX_K` | 2 / 2 | INT8 single-row GEMV on 2-bit tensors only; enabling it on all K measured no gain |
| `SGLANG_EXL3_MGEMM` | 0 | measured 3–5 % slower |
| decode CUDA graphs `--cuda-graph-bs-decode 1` | on | verify forward 75.8 → 69.2 ms |
| `--disable-prefill-cuda-graph` | on | the prefill graph needs two more patches and gave no net gain (padding to captured sizes; more sizes exceed memory) |
| `--chunked-prefill-size` | 4096 | equal to 2048 except at ~7k tokens, where 2048 dropped to ~350 tok/s in two boots |
| `--mem-fraction-static` / `--max-mamba-cache-size` / `--max-running-requests` | 0.84 / 8 / 1 | single-stream sizing that kept host MemAvailable ≥ 12 GB |
| `--attention-backend dsa`, `--dsa-{prefill,decode}-backend triton` | | the triton DSA kernels (enabled on CUDA by patch 0004) |
| `--kv-cache-dtype` | bfloat16 | as measured |
| `--context-length` | 8192 | the longest measured prompt is 7.2k tokens; longer contexts are untested here |

`PROFILE=concurrent` changes: `--max-running-requests 4`, `--max-mamba-cache-size 20`, `--mem-fraction-static 0.90`
(at 0.84 the KV pool was only 3,328 tokens), decode graphs for batch 1/2/4, chunked prefill 256 and
`EXL3_FUSED_MOE=0`, exactly as measured. `EXL3_FUSED_MOE=1` with chunk 4096 at 4 running requests is unmeasured.

## What the image contains

- `lmsysorg/sglang:v0.5.22` (pinned by digest; arm64, CUDA 13.0, torch 2.14.1).
- exllamav3 built from [`vcruz305/exllamav3@475461e`](https://github.com/vcruz305/exllamav3/commit/475461e9a1a288e7c10eb472056239d3fb78b625) for `sm_121a`
  (its sources match the measured build file for file).
- [`vcruz305/sglang-exl3@ad9c2a0`](https://github.com/vcruz305/sglang-exl3/commit/ad9c2a06733bd5237c49f5bff1c644ac3652bb7f) on `PYTHONPATH`.
- Eight SGLang patches, listed in [`docker/patches/README.md`](docker/patches/README.md).
- A fail-closed flashinfer stub ([`docker/flashinfer-stub/`](docker/flashinfer-stub/)): flashinfer kernels froze
  this GB10 host during bring-up, so every flashinfer call raises and SGLang takes its torch/triton fallbacks.

`scripts/download_and_verify.sh` also builds a **config view** of the pack (symlinks to every file plus a
`config.json` with the pack's `tensor_storage` manifest merged in), because the plugin reads that manifest from
`config.json`. The pack itself is never edited.

## Memory

The box has one 128 GB pool for CPU and GPU. Measured with `PROFILE=single`: about 98 GB on the GPU while
serving and a host MemAvailable minimum of 12.4–12.6 GB. `scripts/guard.sh` (started by `serve.sh`) stops the
container if MemAvailable falls below 4 GiB. Do not run a second model or a large build while serving.

## Known limits

- **Reproduction through this repo's scripts is TODO(confirm).** The measured runs used a local image built from
  the same base, with the same exllamav3 sources mounted over an older bundled exllamav3, one extra Python
  package, and the SGLang patches mounted as files instead of applied in the image. The first replay may expose
  a build gap.
- **The first requests after a cold start are slow:** 37–220 s for the first request in the measured boots, and the
  first prompt at each new length is slower than the next one. Send a warm-up request (`scripts/smoke.sh`). The
  exllamav3 tune cache persists in `~/.cache/glm53-exl3-tune` between boots.
- Pass 2 of the decode probe is 2–5 tok/s slower than pass 1 in every boot; not thermal (no throttle flags,
  60–68 °C); cause unknown.
- `PROFILE=concurrent` serves one stream at 24–25 tok/s, against 28–31 in `PROFILE=single`; cause unknown.
- Single-stream decode is close to the memory-bandwidth limit of the routed experts: a standalone sweep read
  ~209 GB/s against a 239 GB/s streaming-read ceiling on this box.
- Greedy (temperature 0) is the measured setting. The DFlash sampling prewarm is skipped
  (`DF1_SKIP_DFLASH_SAMPLING_PREWARM=1`) because it calls stubbed flashinfer sampling kernels; non-greedy
  sampling is untested.
- Text only (`--language-only`).

## Do not

- Do not remove the flashinfer stub or set `SGLANG_IS_FLASHINFER_AVAILABLE=true` on a Spark.
- Do not quote the first boot after an image build, or the first request after READY.
- Do not raise `--context-length` without testing the KV pool and memory floor at that length.

## Quants

The pack table mirrors the [HF model README](https://huggingface.co/r0b0tlab/GLM-5.3-Flash-EXL3-SAGE-2.25bpw).

| bpw | size | smallest card | top-1 vs original | mean KLD | p99 KLD | download |
|---|---|---|---|---|---|---|
| 2.25 | 98.5 GB | 1× DGX Spark (128 GB) | — | — | — | [r0b0tlab/GLM-5.3-Flash-EXL3-SAGE-2.25bpw](https://huggingface.co/r0b0tlab/GLM-5.3-Flash-EXL3-SAGE-2.25bpw) |

top-1 = next-token agreement with the original checkpoint on a held-out set; KLD = mean and 99th-percentile
KL(reference ‖ pack) on those positions. No such figures are published for this pack yet (em dash). Any result is
tied to one card, one pack and one set of server settings.

## Revisions tested

- Pack `r0b0tlab/GLM-5.3-Flash-EXL3-SAGE-2.25bpw@b3194b87af5665bd8b1dd6316c99a48607fc137d`
- Draft `Enntity/GLM-5.3-Flash-DFlash2-enntity@e60c8f86bba18e47bb23927588df6ce31248b569`
- Card: one NVIDIA DGX Spark (GB10).

## Credits

- [GLM-5.3-Flash](https://huggingface.co/zai-org/GLM-5.3-Flash) by Z.ai (MIT).
- EXL3 pack by [r0b0tlab](https://huggingface.co/r0b0tlab); DFlash2 draft by [Enntity](https://huggingface.co/Enntity).
- [SGLang](https://github.com/sgl-project/sglang) (Apache-2.0); [ExLlamaV3](https://github.com/turboderp-org/exllamav3) by turboderp (MIT).
- [sglang-exl3](https://github.com/vcruz305/sglang-exl3) plugin and the [exllamav3 fork](https://github.com/vcruz305/exllamav3) (CoopMK) by Victor Cruz.

## Repo layout

```
scripts/   env.sh (pins), download_and_verify.sh, make_view.py, build_image.sh, serve.sh, guard.sh, smoke.sh
docker/    Dockerfile, patches/sglang-v0.5.22/*.patch, flashinfer-stub/
bench/     decode_probe.py, prefill_probe.py, concurrency_probe.py, prompt sets
results/   raw measured receipts
SHA256SUMS pack + draft file hashes at the pinned revisions
```

## License

Apache-2.0 for this repository. The model, pack and draft keep their own licenses.
