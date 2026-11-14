# Measured receipts

Raw files from the runs quoted in the README. Paths and timings are as recorded.

## `measured-2026-11-13/` — `PROFILE=single` flag set, chunked prefill 4096

- `boot_key_lines.txt` — server log excerpt: flashinfer stub pinned in every process, `server_args`, 42 `[COOPMK] built=True`.
- `warm.out` — warm-up request.
- `C4096_pf.jsonl`, `prefill.out` — `bench/prefill_probe.py` (earlier version of the same client), two prompts per length.
- `dec_p1.json`, `dec_p2.json`, `decode_p1.out`, `decode_p2.out` — `bench/decode_probe.py` passes 1 and 2 (earlier version of the same client), with draft acceptance from `meta_info`.

## `measured-2026-11-12-concurrency/` — `PROFILE=concurrent` flag set

- `conc_p2_n{1,2,4}.out` — `bench/concurrency_probe.py`, steady pass, 1/2/4 simultaneous requests; decode graphs for
  batch 1/2/4. This boot used an unmerged row-chunking variant of the plugin's CoopMK dispatch (see README).
