#!/usr/bin/env python3
"""Single-stream greedy decode probe (native /generate, stdlib only).

decode tok/s = (n - 1) / (T256 - T1), where T1 is the time for a 1-token request (prefill + first
token) and T256 the time for 256 tokens of the same prompt. ignore_eos=True, so every run emits 256
tokens. The draft acceptance fields come from the server's meta_info.

usage: decode_probe.py <label> [--prompts bench/decode_prompts.json] [--out results] [--port 30000]
"""
import argparse
import json
import os
import time
import urllib.request

HERE = os.path.dirname(os.path.abspath(__file__))
SKIP = {"output_token_logprobs", "input_token_logprobs", "output_top_logprobs", "input_top_logprobs"}


def gen(port, ids, n):
    body = {"input_ids": ids, "sampling_params": {"temperature": 0, "max_new_tokens": n, "ignore_eos": True},
            "return_logprob": True}
    req = urllib.request.Request(f"http://127.0.0.1:{port}/generate", data=json.dumps(body).encode(),
                                 headers={"Content-Type": "application/json"})
    t = time.time()
    r = json.load(urllib.request.urlopen(req, timeout=1200))
    return r, time.time() - t


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("label")
    ap.add_argument("--prompts", default=os.path.join(HERE, "decode_prompts.json"))
    ap.add_argument("--out", default="results")
    ap.add_argument("--port", type=int, default=30000)
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    prompts = json.load(open(a.prompts))
    gen(a.port, prompts["prose"]["input_ids"], 16)  # warm-up, not recorded
    res, tps_all = {}, []
    for k, v in prompts.items():
        ids = v["input_ids"]
        _, t1 = gen(a.port, ids, 1)
        r, t = gen(a.port, ids, 256)
        toks = [x[1] for x in r["meta_info"]["output_token_logprobs"]]
        n = len(toks)
        tps = (n - 1) / (t - t1) if n > 1 and t > t1 else None
        meta = {kk: vv for kk, vv in r["meta_info"].items() if kk not in SKIP}
        res[k] = {"n_out": n, "ttft_s": round(t1, 3), "t256_s": round(t, 3),
                  "decode_tps": round(tps, 2) if tps else None, "text": r["text"], "meta": meta}
        tps_all.append(tps or 0)
        print(k, "decode_tps", res[k]["decode_tps"], "accept_len", meta.get("spec_accept_length"), flush=True)
    path = os.path.join(a.out, f"{a.label}.json")
    json.dump(res, open(path, "w"), indent=1)
    print(f"mean decode tok/s {sum(tps_all) / len(tps_all):.2f}; wrote {path}")


if __name__ == "__main__":
    main()
