#!/usr/bin/env python3
"""Prefill probe: streamed /v1/completions, temperature 0, 16 new tokens, prompt tok/s = prompt_tokens / TTFT.

usage: prefill_probe.py warm [--port 30000]
       prefill_probe.py run <tag> [--prompts bench/prefill_prompts.json] [--out results] [--port 30000]

Each prompt starts with a unique request id, so no prefix is reused between prompts.
Stdlib only. Timeouts are long on purpose: the first requests after a cold start stall.
"""
import argparse
import json
import os
import time
import urllib.request

MAX_NEW = 16
HERE = os.path.dirname(os.path.abspath(__file__))


def stream(port, model, prompt):
    body = json.dumps({
        "model": model, "prompt": prompt, "max_tokens": MAX_NEW,
        "temperature": 0, "stream": True, "stream_options": {"include_usage": True},
        "ignore_eos": True,
    }).encode()
    req = urllib.request.Request(f"http://127.0.0.1:{port}/v1/completions", data=body,
                                 headers={"Content-Type": "application/json"})
    t_send = time.monotonic()
    t_first = t_last = None
    pieces, usage = [], None
    with urllib.request.urlopen(req, timeout=1500) as r:
        for raw in r:
            line = raw.decode("utf-8", "replace").strip()
            if not line.startswith("data:"):
                continue
            d = line[5:].strip()
            if d == "[DONE]":
                break
            obj = json.loads(d)
            if obj.get("usage"):
                usage = obj["usage"]
            for ch in obj.get("choices") or []:
                if ch.get("text"):
                    now = time.monotonic()
                    t_first = t_first or now
                    t_last = now
                    pieces.append(ch["text"])
    return {"t_send": t_send, "t_first": t_first, "t_last": t_last, "text": "".join(pieces), "usage": usage}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("mode", choices=["warm", "run"])
    ap.add_argument("tag", nargs="?", default="prefill")
    ap.add_argument("--prompts", default=os.path.join(HERE, "prefill_prompts.json"))
    ap.add_argument("--out", default="results")
    ap.add_argument("--port", type=int, default=30000)
    ap.add_argument("--model", default="glm-5.3-flash")
    a = ap.parse_args()
    if a.mode == "warm":
        res = stream(a.port, a.model, "Reply with the single word OK.")
        print(f"WARM ttft={res['t_first'] - res['t_send']:.2f}s text={res['text']!r}")
        return
    os.makedirs(a.out, exist_ok=True)
    path = os.path.join(a.out, f"{a.tag}.jsonl")
    with open(path, "w") as f:
        for p in json.load(open(a.prompts)):
            res = stream(a.port, a.model, p["prompt"])
            ptoks = res["usage"]["prompt_tokens"] if res["usage"] else None
            ctoks = res["usage"]["completion_tokens"] if res["usage"] else None
            ttft = res["t_first"] - res["t_send"]
            span = res["t_last"] - res["t_first"]
            row = {"target_len": p["target_len"], "variant": p["variant"], "id": p["id"],
                   "prompt_tokens": ptoks, "completion_tokens": ctoks, "ttft_s": round(ttft, 3),
                   "prefill_tps": round(ptoks / ttft, 1) if ptoks else None,
                   "decode_tps": round((ctoks - 1) / span, 2) if ctoks and span > 0 else None,
                   "text": res["text"]}
            f.write(json.dumps(row) + "\n")
            f.flush()
            print(f"len~{p['target_len']} v{p['variant']} prompt_tokens={ptoks} ttft={ttft:.2f}s "
                  f"prefill_tps={row['prefill_tps']}", flush=True)
    print("wrote", path)


if __name__ == "__main__":
    main()
