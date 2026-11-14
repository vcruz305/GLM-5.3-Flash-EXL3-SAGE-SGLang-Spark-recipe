# N-concurrent greedy decode throughput probe (native /generate, stdlib only).
# usage: concurrency_probe.py <label> <prompts.json> <outdir> <N> [port]
# Runs max(4,N) requests in total as ceil(TOTAL/N) rounds of N simultaneous greedy requests (256 new tokens, ignore_eos),
# prompts cycled prose, code, json, math. Aggregate tok/s = total generated tokens / total wall time of all rounds.
# Writes <outdir>/<label>.json (per-request ids, text head, spec meta) and prints one summary line.
import json, os, sys, threading, time, urllib.request
BASE = "http://127.0.0.1:%s" % (sys.argv[5] if len(sys.argv) > 5 else "30000")
label, pj, outdir, N = sys.argv[1], sys.argv[2], sys.argv[3], int(sys.argv[4])
P = json.load(open(pj))
keys = list(P.keys())
os.makedirs(outdir, exist_ok=True)
TOTAL = max(4, N)  # N=8: one round of 8 simultaneous requests (prompts cycled)
reqs = [keys[i % len(keys)] for i in range(TOTAL)]
results = []
lock = threading.Lock()


def gen(k, slot):
    body = {"input_ids": P[k]["input_ids"],
            "sampling_params": {"temperature": 0, "max_new_tokens": 256, "ignore_eos": True},
            "return_logprob": True}
    req = urllib.request.Request(BASE + "/generate", data=json.dumps(body).encode(),
                                 headers={"Content-Type": "application/json"})
    t0 = time.time()
    r = json.load(urllib.request.urlopen(req, timeout=1200))
    dt = time.time() - t0
    mi = r["meta_info"]
    ids = [x[1] for x in mi["output_token_logprobs"]]
    rec = {"key": k, "slot": slot, "n_out": len(ids), "latency_s": round(dt, 3),
           "per_req_tps": round(len(ids) / dt, 3), "output_ids": ids, "text_head": r["text"][:160],
           "meta": {kk: vv for kk, vv in mi.items() if "spec" in kk or "accept" in kk}}
    with lock:
        results.append(rec)


t_all = time.time()
rounds = [reqs[i:i + N] for i in range(0, TOTAL, N)]
round_walls = []
for rd in rounds:
    t0 = time.time()
    th = [threading.Thread(target=gen, args=(k, j)) for j, k in enumerate(rd)]
    [t.start() for t in th]
    [t.join() for t in th]
    round_walls.append(round(time.time() - t0, 3))
wall = time.time() - t_all
tot = sum(r["n_out"] for r in results)
agg = tot / wall
per = [r["per_req_tps"] for r in results]
summary = {"label": label, "N": N, "rounds": len(rounds), "requests": len(results), "total_tokens": tot,
           "wall_s": round(wall, 3), "round_walls_s": round_walls, "aggregate_tps": round(agg, 3),
           "per_request_tps_mean": round(sum(per) / len(per), 3), "per_request_tps_min": round(min(per), 3),
           "per_request_tps_max": round(max(per), 3)}
results.sort(key=lambda r: (r["key"], r["slot"]))
json.dump({"summary": summary, "results": results}, open(os.path.join(outdir, label + ".json"), "w"), indent=1)
print("SUMMARY " + json.dumps(summary), flush=True)
for r in results:
    print("REQ %s n_out=%d per_req_tps=%.2f text_head=%r" % (r["key"], r["n_out"], r["per_req_tps"],
                                                             r["text_head"][:100]), flush=True)
print("DONE", label, flush=True)
