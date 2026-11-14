#!/usr/bin/env bash
# Wait for READY, send the warm-up request, then one chat request through /v1.
# usage: bash scripts/smoke.sh
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=env.sh
. "$HERE/env.sh"
DOCKER="${DOCKER:-docker}"
echo "waiting for READY (no HTTP before it)..."
for _ in $(seq 1 180); do
  if $DOCKER logs "$CONTAINER" 2>&1 | grep -q "fired up and ready to roll"; then break; fi
  $DOCKER ps -q -f "name=^${CONTAINER}$" | grep -q . || { echo "container exited"; $DOCKER logs --tail 50 "$CONTAINER"; exit 1; }
  sleep 5
done
$DOCKER logs "$CONTAINER" 2>&1 | grep -q "fired up and ready to roll" || { echo "not ready after 15 min"; exit 1; }
echo "READY. warm-up (the first request can take one to several minutes)..."
python3 "$HERE/../bench/prefill_probe.py" warm --port "$PORT"
echo "chat:"
curl -s "http://127.0.0.1:$PORT/v1/chat/completions" -H 'Content-Type: application/json' -d '{
  "model": "glm-5.3-flash", "temperature": 0, "max_tokens": 200,
  "messages": [{"role": "user", "content": "Write a Python function fib(n) that returns the nth Fibonacci number iteratively."}],
  "chat_template_kwargs": {"enable_thinking": false}}' | python3 -c 'import sys,json; d=json.load(sys.stdin); print(d["choices"][0]["message"]["content"]); print(d["usage"])'
