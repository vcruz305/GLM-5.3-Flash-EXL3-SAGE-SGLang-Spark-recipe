#!/usr/bin/env bash
# Host memory guard. GB10 shares one 128 GB pool between CPU and GPU; when it runs dry the whole
# box stalls. Stops the container if MemAvailable falls under the floor.
# usage: guard.sh <container>   (serve.sh starts it detached; GUARD=0 disables it)
set -u
C="${1:?container}"
FLOOR_KB=$(( ${GUARD_FLOOR_GIB:-4} * 1024 * 1024 ))
LOG="${GUARD_LOG:-/tmp/$C.guard.log}"
DOCKER="${DOCKER:-docker}"
echo "$(date -Is) guard start container=$C floor=${GUARD_FLOOR_GIB:-4}GiB" >> "$LOG"
min=999999999
while $DOCKER ps -q -f "name=^${C}$" | grep -q .; do
  avail=$(awk '/MemAvailable/ {print $2}' /proc/meminfo)
  (( avail < min )) && min=$avail
  if (( avail < FLOOR_KB )); then
    echo "$(date -Is) MemAvailable ${avail} kB < floor; stopping $C" >> "$LOG"
    $DOCKER stop -t 10 "$C" >> "$LOG" 2>&1
    break
  fi
  sleep 2
done
echo "$(date -Is) guard end; min MemAvailable ${min} kB" >> "$LOG"
