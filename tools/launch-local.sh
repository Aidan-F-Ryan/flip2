#!/usr/bin/env bash
# Runs a flip2 command as N ranks on this host, one process each, joined over TCP through a rendezvous on a free local port.
# Usage: launch-local.sh N command [args...]
#   Each rank's output goes to rank-<r>.log in the working directory. Ranks share GPUs round-robin (FLIP2_DEVICE), so they
#   all run on one GPU if that's all there is. If any rank fails, the rest are stopped and its exit status is returned.
set -u
if [ $# -lt 2 ]; then
    echo "usage: $0 ranks command [args...]" >&2
    exit 2
fi
RANKS=$1
shift
GPUS=$(nvidia-smi -L 2>/dev/null | wc -l)
[ "$GPUS" -gt 0 ] || GPUS=1
PORT=$(python3 -c 'import socket; s = socket.socket(); s.bind(("", 0)); print(s.getsockname()[1])')

pids=()
for rank in $(seq 0 $((RANKS - 1))); do
    FLIP2_WORLD_SIZE=$RANKS FLIP2_RANK=$rank FLIP2_RENDEZVOUS=127.0.0.1:$PORT FLIP2_DEVICE=$((rank % GPUS)) "$@" > rank-$rank.log 2>&1 &
    pids+=($!)
done
trap 'kill "${pids[@]}" 2>/dev/null' INT TERM

status=0
remaining=$RANKS
while [ $remaining -gt 0 ]; do
    for rank in "${!pids[@]}"; do
        pid=${pids[$rank]}
        [ -n "$pid" ] || continue
        kill -0 "$pid" 2>/dev/null && continue
        wait "$pid"
        code=$?
        pids[$rank]=""
        remaining=$((remaining - 1))
        if [ $code -ne 0 ] && [ $status -eq 0 ]; then
            status=$code
            echo "rank $rank exited with $code (see rank-$rank.log); stopping the others" >&2
            for other in "${pids[@]}"; do
                [ -n "$other" ] && kill "$other" 2>/dev/null
            done
        fi
    done
    sleep 0.2
done
exit $status
