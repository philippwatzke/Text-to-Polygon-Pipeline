#!/usr/bin/env bash
set -u
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"

trap 'echo "[supervisor] stopping"; exit 0' INT TERM

while true; do
    python -m ki_geodaten.worker.loop
    rc=$?
    if [ "$rc" -ne 0 ]; then
        echo "[supervisor] worker exited rc=$rc, restarting in 2s"
    else
        echo "[supervisor] worker exited cleanly, restarting in 2s"
    fi
    sleep 2
done
