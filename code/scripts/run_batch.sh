#!/bin/bash
# Sequential batch runner for multiple production runs (crash-resilient).
# Each batch runs in an independent container and writes its own results.jsonl.
# Usage: Edit the batch list below, then: ./run_batch.sh
set -eu
BINDIR=$(cd "$(dirname "$0")/../mdlayer" && pwd)
BATCHES=(
    # "SYSTEM_DIR staged 42"
    # "SYSTEM_DIR weak 42"
)
for batch in "${BATCHES[@]}"; do
    set -- $batch
    SID=$1; PREP=$2; SEED=$3
    OUT=$PWD/output/$(basename "$SID")
    [ -f "$OUT/results.jsonl" ] && { echo "skip $SID"; continue; }
    bash "$(dirname "$0")/run_production.sh" "$SID" "$PREP" "$SEED"
done
echo "ALL_BATCHES_DONE $(date)"
