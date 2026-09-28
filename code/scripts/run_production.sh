#!/bin/bash
# Single 10-ns production run on a local GPU with OpenMM CUDA.
# Prerequisites: OpenMM >= 8.1 compiled for your GPU architecture.
# See docs/GPU_MD_SETUP.md in the main repository for build instructions.
# Usage: ./run_production.sh <SYSTEM_DIR> <staged|weak> <SEED> [--membrane]
set -eu
SID=$1; PREP=$2; SEED=$3; ARM=${4:-}
BINDIR=$(cd "$(dirname "$0")/../mdlayer" && pwd)
WD=$PWD/runs/$(basename "$SID")-$PREP-s$SEED
rm -rf "$WD"; mkdir -p "$WD"; cd "$WD"
cp "$SID"/com_final.prmtop "$SID"/com_final.inpcrd ./
cp "$SID"/*gate*.json ticket.json 2>/dev/null || true
ARGS=()
[ "$ARM" = "--membrane" ] && ARGS+=(--membrane)
[ -s ticket.json ] && ARGS+=(--ticket ticket.json)
python3 "$BINDIR/md_explicit_v7.py" "$(basename "$SID")-$PREP" "$SEED" "$PREP" \
    com_final.prmtop com_final.inpcrd . "${ARGS[@]}"
