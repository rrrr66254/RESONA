#!/bin/bash
# Chain pretrain.sh — auto-resubmit while ckpt exists and target steps not reached.
# Usage: ./chain.sh [N] (default N=10 → up to 10 × 48h = 480h)

set -e
N=${1:-10}
SCRIPT=$(dirname "$(readlink -f "$0")")/pretrain.sh

prev=""
for i in $(seq 1 $N); do
  if [ -z "$prev" ]; then
    jid=$(sbatch --parsable "$SCRIPT")
  else
    jid=$(sbatch --parsable --dependency=afterany:$prev "$SCRIPT")
  fi
  echo "[$i/$N] submitted job $jid (after $prev)"
  prev=$jid
done

echo "Final job: $prev"
echo "Cancel chain: scancel $(seq -s ' ' $((prev - N + 1)) $prev) 2>/dev/null  # approx"
