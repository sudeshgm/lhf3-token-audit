#!/usr/bin/env bash
# Smoke: 20 samples/benchmark (POPE: 20 per split) x {full-576, random k=58 seed0, keep_last k=58}.
set -euo pipefail
cd "$(dirname "$0")/.."
LIMIT=${LIMIT:-20}
OUT=${OUT:-results/smoke}
MODEL=${MODEL:-llava-hf/llava-1.5-7b-hf}
EXTRA=${EXTRA:-}
mkdir -p "$OUT"
for bench in gqa pope textvqa; do
  for spec in "full 576 0" "random 58 0" "keep_last_k 58 0"; do
    set -- $spec
    tag=$([ "$1" = random ] && echo "seed$3" || echo seedNA)
    python eval.py --bench "$bench" --mode "$1" --k "$2" --seed "$3" --limit "$LIMIT" \
      --source "data/smoke/$bench.parquet" --out "$OUT/$bench/${1}_k${2}_${tag}.json" \
      --model "$MODEL" $EXTRA 2>&1 | grep -v -i -E "warn|slow image processor" | tee -a "$OUT/smoke.log"
  done
done
python scripts/summarize.py "$OUT"
