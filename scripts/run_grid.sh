#!/usr/bin/env bash
# FULL grid (do not run before the smoke is reviewed). Loads benchmarks from the HF hub.
set -euo pipefail
cd "$(dirname "$0")/.."
for bench in gqa pope textvqa; do
  python eval.py --bench $bench --mode full --k 576
  for k in 6 29 58 144; do
    python eval.py --bench $bench --mode keep_last_k --k $k
    for s in 0 1 2; do python eval.py --bench $bench --mode random --k $k --seed $s; done
  done
done
python scripts/summarize.py results
