#!/usr/bin/env bash
# Standalone gate: TextVQA full-576 on the first 200 val samples; exit 2 if acc < 0.53.
# (scripts/sweep.py runs the same gate in-process by default, so the main block does not call this.)
set -euo pipefail
cd "$(dirname "$0")/.."
python scripts/sweep.py --gate_only "$@"
