#!/usr/bin/env python
"""LHF3 Experiment -1 eval entrypoint: ONE config (minimal custom loop, NOT lmms-eval; see README).
For many configs with a single model load use scripts/sweep.py.

Example:
  python eval.py --bench gqa --mode random --k 58 --seed 0 --limit 20 \
      --source data/smoke/gqa.parquet --out results/smoke/gqa/random_k58_seed0.json
"""
from __future__ import annotations

import argparse
import json
import os

import core  # sets USE_TF=0 etc. before transformers is imported
from data import load_records


def parse():
    p = argparse.ArgumentParser()
    p.add_argument("--bench", required=True, choices=["gqa", "pope", "textvqa"])
    p.add_argument("--mode", required=True, choices=["full", "random", "keep_last_k"])
    p.add_argument("--k", type=int, default=core.N_VIS)
    p.add_argument("--seed", type=int, default=0, help="random arm only (0/1/2)")
    p.add_argument("--limit", type=int, default=None, help="first N samples (POPE: N per split)")
    p.add_argument("--ids", default=None, help="JSON file with a list of sample ids (subsets/*.json)")
    p.add_argument("--source", default="hub", help="'hub' or a normalized parquet (data/smoke/*.parquet)")
    p.add_argument("--out", default=None)
    p.add_argument("--model", default=os.environ.get("MODEL", core.MODEL_ID))
    p.add_argument("--device", default=None)
    p.add_argument("--dtype", default="float16", choices=["float16", "bfloat16", "float32"])
    p.add_argument("--max_new_tokens", type=int, default=128)
    p.add_argument("--textvqa_ocr", type=int, default=1)
    p.add_argument("--print_every", type=int, default=1)
    p.add_argument("--min_acc", type=float, default=None, help="exit 2 if acc (or POPE mean F1) below this")
    return p.parse_args()


def main():
    a = parse()
    if a.mode == "full":
        a.k = core.N_VIS
    tag = f"seed{a.seed}" if a.mode == "random" else "seedNA"
    a.out = a.out or f"results/{a.bench}/{a.mode}_k{a.k}_{tag}.json"
    runner = core.Runner(a.model, a.device, a.dtype, a.max_new_tokens, bool(a.textvqa_ocr))
    ids = json.load(open(a.ids))["ids"] if a.ids else None
    recs = load_records(a.bench, a.source, a.limit, ids)
    print(f"[lhf3] bench={a.bench} mode={a.mode} k={a.k} seed={a.seed if a.mode == 'random' else 'NA'} n={len(recs)} "
          f"source={a.source} model={a.model} dtype={a.dtype} device={runner.device} git={core.git_rev()} "
          f"textvqa_ocr={bool(a.textvqa_ocr)} load_s={runner.load_s:.1f}", flush=True)
    rows = []
    for i, r in enumerate(recs):
        row = runner.run_sample(a.bench, r, a.mode, a.k, a.seed)
        rows.append(row)
        if i < 3:
            print(f"[lhf3] PROMPT[{i}] {row['prompt']!r}", flush=True)
        if a.print_every and i % a.print_every == 0:
            print(f"[{i+1}/{len(recs)}] vis_tokens_to_LLM={row['visual_tokens_to_llm']} prefill_len={row['llm_prefill_len']} "
                  f"raw={row['raw_generation'][:40]!r} gt={r['answers'][:1]} score={row['score']:.2f} {row['sec']:.2f}s", flush=True)
    s = core.summary_dict(runner, a.bench, a.mode, a.k, a.seed, rows, dict(source=a.source, limit=a.limit, ids=a.ids))
    core.write_json(a.out, s)
    print("[lhf3] SUMMARY " + json.dumps(core.short(s)), flush=True)
    print(f"[lhf3] wrote {a.out}")
    if a.min_acc is not None:
        m = s["metrics"]
        val = m.get("acc", m.get("mean_f1"))
        print(f"[lhf3] GATE {a.bench} {val:.4f} >= {a.min_acc}? {'PASS' if val >= a.min_acc else 'FAIL'}", flush=True)
        if val < a.min_acc:
            raise SystemExit(2)


if __name__ == "__main__":
    main()
