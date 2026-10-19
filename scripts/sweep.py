#!/usr/bin/env python
"""Resumable sweep: load LLaVA ONCE, loop over configs/sweep_step2.json inside one process.

- One results JSON per (bench, subset, arm, k, seed): <out_root>/<bench>/<subset>/<arm>_k<k>_seed<s|NA>.json
  written atomically after each config. Configs whose JSON exists are skipped.
- Within a config, every finished sample is appended to <json>.partial.jsonl, so a crash/preemption
  resumes mid-config.
- Gate (unless --no_gate): TextVQA full-576 on the first 200 samples; exit 2 if acc < 0.53. The gate's
  samples are the first 200 of the full-576 TextVQA run and are reused by it (same partial file).
- Data comes from the lmms-lab Hub parquet shards via hf_hub_download into $HF_HOME (or --cache_dir).

Env/flags: MODEL (local dir or hub id), HF_HOME. Tiny CPU check:
  MODEL=/path/tiny python scripts/sweep.py --device cpu --dtype float32 --limit 2 --out_root /tmp/x --max_new_tokens 3
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
os.chdir(ROOT)

import core  # noqa: E402  (sets USE_TF=0 before transformers import)
from data import load_records  # noqa: E402


def parse():
    p = argparse.ArgumentParser()
    p.add_argument("--config", default="configs/sweep_step2.json")
    p.add_argument("--out_root", default="results/step2")
    p.add_argument("--model", default=os.environ.get("MODEL", core.MODEL_ID))
    p.add_argument("--cache_dir", default=None, help="HF cache dir; default = $HF_HOME/hub")
    p.add_argument("--device", default=None)
    p.add_argument("--dtype", default="float16")
    p.add_argument("--max_new_tokens", type=int, default=128)
    p.add_argument("--limit", type=int, default=None, help="cap samples per config (POPE: per split) for testing")
    p.add_argument("--only", default=None, help="substring filter on config tags, e.g. 'textvqa'")
    p.add_argument("--no_gate", action="store_true")
    p.add_argument("--gate_only", action="store_true")
    p.add_argument("--gate_min", type=float, default=None, help="override gate threshold (testing only)")
    p.add_argument("--print_every", type=int, default=100)
    return p.parse_args()


def tag(mode, k, seed):
    return f"{mode}_k{k}_" + (f"seed{seed}" if mode == "random" else "seedNA")


def out_path(root, bench, subset, mode, k, seed):
    return os.path.join(root, bench, subset, tag(mode, k, seed) + ".json")


class Data:
    def __init__(self, cfg, cache_dir):
        self.cfg, self.cache_dir, self.cache = cfg, cache_dir, {}

    def get(self, bench, subset, limit):
        key = (bench, subset, limit)
        if key not in self.cache:
            f = self.cfg["subsets"][subset][bench]
            ids = json.load(open(f))["ids"] if f else None
            t0 = time.time()
            self.cache[key] = load_records(bench, "hub", limit, ids, self.cache_dir)
            print(f"[sweep] loaded {bench}/{subset} n={len(self.cache[key])} in {time.time() - t0:.0f}s", flush=True)
        return self.cache[key]


def run_config(runner, recs, bench, subset, mode, k, seed, path, print_every, extra=None):
    part = path + ".partial.jsonl"
    done = {}
    if os.path.exists(part):
        for line in open(part):
            try:
                r = json.loads(line)
                done[r["sid"]] = r
            except json.JSONDecodeError:
                pass  # truncated last line after a crash
    todo = [r for r in recs if r["sid"] not in done]
    print(f"[sweep] RUN {bench}/{subset}/{tag(mode, k, seed)} n={len(recs)} resumed={len(recs) - len(todo)}", flush=True)
    t0 = time.time()
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(part, "a") as fp:
        for i, r in enumerate(todo):
            row = runner.run_sample(bench, r, mode, k, seed)
            done[r["sid"]] = row
            fp.write(json.dumps(row) + "\n")
            fp.flush()
            if print_every and (i % print_every == 0 or i == len(todo) - 1):
                el = time.time() - t0
                print(f"[sweep]   {i + 1}/{len(todo)} vis_tok={row['visual_tokens_to_llm']} raw={row['raw_generation'][:30]!r} "
                      f"{el / (i + 1):.2f}s/it eta={el / (i + 1) * (len(todo) - i - 1) / 60:.0f}min", flush=True)
    rows = [done[r["sid"]] for r in recs]
    return core.summary_dict(runner, bench, mode, k, seed, rows, dict(subset=subset, source="hub", **(extra or {})))


def main():
    a = parse()
    cfg = json.load(open(a.config))
    data = Data(cfg, a.cache_dir)
    runner = core.Runner(a.model, a.device, a.dtype, a.max_new_tokens)
    print(f"[sweep] model={a.model} device={runner.device} load_s={runner.load_s:.0f} git={core.git_rev()} "
          f"HF_HOME={os.environ.get('HF_HOME')} out_root={a.out_root}", flush=True)

    if not a.no_gate:
        g = dict(cfg["gate"])
        if a.gate_min is not None:
            g["min_acc"] = a.gate_min
        n = min(g["limit"], a.limit) if a.limit else g["limit"]
        recs = data.get(g["bench"], "main", n)
        for i, r in enumerate(recs[:3]):
            print(f"[sweep] GATE PROMPT[{i}] {runner.build_prompt(g['bench'], r)!r}", flush=True)
        main_path = out_path(a.out_root, g["bench"], "main", "full", core.N_VIS, None)
        if a.limit:  # tiny test: do not seed the real run
            main_path = out_path(a.out_root, g["bench"], "gate", "full", core.N_VIS, None)
        s = run_config(runner, recs, g["bench"], "main", "full", core.N_VIS, None, main_path, a.print_every)
        acc = s["metrics"]["acc"]
        core.write_json(os.path.join(a.out_root, "gate", f"{g['bench']}_full576_first{n}.json"), s)
        print(f"[sweep] GATE textvqa full-576 first {n}: acc={acc:.4f} (threshold {g['min_acc']}) "
              f"empty={s['metrics']['n_empty_outputs']} -> {'PASS' if acc >= g['min_acc'] else 'FAIL'}", flush=True)
        if acc < g["min_acc"]:
            sys.exit(2)
        if a.gate_only:
            return

    for c in cfg["configs"]:
        for bench in c["benches"]:
            k = core.N_VIS if c["mode"] == "full" else c["k"]
            path = out_path(a.out_root, bench, c["subset"], c["mode"], k, c["seed"])
            t = f"{bench}/{c['subset']}/{tag(c['mode'], k, c['seed'])}"
            if a.only and a.only not in t:
                continue
            if os.path.exists(path):
                print(f"[sweep] SKIP {t} (exists)", flush=True)
                continue
            recs = data.get(bench, c["subset"], a.limit)
            s = run_config(runner, recs, bench, c["subset"], c["mode"], k, c["seed"] if c["mode"] == "random" else 0,
                           path, a.print_every)
            core.write_json(path, s)
            os.remove(path + ".partial.jsonl")
            print("[sweep] DONE " + json.dumps(core.short(s)), flush=True)
    print("[sweep] ALL DONE", flush=True)


if __name__ == "__main__":
    main()
