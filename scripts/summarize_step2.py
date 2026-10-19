#!/usr/bin/env python
"""Step-2 summary table in percentage points.

Columns: full-576 | per k: random-s0, keep_last | per k: random seed min/max over s0,s1,s2 on the 1k subsets
(s0 on 1k = the main seed-0 run restricted to the 1k ids). POPE = mean F1 over random/popular/adversarial,
plus per-split F1, mean acc, and a yes-ratio table. Usage: python scripts/summarize_step2.py [results/step2]
"""
import json
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
from data import POPE_CATS  # noqa: E402
import metrics as M  # noqa: E402

root = sys.argv[1] if len(sys.argv) > 1 else os.path.join(ROOT, "results/step2")
KS = [144, 58]
TARGET = {"gqa": 61.9, "pope_f1": 85.9, "textvqa": 58.2}
SUB1K = {"gqa": "subsets/gqa_1k.json", "pope": "subsets/pope_1k_per_split.json", "textvqa": "subsets/textvqa_1k.json"}


def load(bench, subset, mode, k, seed):
    t = f"{mode}_k{k}_" + (f"seed{seed}" if mode == "random" else "seedNA")
    p = os.path.join(root, bench, subset, t + ".json")
    return json.load(open(p)) if os.path.exists(p) else None


def agg(bench, rows):
    if bench == "pope":
        out = {}
        for c in POPE_CATS:
            sub = [r for r in rows if r["category"] == c]
            if sub:
                out[c] = M.pope_metrics([r["parsed"] for r in sub], [r["gt"][0] for r in sub])
        cats = [c for c in POPE_CATS if c in out]
        if cats:
            out["mean_f1"] = sum(out[c]["f1"] for c in cats) / len(cats)
            out["mean_acc"] = sum(out[c]["acc"] for c in cats) / len(cats)
        return out
    return {"acc": sum(r["score"] for r in rows) / max(len(rows), 1)}


def restrict(d, bench):
    if d is None:
        return None
    ids = set(map(str, json.load(open(os.path.join(ROOT, SUB1K[bench])))["ids"]))
    rows = [r for r in d["samples"] if r["sid"] in ids]
    return agg(bench, rows) if rows else None


LINES = [("gqa", "acc", "GQA acc"), ("textvqa", "acc", "TextVQA acc"), ("pope", "mean_f1", "POPE F1 (mean 3 splits)"),
         ("pope", "random", "  POPE F1 random"), ("pope", "popular", "  POPE F1 popular"),
         ("pope", "adversarial", "  POPE F1 adversarial"), ("pope", "mean_acc", "POPE acc (mean 3 splits)")]


def val(m, key):
    if m is None:
        return None
    v = m.get(key)
    if isinstance(v, dict):
        v = v["f1"]
    return None if v is None else 100 * v


def fmt(v):
    return "   -  " if v is None else f"{v:6.1f}"


hdr = f"{'metric (pp)':26} {'target':>6} {'full':>6} | " + " | ".join(f"k{k} rnd0 k{k} last" for k in KS) + \
      " | " + " | ".join(f"k{k} seed min/max (1k)" for k in KS)
print(hdr)
print("-" * len(hdr))
for bench, key, name in LINES:
    full = load(bench, "main", "full", 576, None)
    cells = [fmt(val(full and full["metrics"], key))]
    for k in KS:
        r0 = load(bench, "main", "random", k, 0)
        kl = load(bench, "main", "keep_last_k", k, None)
        cells.append(f"{fmt(val(r0 and r0['metrics'], key))}    {fmt(val(kl and kl['metrics'], key))}")
    for k in KS:
        vs = [val(restrict(load(bench, "main", "random", k, 0), bench), key)]
        vs += [val(d and d["metrics"], key) for d in (load(bench, "1k", "random", k, s) for s in (1, 2))]
        got = [v for v in vs if v is not None]
        cells.append(f"{min(got):6.1f}/{max(got):6.1f} (n={len(got)})" if got else "        -         ")
    tgt = TARGET.get("pope_f1" if key == "mean_f1" else bench) if key in ("acc", "mean_f1") else None
    print(f"{name:26} {fmt(tgt)} {cells[0]} | " + " | ".join(cells[1:1 + len(KS)]) + " | " + " | ".join(cells[1 + len(KS):]))

print("\nPOPE yes-ratio (%) per split [random/popular/adversarial]; balanced GT = 50")
for subset, mode, k, seed in [("main", "full", 576, None)] + [("main", m, k, 0 if m == "random" else None)
                                                             for k in KS for m in ("random", "keep_last_k")] + \
        [("1k", "random", k, s) for k in KS for s in (1, 2)]:
    d = load("pope", subset, mode, k, seed)
    if d:
        m = d["metrics"]
        yr = "/".join(f"{100 * m[c]['yes_ratio']:.1f}" for c in POPE_CATS if c in m)
        print(f"  {subset:4} {mode:12} k={k:<3} seed={str(seed):4} n={d['n']:<5} yes={yr}  empty={m.get('n_empty_outputs', 0)}")

print("\nRuntime / VRAM")
for bench in ("gqa", "pope", "textvqa"):
    for sub in ("main", "1k"):
        d_ = os.path.join(root, bench, sub)
        if os.path.isdir(d_):
            for f in sorted(os.listdir(d_)):
                if f.endswith(".json"):
                    d = json.load(open(os.path.join(d_, f)))
                    pk = d.get("peak_vram_gib")
                    print(f"  {bench:7} {sub:4} {f[:-5]:24} n={d['n']:<5} vis_tok={d['visual_tokens_to_llm']} "
                          f"{d['sec_per_sample_mean']:.3f}s/sample peak={('%.2f GiB' % pk) if pk else '-'}")
