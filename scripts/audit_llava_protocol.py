#!/usr/bin/env python
"""Audit our prompts + scorers against the official LLaVA-1.5 eval files (third_party/llava_eval).

1. For every row of lmms-lab GQA testdev-balanced / POPE / TextVQA val: build our question text and diff it
   against the official question file (`text` field). Also diff TextVQA OCR tokens and GT answers.
2. Re-score the official llava-v1.5-13b answers with OUR scorers -> must reproduce the paper numbers
   (13B: GQA 63.3, TextVQA 61.3, POPE F1 rand/pop/adv 87.3/86.1/84.2 - our POPE uses lmms-lab's 9000 rows,
   LLaVA's file has 8910).
3. Print 3 exact TextVQA prompts fed to the processor.
Usage: python scripts/audit_llava_protocol.py [hub|LOCAL_DIR]   (LOCAL_DIR as in make_subsets.py)
"""
import collections
import gzip
import json
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
import pyarrow as pa  # noqa: E402
import pyarrow.parquet as pq  # noqa: E402

import metrics as M  # noqa: E402
import prompts as P  # noqa: E402
from data import POPE_CATS, hub_paths  # noqa: E402

src = sys.argv[1] if len(sys.argv) > 1 else "hub"
TP = os.path.join(ROOT, "third_party/llava_eval")
jl = lambda f: [json.loads(l) for l in gzip.open(os.path.join(TP, f), "rt")]


def paths(bench, which=0):
    if src == "hub":
        return hub_paths(bench)[which]
    local = {"gqa": ["gqa_instr.parquet"], "pope": [f"pope_{i}.parquet" for i in range(3)],
             "textvqa": [f"tvqa_{i}.parquet" for i in range(3)]}[bench]
    return [os.path.join(src, f) for f in local]


read = lambda b, cols: pa.concat_tables([pq.read_table(p, columns=cols) for p in paths(b)]).to_pylist()

# ---------- TextVQA ----------
T = read("textvqa", ["image_id", "question_id", "question", "answers", "ocr_tokens"])
L = jl("llava_textvqa_val_v051_ocr.jsonl.gz")
assert len(T) == len(L) == 5000
recs = [dict(question=t["question"], ocr_tokens=t["ocr_tokens"] or []) for t in T]
order = sum(t["image_id"] == l["question_id"] for t, l in zip(T, L))
ours = sum(P.question_text("textvqa", r) == l["text"] for r, l in zip(recs, L))
old = sum((r["question"] + "\nReference OCR token: " + ", ".join(r["ocr_tokens"]) + P.SHORT) == l["text"] for r, l in zip(recs, L))
ocr = sum(l["text"].split("\n")[1] == "Reference OCR token: " + ", ".join(r["ocr_tokens"]) for r, l in zip(recs, L))
print(f"[textvqa] same image order {order}/5000 | OCR line identical {ocr}/5000 | "
      f"question text identical: fixed {ours}/5000, step-1 (no capitalize) {old}/5000")
for i in range(3):
    print(f"[textvqa] PROMPT[{i}] {P.full_prompt(P.question_text('textvqa', recs[i]))!r}")
A = jl("textvqa_answers_llava-v1.5-13b.jsonl.gz")
assert all(a["question_id"] == t["image_id"] for a, t in zip(A, T))
acc = sum(M.textvqa_acc(a["text"], t["answers"]) for a, t in zip(A, T)) / len(T)
print(f"[textvqa] official 13B answers rescored with our scorer: {100 * acc:.2f} (paper 61.3)")

# ---------- GQA ----------
G = read("gqa", ["id", "question", "answer"])
L = jl("llava_gqa_testdev_balanced.jsonl.gz")
same = sum(g["id"] == l["question_id"] and P.question_text("gqa", g) == l["text"] for g, l in zip(G, L))
print(f"[gqa] question id + text identical {same}/{len(L)} (lmms rows {len(G)})")
A = {a["question_id"]: a["text"] for a in jl("gqa_answers_llava-v1.5-13b.jsonl.gz")}
acc = sum(M.gqa_correct(A[g["id"]], g["answer"]) for g in G) / len(G)
print(f"[gqa] official 13B answers rescored with our scorer: {100 * acc:.2f} (paper 63.3)")

# ---------- POPE ----------
R = read("pope", ["id", "question", "answer", "image_source", "category"])
L = jl("llava_pope_test.jsonl.gz")
A = {a["question_id"]: a["text"] for a in jl("pope_answers_llava-v1.5-13b.jsonl.gz")}
idx = {(r["category"], r["image_source"] + ".jpg", r["question"]): r for r in R}
hit = [(l, idx.get((l["category"], l["image"], l["text"].split("\n")[0]))) for l in L]
m = [(l, r) for l, r in hit if r]
print(f"[pope] lmms rows {len(R)} {dict(collections.Counter(r['category'] for r in R))}; LLaVA file {len(L)} "
      f"{dict(collections.Counter(l['category'] for l in L))}; matched by (split,image,question) {len(m)}")
print(f"[pope] prompt identical for matched: {sum(P.question_text('pope', r) == l['text'] for l, r in m)}/{len(m)}")
f1s = []
for c in POPE_CATS:
    sub = [(A[l["question_id"]], r["answer"]) for l, r in m if l["category"] == c]
    mm = M.pope_metrics([M.pope_parse(p) for p, _ in sub], [g for _, g in sub])
    f1s.append(mm["f1"])
    print(f"[pope] 13B {c:11} n={mm['n']} F1={100 * mm['f1']:.2f} acc={100 * mm['acc']:.2f} yes={100 * mm['yes_ratio']:.1f}")
print(f"[pope] 13B mean F1 {100 * sum(f1s) / 3:.2f} (paper: 85.9 for both 7B and 13B)")
