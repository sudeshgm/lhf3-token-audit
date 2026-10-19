"""Protocol tests against the official LLaVA-1.5 eval files (third_party/llava_eval)."""
import gzip
import json
import os
import re
import sys

import pyarrow.parquet as pq

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
import metrics as M  # noqa: E402
import prompts as P  # noqa: E402
from third_party.m4c_evaluator import TextVQAAccuracyEvaluator  # noqa: E402

TP = os.path.join(ROOT, "third_party/llava_eval")
jl = lambda f: [json.loads(l) for l in gzip.open(os.path.join(TP, f), "rt")]
smoke = lambda b: pq.read_table(os.path.join(ROOT, f"data/smoke/{b}.parquet")).to_pylist()


def test_system_prompt_is_official_vicuna_v1():
    src = open(os.path.join(TP, "conversation.py.txt")).read()
    block = src[src.index("conv_vicuna_v1 = Conversation("):]
    m = re.search(r'system="(.*?)"\s*"(.*?)",', block, re.S)
    assert P.SYSTEM == m.group(1) + m.group(2)
    assert P.full_prompt("Q") == P.SYSTEM + " USER: <image>\nQ ASSISTANT:"


def test_textvqa_prompts_match_official_file():
    L = jl("llava_textvqa_val_v051_ocr.jsonl.gz")
    rows = smoke("textvqa")
    assert len(rows) == 20
    for r, l in zip(rows, L):
        assert P.question_text("textvqa", r) == l["text"]
        assert "\nReference OCR token: " in P.question_text("textvqa", r)


def test_gqa_pope_prompts_match_official_file():
    L = {l["question_id"]: l for l in jl("llava_gqa_testdev_balanced.jsonl.gz")}
    for r in smoke("gqa"):
        assert P.question_text("gqa", r) == L[r["sid"]]["text"]
    L = {(l["category"], l["text"].split("\n")[0]): l["text"] for l in jl("llava_pope_test.jsonl.gz")}
    hits = 0
    for r in smoke("pope"):
        k = (r["category"], r["question"])
        if k in L:
            hits += 1
            assert P.question_text("pope", r) == L[k]
    assert hits >= 50


def test_textvqa_scorer_equals_official_on_13b_answers():
    A = jl("textvqa_answers_llava-v1.5-13b.jsonl.gz")[:20]
    rows = smoke("textvqa")
    ours = [M.textvqa_acc(a["text"], r["answers"]) for a, r in zip(A, rows)]
    ev = TextVQAAccuracyEvaluator()
    off = ev.eval_pred_list([{"pred_answer": a["text"], "gt_answers": r["answers"]} for a, r in zip(A, rows)])
    assert abs(sum(ours) / 20 - off) < 1e-12
    assert abs(off - 0.51) < 1e-9          # official 13B on the first 20 val questions


def test_pope_parser_official_semantics():
    assert M.pope_parse("No") == "no"
    assert M.pope_parse("Yes") == "yes"
    assert M.pope_parse("There is no dog in the image.") == "no"
    assert M.pope_parse("Yes, there is a dog. No other animals.") == "yes"  # first sentence only
    assert M.pope_parse("") == "yes"       # empty -> yes (official); cannot produce F1=0
    assert M.pope_parse("NO") == "yes"     # official quirk: only 'No'/'no'/'not' count
    m = M.pope_metrics(["no"] * 4, ["yes", "no", "yes", "no"])
    assert m["f1"] == 0.0 and m["acc"] == 0.5 and m["yes_ratio"] == 0.0   # the smoke keep_last pattern


def test_gqa_scorer_official():
    assert M.gqa_correct("Yes.", "yes") == 1.0
    assert M.gqa_correct(" White ", "white") == 1.0
    assert M.gqa_correct("white car", "white") == 0.0
