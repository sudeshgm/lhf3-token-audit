"""Scoring, following the official LLaVA-1.5 eval scripts.

- GQA: llava/eval/eval_gqa.py style exact match (lowercase, strip trailing '.').
- POPE: llava/eval/eval_pope.py parsing (first sentence; 'No'/'not'/'no' -> no) + acc/P/R/F1/yes-ratio.
- TextVQA: m4c TextVQAAccuracyEvaluator with EvalAIAnswerProcessor
  (acc = mean over 10 leave-one-out subsets of min(1, #match/3)).
"""
from __future__ import annotations



# ---------------- GQA ----------------
# Official: scripts/convert_gqa_for_eval.py -> prediction = text.rstrip('.').lower(); GQA eval.py: prediction == gold.
def gqa_norm(s: str) -> str:
    return s.strip().rstrip(".").lower()


def gqa_correct(pred: str, gt: str) -> float:
    return float(gqa_norm(pred) == gt)


# ---------------- POPE ----------------
def pope_parse(text: str) -> str:
    text = text.strip()
    if text.find(".") != -1:
        text = text.split(".")[0]
    text = text.replace(",", "")
    words = text.split(" ")
    if "No" in words or "not" in words or "no" in words:
        return "no"
    return "yes"


def pope_metrics(preds: list[str], gts: list[str]) -> dict:
    tp = fp = tn = fn = 0
    for p, g in zip(preds, gts):
        p, g = p.lower(), g.lower()
        if p == "yes" and g == "yes": tp += 1
        elif p == "yes" and g == "no": fp += 1
        elif p == "no" and g == "no": tn += 1
        else: fn += 1
    n = max(tp + fp + tn + fn, 1)
    prec = tp / (tp + fp) if tp + fp else 0.0
    rec = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * prec * rec / (prec + rec) if prec + rec else 0.0
    return {"n": tp + fp + tn + fn, "acc": (tp + tn) / n, "precision": prec, "recall": rec,
            "f1": f1, "yes_ratio": (tp + fp) / n, "tp": tp, "fp": fp, "tn": tn, "fn": fn}


# ---------------- TextVQA ----------------
# Verbatim official scorer: third_party/m4c_evaluator.py (LLaVA llava/eval/m4c_evaluator.py).
from third_party.m4c_evaluator import EvalAIAnswerProcessor, TextVQAAccuracyEvaluator

_PROC = EvalAIAnswerProcessor()
_EVAL = TextVQAAccuracyEvaluator()


def evalai_process(ans: str) -> str:
    return _PROC(ans)


def textvqa_acc(pred: str, gt_answers: list[str]) -> float:
    """Per-question soft accuracy exactly as TextVQAAccuracyEvaluator.eval_pred_list:
    EvalAI-normalize pred and the 10 GT answers, then mean over the 10 leave-one-out subsets of
    min(1, #matches/3)."""
    scores = _EVAL._compute_answer_scores(gt_answers)
    return scores.get(_PROC(pred), 0.0)
