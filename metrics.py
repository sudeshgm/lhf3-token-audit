"""Scoring, following the official LLaVA-1.5 eval scripts.

- GQA: llava/eval/eval_gqa.py style exact match (lowercase, strip trailing '.').
- POPE: llava/eval/eval_pope.py parsing (first sentence; 'No'/'not'/'no' -> no) + acc/P/R/F1/yes-ratio.
- TextVQA: m4c TextVQAAccuracyEvaluator with EvalAIAnswerProcessor
  (acc = mean over 10 leave-one-out subsets of min(1, #match/3)).
"""
from __future__ import annotations

import re


# ---------------- GQA ----------------
def gqa_norm(s: str) -> str:
    s = s.strip().lower()
    s = s.rstrip(".")
    return s.strip()


def gqa_correct(pred: str, gt: str) -> float:
    return float(gqa_norm(pred) == gqa_norm(gt))


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


# ---------------- TextVQA (EvalAI answer processor) ----------------
_CONTRACTIONS = {
    "aint": "ain't", "arent": "aren't", "cant": "can't", "couldve": "could've", "couldnt": "couldn't",
    "couldn'tve": "couldn't've", "couldnt've": "couldn't've", "didnt": "didn't", "doesnt": "doesn't",
    "dont": "don't", "hadnt": "hadn't", "hadnt've": "hadn't've", "hadn'tve": "hadn't've",
    "hasnt": "hasn't", "havent": "haven't", "hed": "he'd", "hed've": "he'd've", "he'dve": "he'd've",
    "hes": "he's", "howd": "how'd", "howll": "how'll", "hows": "how's", "Id've": "I'd've",
    "I'dve": "I'd've", "Im": "I'm", "Ive": "I've", "isnt": "isn't", "itd": "it'd", "itd've": "it'd've",
    "it'dve": "it'd've", "itll": "it'll", "let's": "let's", "maam": "ma'am", "mightnt": "mightn't",
    "mightnt've": "mightn't've", "mightn'tve": "mightn't've", "mightve": "might've", "mustnt": "mustn't",
    "mustve": "must've", "neednt": "needn't", "notve": "not've", "oclock": "o'clock", "oughtnt": "oughtn't",
    "ow's'at": "'ow's'at", "'ows'at": "'ow's'at", "'ow'sat": "'ow's'at", "shant": "shan't",
    "shed've": "she'd've", "she'dve": "she'd've", "she's": "she's", "shouldve": "should've",
    "shouldnt": "shouldn't", "shouldnt've": "shouldn't've", "shouldn'tve": "shouldn't've",
    "somebody'd": "somebodyd", "somebodyd've": "somebody'd've", "somebody'dve": "somebody'd've",
    "somebodyll": "somebody'll", "somebodys": "somebody's", "someoned": "someone'd",
    "someoned've": "someone'd've", "someone'dve": "someone'd've", "someonell": "someone'll",
    "someones": "someone's", "somethingd": "something'd", "somethingd've": "something'd've",
    "something'dve": "something'd've", "somethingll": "something'll", "thats": "that's",
    "thered": "there'd", "thered've": "there'd've", "there'dve": "there'd've", "therere": "there're",
    "theres": "there's", "theyd": "they'd", "theyd've": "they'd've", "they'dve": "they'd've",
    "theyll": "they'll", "theyre": "they're", "theyve": "they've", "twas": "'twas", "wasnt": "wasn't",
    "wed've": "we'd've", "we'dve": "we'd've", "weve": "we've", "werent": "weren't", "whatll": "what'll",
    "whatre": "what're", "whats": "what's", "whatve": "what've", "whens": "when's", "whered": "where'd",
    "wheres": "where's", "whereve": "where've", "whod": "who'd", "whod've": "who'd've",
    "who'dve": "who'd've", "wholl": "who'll", "whos": "who's", "whove": "who've", "whyll": "why'll",
    "whyre": "why're", "whys": "why's", "wont": "won't", "wouldve": "would've", "wouldnt": "wouldn't",
    "wouldnt've": "wouldn't've", "wouldn'tve": "wouldn't've", "yall": "y'all", "yall'll": "y'all'll",
    "y'allll": "y'all'll", "yall'd've": "y'all'd've", "y'alld've": "y'all'd've", "y'all'dve": "y'all'd've",
    "youd": "you'd", "youd've": "you'd've", "you'dve": "you'd've", "youll": "you'll", "youre": "you're",
    "youve": "you've",
}
_NUM = {"none": "0", "zero": "0", "one": "1", "two": "2", "three": "3", "four": "4", "five": "5",
        "six": "6", "seven": "7", "eight": "8", "nine": "9", "ten": "10"}
_ARTICLES = ["a", "an", "the"]
_PERIOD_STRIP = re.compile(r"(?!<=\d)(\.)(?!\d)")
_COMMA_STRIP = re.compile(r"(\d)(\,)(\d)")
_PUNCT = [";", r"/", "[", "]", '"', "{", "}", "(", ")", "=", "+", "\\", "_", "-",
          ">", "<", "@", "`", ",", "?", "!"]


def _process_punctuation(in_text: str) -> str:
    out = in_text
    for p in _PUNCT:
        if (p + " " in in_text or " " + p in in_text) or (re.search(_COMMA_STRIP, in_text) is not None):
            out = out.replace(p, "")
        else:
            out = out.replace(p, " ")
    out = _PERIOD_STRIP.sub("", out, re.UNICODE)
    return out


def _process_digit_article(in_text: str) -> str:
    out = []
    for w in in_text.lower().split():
        w = _NUM.setdefault(w, w)
        if w not in _ARTICLES:
            out.append(w)
    for i, w in enumerate(out):
        if w in _CONTRACTIONS:
            out[i] = _CONTRACTIONS[w]
    return " ".join(out)


def evalai_process(ans: str) -> str:
    ans = ans.replace("\n", " ").replace("\t", " ").strip()
    ans = _process_punctuation(ans)
    ans = _process_digit_article(ans)
    return ans


def textvqa_acc(pred: str, gt_answers: list[str]) -> float:
    pred = evalai_process(pred)
    gts = [evalai_process(a) for a in gt_answers]
    accs = []
    for i in range(len(gts)):
        others = gts[:i] + gts[i + 1:]
        accs.append(min(1.0, sum(a == pred for a in others) / 3.0))
    return sum(accs) / len(accs) if accs else 0.0
