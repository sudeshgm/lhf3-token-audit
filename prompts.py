"""LLaVA-1.5 evaluation prompts, audited against the official question files in LLaVA's eval.zip
(third_party/llava_eval/*.jsonl) and llava/conversation.py `conv_vicuna_v1`.

Official model_vqa_loader: qs = "<image>\n" + line["text"]; conv vicuna_v1 ->
"{SYSTEM} USER: <image>\n{text} ASSISTANT:"; greedy (temperature 0), max_new_tokens 128.
"""

# conv_vicuna_v1.system (llava/conversation.py). NOTE: "user", not "human" (fixed in step 2A).
SYSTEM = ("A chat between a curious user and an artificial intelligence assistant. "
          "The assistant gives helpful, detailed, and polite answers to the user's questions.")
SHORT = "\nAnswer the question using a single word or phrase."

# TextVQA: LLaVA-1.5's published 58.2 (7B) uses llava_textvqa_val_v051_ocr.jsonl:
#   "{question.capitalize()}\nReference OCR token: {', '.join(ocr_tokens)}\nAnswer the question using a single word or phrase."
# (verified identical for all 5000 lmms-lab/textvqa validation rows; lmms-lab questions are all lowercase).
TEXTVQA_OCR_DEFAULT = True


def question_text(bench: str, rec: dict, textvqa_ocr: bool = TEXTVQA_OCR_DEFAULT) -> str:
    q = rec["question"]
    if bench == "textvqa":
        q = q.capitalize()
        if textvqa_ocr:
            q = q + "\nReference OCR token: " + ", ".join(rec.get("ocr_tokens") or [])
    return q + SHORT


def full_prompt(qtext: str) -> str:
    return f"{SYSTEM} USER: <image>\n{qtext} ASSISTANT:"
