"""LLaVA-1.5 evaluation prompts (conv_mode=vicuna_v1, as in haotian-liu/LLaVA scripts/v1_5/eval)."""

SYSTEM = ("A chat between a curious human and an artificial intelligence assistant. "
          "The assistant gives helpful, detailed, and polite answers to the human's questions.")
SHORT = "\nAnswer the question using a single word or phrase."

# TextVQA: LLaVA-1.5's published number uses llava_textvqa_val_v051_ocr.jsonl, i.e. the
# question followed by "Reference OCR token: ..." (TextVQA-provided Rosetta OCR tokens).
TEXTVQA_OCR_DEFAULT = True


def question_text(bench: str, rec: dict, textvqa_ocr: bool = TEXTVQA_OCR_DEFAULT) -> str:
    q = rec["question"]
    if bench == "textvqa" and textvqa_ocr:
        q = q + "\nReference OCR token: " + ", ".join(rec.get("ocr_tokens") or [])
    return q + SHORT


def full_prompt(qtext: str) -> str:
    return f"{SYSTEM} USER: <image>\n{qtext} ASSISTANT:"
