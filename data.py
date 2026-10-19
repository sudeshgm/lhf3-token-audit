"""Benchmark loaders -> normalized records.

Record: {sid, category, question, answer, answers, ocr_tokens, image_bytes}
Sources:
  - 'hub'           : lmms-lab HF datasets (lmms-lab/GQA, lmms-lab/POPE, lmms-lab/textvqa)
  - path to .parquet: a normalized subset produced by scripts/make_smoke_subsets.py
"""
from __future__ import annotations

import io

HUB = {
    "gqa": dict(repo="lmms-lab/GQA", instr_config="testdev_balanced_instructions",
                img_config="testdev_balanced_images", split="testdev"),
    "pope": dict(repo="lmms-lab/POPE", config="default", split="test"),
    "textvqa": dict(repo="lmms-lab/textvqa", split="validation"),
}
POPE_CATS = ("random", "popular", "adversarial")


def _img_bytes(img) -> bytes:
    if isinstance(img, dict) and img.get("bytes") is not None:
        return img["bytes"]
    buf = io.BytesIO()
    img.convert("RGB").save(buf, format="PNG")
    return buf.getvalue()


def _limit_pope(records, limit):
    if not limit:
        return records
    out, seen = [], {c: 0 for c in POPE_CATS}
    for r in records:
        if seen[r["category"]] < limit:
            seen[r["category"]] += 1
            out.append(r)
    return out


def load_records(bench: str, source: str = "hub", limit: int | None = None) -> list[dict]:
    """limit: first N samples (for POPE: first N per category, i.e. 3N total)."""
    if source != "hub":
        import pyarrow.parquet as pq
        recs = pq.read_table(source).to_pylist()
        recs = [r for r in recs if r.get("bench", bench) == bench]
        if bench == "pope":
            return _limit_pope(recs, limit)
        return recs[:limit] if limit else recs

    from datasets import load_dataset, Image as HFImage
    if bench == "gqa":
        c = HUB["gqa"]
        q = load_dataset(c["repo"], c["instr_config"], split=c["split"])
        if limit:
            q = q.select(range(min(limit, len(q))))
        need = set(q["imageId"])
        im = load_dataset(c["repo"], c["img_config"], split=c["split"]).cast_column("image", HFImage(decode=False))
        id2img = {r["id"]: r["image"] for r in im if r["id"] in need}
        return [dict(sid=str(r["id"]), category="", question=r["question"], answer=r["answer"],
                     answers=[r["answer"]], ocr_tokens=[], image_bytes=_img_bytes(id2img[r["imageId"]]))
                for r in q]
    if bench == "pope":
        c = HUB["pope"]
        ds = load_dataset(c["repo"], split=c["split"]).cast_column("image", HFImage(decode=False))
        recs, seen = [], {k: 0 for k in POPE_CATS}
        for r in ds:
            cat = r["category"]
            if limit and seen[cat] >= limit:
                if all(v >= limit for v in seen.values()):
                    break
                continue
            seen[cat] += 1
            recs.append(dict(sid=str(r["id"]), category=cat, question=r["question"], answer=r["answer"],
                             answers=[r["answer"]], ocr_tokens=[], image_bytes=_img_bytes(r["image"])))
        return recs
    if bench == "textvqa":
        c = HUB["textvqa"]
        ds = load_dataset(c["repo"], split=c["split"]).cast_column("image", HFImage(decode=False))
        if limit:
            ds = ds.select(range(min(limit, len(ds))))
        return [dict(sid=str(r["question_id"]), category="", question=r["question"], answer="",
                     answers=list(r["answers"]), ocr_tokens=list(r["ocr_tokens"] or []),
                     image_bytes=_img_bytes(r["image"])) for r in ds]
    raise ValueError(bench)
