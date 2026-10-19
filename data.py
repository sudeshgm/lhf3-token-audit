"""Benchmark loaders -> normalized records.

Record: {sid, category, question, answer, answers, ocr_tokens, image_bytes}
Sources:
  - 'hub'           : the lmms-lab HF dataset parquet shards (lmms-lab/GQA, lmms-lab/POPE, lmms-lab/textvqa),
                      fetched with hf_hub_download into $HF_HOME (no `datasets` Arrow copy) and read with pyarrow.
  - path to .parquet: a normalized subset produced by scripts/make_smoke_subsets.py
Optional `ids`: keep only these sample ids (see subsets/*.json), in dataset order.
"""
from __future__ import annotations

import pyarrow as pa
import pyarrow.parquet as pq

# lmms-lab/* now redirect to lmms-lab-encoder/*; both resolve on the Hub.
HUB_FILES = {
    "gqa": ("lmms-lab/GQA", ["testdev_balanced_instructions/testdev-00000-of-00001.parquet"],
            ["testdev_balanced_images/testdev-00000-of-00001.parquet"]),
    "pope": ("lmms-lab/POPE", [f"data/test-0000{i}-of-00003.parquet" for i in range(3)], []),
    "textvqa": ("lmms-lab/textvqa", [f"data/validation-0000{i}-of-00003.parquet" for i in range(3)], []),
}
HUB_BYTES = {"gqa": 67_738_080, "pope": 255_022_914, "textvqa": 920_321_540}
POPE_CATS = ("random", "popular", "adversarial")


def _limit(records, bench, limit):
    if not limit:
        return records
    if bench != "pope":
        return records[:limit]
    out, seen = [], {c: 0 for c in POPE_CATS}
    for r in records:
        if seen[r["category"]] < limit:
            seen[r["category"]] += 1
            out.append(r)
    return out


def hub_paths(bench: str, cache_dir: str | None = None):
    from huggingface_hub import hf_hub_download
    repo, main, extra = HUB_FILES[bench]
    get = lambda f: hf_hub_download(repo, f, repo_type="dataset", cache_dir=cache_dir)
    return [get(f) for f in main], [get(f) for f in extra]


def _read(paths, columns):
    return pa.concat_tables([pq.read_table(p, columns=columns) for p in paths]).to_pylist()


def load_records(bench: str, source: str = "hub", limit: int | None = None,
                 ids: list[str] | None = None, cache_dir: str | None = None) -> list[dict]:
    """limit: first N samples (for POPE: first N per split). ids: whitelist of sids."""
    keep = set(map(str, ids)) if ids is not None else None
    if source != "hub":
        recs = [r for r in pq.read_table(source).to_pylist() if r.get("bench", bench) == bench]
        if keep is not None:
            recs = [r for r in recs if r["sid"] in keep]
        return _limit(recs, bench, limit)

    main, extra = hub_paths(bench, cache_dir)
    if bench == "gqa":
        q = _read(main, ["id", "imageId", "question", "answer"])
        if keep is not None:
            q = [r for r in q if r["id"] in keep]
        q = _limit(q, bench, limit)
        need = {r["imageId"] for r in q}
        id2img = {}
        for b in pq.ParquetFile(extra[0]).iter_batches(batch_size=64):
            for r in b.to_pylist():
                if r["id"] in need:
                    id2img[r["id"]] = r["image"]["bytes"]
        return [dict(sid=str(r["id"]), category="", question=r["question"], answer=r["answer"],
                     answers=[r["answer"]], ocr_tokens=[], image_bytes=id2img[r["imageId"]]) for r in q]
    if bench == "pope":
        meta = [dict(sid=m["id"], category=m["category"]) for m in _read(main, ["id", "category"])
                if keep is None or m["id"] in keep]
        sel = {r["sid"] for r in _limit(meta, bench, limit)}
        recs = []
        for p in main:
            for b in pq.ParquetFile(p).iter_batches(batch_size=256):
                for r in b.to_pylist():
                    if r["id"] in sel:
                        recs.append(dict(sid=str(r["id"]), category=r["category"], question=r["question"],
                                         answer=r["answer"], answers=[r["answer"]], ocr_tokens=[],
                                         image_bytes=r["image"]["bytes"]))
        return recs
    if bench == "textvqa":
        meta = [str(m["question_id"]) for m in _read(main, ["question_id"])]
        sel = [s for s in meta if keep is None or s in keep]
        sel = set(sel[:limit] if limit else sel)
        recs = []
        for p in main:
            for b in pq.ParquetFile(p).iter_batches(batch_size=64):
                for r in b.to_pylist():
                    if str(r["question_id"]) in sel:
                        recs.append(dict(sid=str(r["question_id"]), category="", question=r["question"], answer="",
                                         answers=list(r["answers"]), ocr_tokens=list(r["ocr_tokens"] or []),
                                         image_bytes=r["image"]["bytes"]))
        return recs
    raise ValueError(bench)
