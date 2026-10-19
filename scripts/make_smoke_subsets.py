"""Build normalized smoke subsets (first 20 samples per benchmark; POPE: first 20 per split)
from locally downloaded lmms-lab parquet shards. Output: data/smoke/{gqa,pope,textvqa}.parquet

Shards used (huggingface.co/datasets/lmms-lab/...):
  GQA     testdev_balanced_instructions/testdev-00000-of-00001.parquet
          testdev_balanced_images/testdev-00000-of-00001.parquet
  POPE    data/test-0000{0,1,2}-of-00003.parquet   (adversarial / popular / random)
  TextVQA data/validation-00000-of-00003.parquet
"""
import sys
import pyarrow as pa
import pyarrow.parquet as pq

src, out, N = sys.argv[1], sys.argv[2], int(sys.argv[3]) if len(sys.argv) > 3 else 20
SCHEMA = pa.schema([("bench", pa.string()), ("sid", pa.string()), ("category", pa.string()),
                    ("question", pa.string()), ("answer", pa.string()), ("answers", pa.list_(pa.string())),
                    ("ocr_tokens", pa.list_(pa.string())), ("image_bytes", pa.binary())])


def write(name, recs):
    pq.write_table(pa.Table.from_pylist(recs, schema=SCHEMA), f"{out}/{name}.parquet")
    print(name, len(recs))


q = pq.read_table(f"{src}/gqa_instr.parquet", columns=["id", "imageId", "question", "answer"]).slice(0, N).to_pylist()
need = {r["imageId"] for r in q}
im = {r["id"]: r["image"]["bytes"] for r in pq.read_table(f"{src}/gqa_images.parquet").to_pylist() if r["id"] in need}
write("gqa", [dict(bench="gqa", sid=r["id"], category="", question=r["question"], answer=r["answer"],
                   answers=[r["answer"]], ocr_tokens=[], image_bytes=im[r["imageId"]]) for r in q])

recs = []
for i in (2, 1, 0):  # random, popular, adversarial
    t = pq.read_table(f"{src}/pope_{i}.parquet").slice(0, N).to_pylist()
    recs += [dict(bench="pope", sid=r["id"], category=r["category"], question=r["question"], answer=r["answer"],
                  answers=[r["answer"]], ocr_tokens=[], image_bytes=r["image"]["bytes"]) for r in t]
write("pope", recs)

f = pq.ParquetFile(f"{src}/tvqa_0.parquet")
t = f.read_row_group(0).slice(0, N).to_pylist()
write("textvqa", [dict(bench="textvqa", sid=str(r["question_id"]), category="", question=r["question"], answer="",
                       answers=r["answers"], ocr_tokens=r["ocr_tokens"] or [], image_bytes=r["image"]["bytes"])
                  for r in t])
