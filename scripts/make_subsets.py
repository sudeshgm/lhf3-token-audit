"""Fixed evaluation subsets (seed 0), committed under subsets/.

- gqa_5k:  5000 question ids from GQA testdev-balanced (12578), first 5000 of rng(0).permutation.
- gqa_1k:  first 1000 of the SAME permutation (so gqa_1k ⊂ gqa_5k).
- textvqa_1k: first 1000 of rng(0).permutation over the 5000 TextVQA val question_ids.
- pope_1k_per_split: 1000 per split (random/popular/adversarial; 3000 total), each the first 1000
  of rng(0).permutation over that split's 3000 rows. Ids are lmms-lab/POPE `id` (0..8999).
The random arm's per-sample seed depends only on (seed, sid), so seed-0 results on a 1k subset are
exactly the seed-0 full-run results restricted to those ids.
Usage: python scripts/make_subsets.py [hub|LOCAL_DIR_WITH_PARQUETS]
"""
import json
import os
import sys

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from data import POPE_CATS, hub_paths  # noqa: E402

src = sys.argv[1] if len(sys.argv) > 1 else "hub"
OUT = "subsets"


def paths(bench):
    if src == "hub":
        return hub_paths(bench)[0]
    local = {"gqa": ["gqa_instr.parquet"], "pope": [f"pope_{i}.parquet" for i in range(3)],
             "textvqa": [f"tvqa_{i}.parquet" for i in range(3)]}[bench]
    return [os.path.join(src, f) for f in local]


def read(bench, cols):
    return pa.concat_tables([pq.read_table(p, columns=cols) for p in paths(bench)]).to_pylist()


def dump(name, ids, desc, total):
    json.dump({"name": name, "description": desc, "seed": 0, "n": len(ids), "population": total, "ids": ids},
              open(f"{OUT}/{name}.json", "w"), indent=0)
    print(name, len(ids))


os.makedirs(OUT, exist_ok=True)
g = [r["id"] for r in read("gqa", ["id"])]
perm = np.random.default_rng(0).permutation(len(g))
order = {sid: i for i, sid in enumerate(g)}
pick = lambda n: sorted([g[i] for i in perm[:n]], key=order.get)
dump("gqa_5k", pick(5000), "GQA testdev-balanced, 5000 ids, first 5000 of default_rng(0).permutation", len(g))
dump("gqa_1k", pick(1000), "GQA testdev-balanced, first 1000 of the same permutation (subset of gqa_5k)", len(g))

t = [str(r["question_id"]) for r in read("textvqa", ["question_id"])]
perm = np.random.default_rng(0).permutation(len(t))
dump("textvqa_1k", [t[i] for i in sorted(perm[:1000])], "TextVQA val, first 1000 of default_rng(0).permutation", len(t))

p = read("pope", ["id", "category"])
ids = []
for c in POPE_CATS:
    cs = [r["id"] for r in p if r["category"] == c]
    perm = np.random.default_rng(0).permutation(len(cs))
    ids += [cs[i] for i in sorted(perm[:1000])]
dump("pope_1k_per_split", ids, "POPE, 1000 per split (random/popular/adversarial = 3000 total), "
     "first 1000 of default_rng(0).permutation within each split", len(p))
