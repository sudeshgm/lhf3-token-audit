# RUNLOG

## 2026-10-19 17:xx IST — Step 1: repo setup + CPU validation (LHF3 CPU Agent, no GPU)
- Harness: custom minimal eval loop (not lmms-eval); see README "Harness".
- Data: downloaded lmms-lab parquet shards on the CPU box (GQA testdev_balanced instructions+images,
  POPE data/test-0000{0,1,2}, TextVQA validation-00000); built `data/smoke/{gqa,pope,textvqa}.parquet`
  (first 20 per benchmark; POPE first 20 per split = 60) with `scripts/make_smoke_subsets.py`.
  Note: `lmms-lab/*` dataset repos now redirect to `lmms-lab-encoder/*`; `datasets.load_dataset("lmms-lab/...")`
  still resolves (verified for GQA and POPE via `--source hub`).
- CPU env: python 3.13.5, torch 2.6.0+cpu, transformers 4.51.3, datasets 3.5.0.
- Tests: `pytest tests` → 20 passed, incl. end-to-end on a tiny random-weight LLaVA
  (`scripts/make_tiny_llava.py`, real llava-1.5-7b-hf processor/tokenizer, 576 visual tokens):
  LLM prefill length == n_text + k for k ∈ {576, 58, 6, 144}; our projector features == HF
  `get_image_features`; full-576 via our inputs_embeds path gives the same logits as HF's native pixel_values path.
- `scripts/smoke.sh` dry-run on CPU with the tiny model (LIMIT=2): all 9 runs complete, visual tokens to LLM = [576]/[58]/[58].
- Scores from the tiny model are meaningless (random weights). No GPU numbers yet.

## Pending
- GPU smoke (SMOKE_COMMANDS.md). A failed download / OOM = invalid run, not a no-go.

## 2026-10-19 Smoke test (user GPU: Azure T4 16GB, azureml_py38 env py3.10, fp16)
Commit 06a4c89. n=20 per bench (POPE 20/split). Model load 169 s from network storage.

| bench | arm | k | seed | vis_tok | score | s/sample | peak GiB |
|---|---|---|---|---|---|---|---|
| gqa | full | 576 | - | 576 | acc 0.700 | 0.522 | 13.55 |
| gqa | keep_last_k | 58 | - | 58 | acc 0.350 | 0.202 | 13.25 |
| gqa | random | 58 | 0 | 58 | acc 0.550 | 0.199 | 13.25 |
| pope | full | 576 | - | 576 | ran F1 .889/acc .900; pop .842/.850; adv .762/.750 | 0.479 | 13.55 |
| pope | keep_last_k | 58 | - | 58 | all splits F1 0.000/acc 0.500 | 0.178 | 13.25 |
| pope | random | 58 | 0 | 58 | ran .571/.700; pop .706/.750; adv .706/.750 | 0.178 | 13.25 |
| textvqa | full | 576 | - | 576 | acc 0.410 | 0.689 | 13.70 |
| textvqa | keep_last_k | 58 | - | 58 | acc 0.300 | 0.366 | 13.36 |
| textvqa | random | 58 | 0 | 58 | acc 0.400 | 0.348 | 13.36 |

Env fixes needed: torchvision==0.21.0, numpy<2, USE_TF=0, model on local-dir path.

## 2026-10-19 ~18:30 IST, Step 2A: LLaVA-1.5 protocol audit + sweep tooling (LHF3 CPU Agent, CPU only)
Reference: LLaVA's official `eval.zip` (HF mirror Xiaodong/LLaVA-1.5-Eval, sha256 2df36a33…), with question
files + llava-v1.5-13b answers committed under `third_party/llava_eval/`, and LLaVA main
`llava/eval/{m4c_evaluator,eval_textvqa,eval_pope,model_vqa_loader}.py`, `llava/conversation.py`, `llava/mm_utils.py`.
Full diff: `python scripts/audit_llava_protocol.py` (output below).

### Diff vs official, what was wrong and what was fixed
| item | step 1 (smoke, 06a4c89) | official LLaVA-1.5 | status |
|---|---|---|---|
| system prompt | "...curious **human**... the **human's** questions." | conv_vicuna_v1: "...curious **user**... the **user's** questions." | **FIXED** (prompts.SYSTEM) |
| TextVQA question case | lmms-lab raw, lowercase (`what is the brand…`) | capitalized (`What is the brand…`, = `str.capitalize()`) | **FIXED**: identical for 5000/5000 (before: 5/5000) |
| TextVQA OCR line | `\nReference OCR token: a, b, …` | same | already identical, 5000/5000. **The OCR tokens were reaching the prompt.** |
| lmms-lab ocr_tokens vs official file | — | — | identical, 5000/5000; row order identical (image ids) |
| TextVQA GT answers | lmms-lab `answers` | TextVQA_0.5.1_val.json | identical, 5000/5000 |
| suffix | `\nAnswer the question using a single word or phrase.` | same | OK (GQA 12578/12578 and POPE 8721/8721 matched prompts identical) |
| template | `{SYS} USER: <image>\n{q} ASSISTANT:` | same (sep " ", roles USER/ASSISTANT) | OK |
| tokenization | HF processor (fast tok, 576 expanded placeholders) | `tokenizer_image_token`, slow tokenizer, transformers 4.37.2 (liuhaotian/llava-v1.5-7b tokenizer) | identical ids (checked with a 4.37.2 venv) |
| image preproc | expand2square(image_mean) + CLIP 336 resize/crop | `image_aspect_ratio=pad`: same | OK (preprocessor config values identical to openai/clip-vit-large-patch14-336) |
| decoding | greedy, max_new_tokens 128 | temperature 0, num_beams 1, max_new_tokens 128 | OK |
| TextVQA scorer | own re-implementation of EvalAI processor | m4c `TextVQAAccuracyEvaluator` | **REPLACED by the verbatim copy** (`third_party/m4c_evaluator.py`). The old one lacked `word_tokenize` (the `'s` split, comma/? strip) and used a slightly different COMMA_STRIP regex; it gave the same 61.25 on the 13B answers. |
| GQA scorer | strip/lower/rstrip('.') on both sides | `convert_gqa_for_eval`: pred.rstrip('.').lower() == gold | aligned to the official one |
| POPE parser | LLaVA eval_pope.py | same | OK. 13B answers rescored: 87.32/86.30/84.45 (paper 87.3/86.1/84.2) |
| eval.py at 2d299b8 | `os.environ` lines placed before `from __future__` → **SyntaxError** | — | **FIXED**: env defaults moved into core.py |

Scorer validation on the official llava-v1.5-13b answers: TextVQA **61.25** (paper 61.3), GQA **63.31** (paper 63.3),
POPE mean F1 **86.02** (paper 85.9).
POPE data note: lmms-lab/POPE has 9000 rows (3×3000). LLaVA's file has 8910 (random: 2910). We keep lmms-lab's 9000 as pre-registered.

Three exact TextVQA prompts fed to the processor (OCR line present):
```
'A chat between a curious user and an artificial intelligence assistant. The assistant gives helpful, detailed, and polite answers to the user\'s questions. USER: <image>\nWhat is the brand of this camera?\nReference OCR token: DAKOTA, DIGITAL, Single-Use, Camera, Pire, digitat\nAnswer the question using a single word or phrase. ASSISTANT:'
'A chat between a curious user and an artificial intelligence assistant. The assistant gives helpful, detailed, and polite answers to the user\'s questions. USER: <image>\nWhat does the small white text spell?\nReference OCR token: D, RUPALCON, PALCON, COPENHAGEN\nAnswer the question using a single word or phrase. ASSISTANT:'
'A chat between a curious user and an artificial intelligence assistant. The assistant gives helpful, detailed, and polite answers to the user\'s questions. USER: <image>\nWhat kind of beer is this?\nReference OCR token: NINK, NK, BOWING, CC, STON, SUE, ED, Sublimely, SELF, ELF-RICHEE, swAaVd, KGy, ALE\nAnswer the question using a single word or phrase. ASSISTANT:'
```

### Smoke TextVQA 0.41: interpretation
- The OCR tokens were in the prompt (step-1 OCR line byte-identical to official for all 5000), so this is not the no-OCR (~46) regime.
- The first 20 val questions are hard: the official **13B** answers score **0.510** on them (0.621 on the first 200, 0.613 on all 5000).
  7B (58.2 overall) would be expected around 0.45–0.50 there, so 0.41 is 1–2 questions low. That is within noise (SE ≈ 0.1 at n=20), plus the two prompt deviations above.
- Smoke per-sample predictions were not pasted, so I could not rescore them. The 200-sample gate (≥ 0.53) decides.

### POPE keep_last k=58 F1=0 / acc 0.5
- The parser matches LLaVA eval_pope.py. Empty output maps to **yes**, so empty or odd outputs would give yes-ratio 1, F1 0.667, acc 0.5, *not* F1 0.
- F1=0 with acc 0.5 on balanced labels means every one of the 60 generations parsed as "no", i.e. contained "No"/"no"/"not" in its first sentence.
  That is the model answering "No" to every object question when it only sees the last 58 raster tokens (bottom ~2.4 rows of the 24×24 grid). It looks like real model behaviour, not a scoring bug.
  This is unconfirmed until the rerun: results now log `raw_generation`, `n_generated_tokens`, `prompt` per sample, plus `yes_ratio` and `empty_outputs` per split.

### Subsets (committed, seed 0, `scripts/make_subsets.py`)
- `subsets/gqa_5k.json` (5000 of 12578), `subsets/gqa_1k.json` (first 1000 of the same permutation, ⊂ gqa_5k)
- `subsets/textvqa_1k.json` (1000 of 5000); `subsets/pope_1k_per_split.json` (**1000 per split**, 3000 total, stratified)

### Sweep tooling
- `scripts/sweep.py` + `configs/sweep_step2.json`: one model load, resumable (skip finished JSONs, per-sample `.partial.jsonl`),
  in-process TextVQA gate (first 200, full-576, acc ≥ 0.53 else exit 2). Data comes from lmms-lab parquet via hf_hub_download into $HF_HOME.
- `scripts/summarize_step2.py`: table in pp (full | random-s0 | keep_last per k | seed min/max on 1k), POPE mean F1 over 3 splits + per-split, POPE yes-ratios.
- CPU validation: tiny random LLaVA, `--limit 3`: gate + all 27 configs ran, resume from a truncated partial file worked, summarize OK. `pytest tests`: 26 passed.
