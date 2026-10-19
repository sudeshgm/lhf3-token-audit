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
