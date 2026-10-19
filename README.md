# lhf3-token-audit — LHF3 Experiment -1: random / keep-last visual-token pruning audit

## Fixed design (pre-registered, do not change)
- **Model:** `llava-hf/llava-1.5-7b-hf`, frozen, fp16, greedy decoding, max_new_tokens per standard LLaVA eval prompts (128).
- **Pruning point:** drop visual tokens **after the projector, before the LLM** (576 tokens → k). Kept tokens stay in their original order.
- **Retention:** 1%, 5%, 10%, 25% of 576 → **k = 6, 29, 58, 144** (rounded). Plus a **full-576 baseline**.
- **Arms:** (a) **random** selection, seeds 0/1/2; (b) **keep-last-k** (last k tokens in raster order).
- **Benchmarks:** GQA testdev-balanced (accuracy); POPE random/popular/adversarial (F1 + acc); TextVQA val (VQA accuracy, standard LLaVA-1.5 prompt with OCR tokens if that is the published protocol; logged which one is used).
- **Preferred harness:** lmms-eval with lmms-lab HF datasets (lmms-lab/GQA, lmms-lab/POPE, lmms-lab/textvqa); if awkward, a minimal custom eval loop, stated clearly.

## Harness: custom minimal eval loop (NOT lmms-eval)
lmms-eval's `llava_hf` wrapper hands `pixel_values` to `model.generate`, and the image-feature merge
happens inside transformers (`masked_scatter` over the 576 placeholder ids, whose layout changed across
transformers versions). Pruning there means monkeypatching transformers internals *and* rewriting the
placeholder count inside lmms-eval's batching. The custom loop (`eval.py`) instead does the LLaVA
forward explicitly and lets us *assert* the token count the LLM receives. It uses the same lmms-lab
datasets, the official LLaVA-1.5 prompts (conv `vicuna_v1`) and the official LLaVA-1.5 scorers.

Per sample (batch 1):
1. image → `expand2square` with CLIP mean colour (LLaVA-1.5 `image_aspect_ratio=pad`) → HF CLIP processor (336px).
2. vision tower, hidden layer −2, drop CLS → 576 tokens → `multi_modal_projector`.
3. **`pruning.prune(feats, k, mode, seed)`** → [1, k, 4096], original order.
4. `splice.splice_visual`: the run of image-placeholder ids in `input_ids` is replaced by the k vectors
   → `inputs_embeds` of length n_text + k and an attention mask of the same length.
5. `model.generate(inputs_embeds=..., attention_mask=..., do_sample=False, max_new_tokens=128)` (no pixel_values, so HF does no image merge).
6. A forward pre-hook on the LLM decoder records the prefill length actually received; the script asserts
   `prefill_len − n_text == k` and logs `visual_tokens_to_llm` per sample.

Full-576 through this path is numerically identical to HF's native path (tested, `tests/test_pruning.py`).

## Prompts and scoring (LLaVA-1.5 protocol)
- Template: `"<system> USER: <image>\n{question}\nAnswer the question using a single word or phrase. ASSISTANT:"`, system = vicuna_v1.
- **TextVQA: WITH OCR tokens** (`{question}\nReference OCR token: t1, t2, ...\nAnswer the question using a single word or phrase.`),
  matching LLaVA-1.5's published `llava_textvqa_val_v051_ocr.jsonl`; tokens = `ocr_tokens` field of lmms-lab/textvqa.
  (lmms-eval's default is `ocr: false`; set `--textvqa_ocr 0` for that variant.) Logged as `protocol.textvqa_ocr` in every JSON.
- GQA: exact match after lowercase + strip trailing period (LLaVA `eval_gqa.py`).
- POPE: LLaVA `eval_pope.py` yes/no parsing; acc, precision, recall, F1, yes-ratio per split.
- TextVQA: EvalAI answer normalisation + 10-choose-9 VQA accuracy (m4c `TextVQAAccuracyEvaluator`).

## Random arm seeding
`random` with seed s ∈ {0,1,2}: each sample gets its own index set from `torch.randperm` seeded by
`sha256(f"{s}:{sample_id}")` (stable across machines), sorted to keep raster order. So each seed is an
independent random draw over the benchmark (not one fixed mask for all images).

## Layout
```
pruning.py        prune(feats, k, mode, seed): 'full' | 'random' | 'keep_last_k'
splice.py         replace image placeholders with k vectors (works for expanded or legacy single <image>)
eval.py           entrypoint: --bench {gqa,pope,textvqa} --mode {full,random,keep_last_k} --k --seed --limit --source --out
data.py           loaders: --source hub (lmms-lab) or a normalized parquet
prompts.py, metrics.py
configs/grid.json full pre-registered grid
data/smoke/       20-sample subsets (POPE: 20 per split), committed
results/          one JSON per arm × k × benchmark × seed: results/<bench>/<arm>_k<k>_seed<s|NA>.json
scripts/          smoke.sh, run_grid.sh, summarize.py, make_smoke_subsets.py, make_tiny_llava.py
tests/            CPU unit tests (+ tiny random LLaVA end-to-end if LHF3_TINY_MODEL is set)
RUNLOG.md, SMOKE_COMMANDS.md
```
Each JSON has: metrics, `visual_tokens_to_llm`, `sec_per_sample_mean`, `peak_vram_gib`
(`torch.cuda.max_memory_allocated`, includes weights), full protocol/env, per-sample predictions.

## Versions
Pinned in `requirements.txt` (transformers 4.51.3, torch 2.6.0). Model submodules are accessed version-agnostically
(`model.X` or `model.model.X`), and the splice handles both processor behaviours (576 expanded placeholders or one `<image>`).
