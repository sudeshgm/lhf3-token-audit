#!/usr/bin/env python
"""LHF3 Experiment -1 eval entrypoint (minimal custom loop, NOT lmms-eval; see README).

Pipeline per sample (batch size 1, greedy):
  image -> pad-to-square (LLaVA-1.5 'pad') -> CLIP ViT-L/336 -> layer -2, drop CLS (576 tokens)
        -> multi_modal_projector -> prune(feats, k, mode, seed)   <-- PRUNING POINT
        -> splice k vectors in place of the image placeholder run -> LLM (inputs_embeds) -> generate
A forward pre-hook on the LLM decoder records the prefill sequence length actually fed to the
LLM and asserts it equals n_text_tokens + k, so the logged visual-token count is verified.

Example:
  python eval.py --bench gqa --mode random --k 58 --seed 0 --limit 20 \
      --source data/smoke/gqa.parquet --out results/smoke/gqa/random_k58_seed0.json
"""
from __future__ import annotations

import argparse
import io
import json
import os
import platform
import sys
import time

import torch
from PIL import Image

import metrics
import prompts
from data import POPE_CATS, load_records
from pruning import prune, sample_seed
from splice import splice_visual

N_VIS = 576
MODEL_ID = "llava-hf/llava-1.5-7b-hf"


def get_sub(model, name):
    """Version-agnostic access (transformers <4.52: model.X; >=4.52: model.model.X / properties)."""
    for obj in (model, getattr(model, "model", None)):
        if obj is not None and getattr(obj, name, None) is not None:
            return getattr(obj, name)
    raise AttributeError(name)


def find_decoder(model):
    """The LLM transformer stack (receives inputs_embeds; no image handling inside)."""
    lm = get_sub(model, "language_model")
    return lm.model if hasattr(lm, "model") and hasattr(lm.model, "layers") else lm


def expand2square(img: Image.Image, bg) -> Image.Image:
    w, h = img.size
    if w == h:
        return img
    s = max(w, h)
    out = Image.new(img.mode, (s, s), bg)
    out.paste(img, ((s - w) // 2, (s - h) // 2))
    return out


@torch.no_grad()
def visual_features(model, pixel_values):
    """CLIP features after the projector: [1, 576, D_llm]."""
    cfg = model.config
    vt = get_sub(model, "vision_tower")
    proj = get_sub(model, "multi_modal_projector")
    out = vt(pixel_values, output_hidden_states=True)
    layer = cfg.vision_feature_layer
    h = out.hidden_states[layer] if isinstance(layer, int) else torch.cat(
        [out.hidden_states[i] for i in layer], dim=-1)
    if cfg.vision_feature_select_strategy == "default":
        h = h[:, 1:]
    return proj(h)


class PrefillProbe:
    def __init__(self, decoder):
        self.seq_lens = []
        self.h = decoder.register_forward_pre_hook(self._hook, with_kwargs=True)

    def _hook(self, mod, args, kwargs):
        e = kwargs.get("inputs_embeds")
        ids = kwargs.get("input_ids")
        n = e.shape[1] if e is not None else (ids.shape[1] if ids is not None else -1)
        self.seq_lens.append(int(n))

    def reset(self):
        self.seq_lens = []


def parse():
    p = argparse.ArgumentParser()
    p.add_argument("--bench", required=True, choices=["gqa", "pope", "textvqa"])
    p.add_argument("--mode", required=True, choices=["full", "random", "keep_last_k"])
    p.add_argument("--k", type=int, default=N_VIS)
    p.add_argument("--seed", type=int, default=0, help="random arm only (0/1/2)")
    p.add_argument("--limit", type=int, default=None, help="first N samples (POPE: N per split)")
    p.add_argument("--source", default="hub", help="'hub' or a normalized parquet (data/smoke/*.parquet)")
    p.add_argument("--out", default=None)
    p.add_argument("--model", default=MODEL_ID)
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--dtype", default="float16", choices=["float16", "bfloat16", "float32"])
    p.add_argument("--max_new_tokens", type=int, default=128)
    p.add_argument("--textvqa_ocr", type=int, default=int(prompts.TEXTVQA_OCR_DEFAULT))
    p.add_argument("--n_vis", type=int, default=N_VIS, help="expected visual tokens before pruning")
    p.add_argument("--print_every", type=int, default=1)
    return p.parse_args()


def main():
    a = parse()
    if a.mode == "full":
        a.k = a.n_vis
    seed_tag = f"seed{a.seed}" if a.mode == "random" else "seedNA"
    if a.out is None:
        a.out = f"results/{a.bench}/{a.mode}_k{a.k}_{seed_tag}.json"
    os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True)

    import transformers
    from transformers import AutoProcessor, LlavaForConditionalGeneration

    dtype = getattr(torch, a.dtype)
    t0 = time.time()
    processor = AutoProcessor.from_pretrained(a.model)
    model = LlavaForConditionalGeneration.from_pretrained(a.model, torch_dtype=dtype).to(a.device).eval()
    for prm in model.parameters():
        prm.requires_grad_(False)  # frozen
    load_s = time.time() - t0
    tok = processor.tokenizer
    image_token_id = model.config.image_token_index
    bg = tuple(int(x * 255) for x in processor.image_processor.image_mean)
    probe = PrefillProbe(find_decoder(model))
    cuda = a.device.startswith("cuda")
    if cuda:
        torch.cuda.reset_peak_memory_stats()

    recs = load_records(a.bench, a.source, a.limit)
    print(f"[lhf3] bench={a.bench} mode={a.mode} k={a.k} seed={a.seed if a.mode=='random' else 'NA'} "
          f"n={len(recs)} source={a.source} model={a.model} dtype={a.dtype} device={a.device} "
          f"textvqa_ocr={bool(a.textvqa_ocr)} load_s={load_s:.1f}", flush=True)

    rows, times, vis_counts = [], [], []
    for i, r in enumerate(recs):
        ts = time.time()
        img = expand2square(Image.open(io.BytesIO(r["image_bytes"])).convert("RGB"), bg)
        qtext = prompts.question_text(a.bench, r, bool(a.textvqa_ocr))
        enc = processor(images=img, text=prompts.full_prompt(qtext), return_tensors="pt")
        input_ids = enc["input_ids"].to(a.device)
        pixel_values = enc["pixel_values"].to(a.device, dtype)
        with torch.no_grad():
            feats = visual_features(model, pixel_values)              # [1, 576, D]
            assert feats.shape[1] == a.n_vis, f"expected {a.n_vis} visual tokens, got {feats.shape[1]}"
            s = sample_seed(a.seed, r["sid"]) if a.mode == "random" else None
            kept = prune(feats, a.k, a.mode, s)                        # [1, k, D]
            text_emb = model.get_input_embeddings()(input_ids)
            embeds, mask, info = splice_visual(input_ids, text_emb, kept, image_token_id)
            probe.reset()
            gen = model.generate(inputs_embeds=embeds, attention_mask=mask, do_sample=False, num_beams=1,
                                 max_new_tokens=a.max_new_tokens, temperature=None, top_p=None,
                                 pad_token_id=tok.pad_token_id if tok.pad_token_id is not None else tok.eos_token_id)
        if cuda:
            torch.cuda.synchronize()
        dt = time.time() - ts
        prefill = probe.seq_lens[0]
        assert prefill == info["seq_len"], (prefill, info)
        n_vis_llm = prefill - info["n_text_tokens"]
        assert n_vis_llm == a.k, (n_vis_llm, a.k)
        pred = tok.decode(gen[0], skip_special_tokens=True).strip()

        if a.bench == "gqa":
            score = metrics.gqa_correct(pred, r["answer"]); parsed = pred
        elif a.bench == "pope":
            parsed = metrics.pope_parse(pred); score = float(parsed == r["answer"].lower())
        else:
            score = metrics.textvqa_acc(pred, r["answers"]); parsed = pred
        times.append(dt); vis_counts.append(n_vis_llm)
        rows.append(dict(sid=r["sid"], category=r["category"], question=r["question"], gt=r["answers"],
                         pred=pred, parsed=parsed, score=score, visual_tokens_to_llm=n_vis_llm,
                         llm_prefill_len=prefill, n_text_tokens=info["n_text_tokens"],
                         placeholders_from_processor=info["n_placeholders_in"], sec=dt))
        if a.print_every and i % a.print_every == 0:
            print(f"[{i+1}/{len(recs)}] vis_tokens_to_LLM={n_vis_llm} prefill_len={prefill} "
                  f"pred={pred[:40]!r} gt={r['answers'][:1]} score={score:.2f} {dt:.2f}s", flush=True)

    res = {}
    if a.bench == "pope":
        for c in POPE_CATS:
            sub = [x for x in rows if x["category"] == c]
            if sub:
                res[c] = metrics.pope_metrics([x["parsed"] for x in sub], [x["gt"][0] for x in sub])
        if res:
            res["mean_f1"] = sum(res[c]["f1"] for c in POPE_CATS if c in res) / sum(c in res for c in POPE_CATS)
            res["mean_acc"] = sum(res[c]["acc"] for c in POPE_CATS if c in res) / sum(c in res for c in POPE_CATS)
    else:
        res["acc"] = sum(x["score"] for x in rows) / max(len(rows), 1)
    peak = torch.cuda.max_memory_allocated() / 2**30 if cuda else None
    summary = dict(
        bench=a.bench, arm=a.mode, k=a.k, retention=a.k / a.n_vis,
        seed=a.seed if a.mode == "random" else None, n=len(rows), metrics=res,
        visual_tokens_to_llm=sorted(set(vis_counts)),
        sec_per_sample_mean=sum(times) / max(len(times), 1),
        sec_per_sample_excl_first=(sum(times[1:]) / (len(times) - 1)) if len(times) > 1 else None,
        peak_vram_gib=peak, model_load_s=load_s,
        protocol=dict(model=a.model, dtype=a.dtype, decoding="greedy", max_new_tokens=a.max_new_tokens,
                      pruning_point="after multi_modal_projector, before LLM", order="original raster order",
                      random_seeding="per-sample: sha256(f'{seed}:{sid}')" if a.mode == "random" else None,
                      image_preproc="expand2square(image_mean) + CLIP processor 336",
                      conv="vicuna_v1 (LLaVA-1.5 eval)", textvqa_ocr=bool(a.textvqa_ocr) if a.bench == "textvqa" else None,
                      harness="custom minimal loop (not lmms-eval)", source=a.source, limit=a.limit),
        env=dict(torch=torch.__version__, transformers=transformers.__version__, python=platform.python_version(),
                 gpu=torch.cuda.get_device_name(0) if cuda else None, argv=sys.argv),
        samples=rows,
    )
    with open(a.out, "w") as f:
        json.dump(summary, f, indent=1)
    print("[lhf3] SUMMARY " + json.dumps({k: summary[k] for k in (
        "bench", "arm", "k", "seed", "n", "metrics", "visual_tokens_to_llm", "sec_per_sample_mean",
        "sec_per_sample_excl_first", "peak_vram_gib")}), flush=True)
    print(f"[lhf3] wrote {a.out}")


if __name__ == "__main__":
    main()
