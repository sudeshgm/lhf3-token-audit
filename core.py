"""Shared model/eval core for eval.py (one config) and scripts/sweep.py (many configs, one model load).

Pipeline per sample (batch size 1, greedy), matching LLaVA-1.5 model_vqa_loader:
  image -> expand2square(image_mean) -> CLIP processor 336 -> vision tower layer -2, drop CLS (576)
        -> multi_modal_projector -> prune(feats, k, mode, seed)   <-- PRUNING POINT
        -> splice k vectors in place of the image placeholder run -> LLM (inputs_embeds) -> generate
"""
from __future__ import annotations

import os

os.environ.setdefault("USE_TF", "0")
os.environ.setdefault("TRANSFORMERS_NO_TF", "1")
os.environ.setdefault("USE_FLAX", "0")

import io
import json
import platform
import subprocess
import sys
import time

import torch
from PIL import Image

import metrics
import prompts
from data import POPE_CATS
from pruning import prune, sample_seed
from splice import splice_visual

N_VIS = 576
MODEL_ID = "llava-hf/llava-1.5-7b-hf"
REPO = os.path.dirname(os.path.abspath(__file__))


def git_rev() -> str | None:
    try:
        return subprocess.check_output(["git", "-C", REPO, "rev-parse", "--short", "HEAD"],
                                       stderr=subprocess.DEVNULL).decode().strip()
    except Exception:
        return None


def get_sub(model, name):
    """Version-agnostic access (transformers <4.52: model.X; >=4.52: model.model.X / properties)."""
    for obj in (model, getattr(model, "model", None)):
        if obj is not None and getattr(obj, name, None) is not None:
            return getattr(obj, name)
    raise AttributeError(name)


def find_decoder(model):
    lm = get_sub(model, "language_model")
    return lm.model if hasattr(lm, "model") and hasattr(lm.model, "layers") else lm


def expand2square(img: Image.Image, bg) -> Image.Image:
    """Identical to llava/mm_utils.py expand2square."""
    w, h = img.size
    if w == h:
        return img
    if w > h:
        out = Image.new(img.mode, (w, w), bg)
        out.paste(img, (0, (w - h) // 2))
        return out
    out = Image.new(img.mode, (h, h), bg)
    out.paste(img, ((h - w) // 2, 0))
    return out


@torch.no_grad()
def visual_features(model, pixel_values):
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
        self.seq_lens.append(int(e.shape[1] if e is not None else (ids.shape[1] if ids is not None else -1)))

    def reset(self):
        self.seq_lens = []


class Runner:
    def __init__(self, model_path=MODEL_ID, device=None, dtype="float16", max_new_tokens=128,
                 textvqa_ocr=prompts.TEXTVQA_OCR_DEFAULT, n_vis=N_VIS):
        import transformers
        from transformers import AutoProcessor, LlavaForConditionalGeneration
        self.transformers_version = transformers.__version__
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.dtype_name, self.dtype = dtype, getattr(torch, dtype)
        self.model_path, self.max_new_tokens, self.textvqa_ocr, self.n_vis = model_path, max_new_tokens, textvqa_ocr, n_vis
        t0 = time.time()
        self.processor = AutoProcessor.from_pretrained(model_path)
        self.model = LlavaForConditionalGeneration.from_pretrained(model_path, torch_dtype=self.dtype).to(self.device).eval()
        for p in self.model.parameters():
            p.requires_grad_(False)
        self.load_s = time.time() - t0
        self.tok = self.processor.tokenizer
        self.image_token_id = self.model.config.image_token_index
        self.bg = tuple(int(x * 255) for x in self.processor.image_processor.image_mean)
        self.probe = PrefillProbe(find_decoder(self.model))
        self.cuda = self.device.startswith("cuda")
        if self.cuda:
            torch.cuda.reset_peak_memory_stats()

    def build_prompt(self, bench, rec):
        return prompts.full_prompt(prompts.question_text(bench, rec, self.textvqa_ocr))

    @torch.no_grad()
    def run_sample(self, bench, rec, mode, k, seed):
        ts = time.time()
        img = expand2square(Image.open(io.BytesIO(rec["image_bytes"])).convert("RGB"), self.bg)
        prompt = self.build_prompt(bench, rec)
        enc = self.processor(images=img, text=prompt, return_tensors="pt")
        input_ids = enc["input_ids"].to(self.device)
        pixel_values = enc["pixel_values"].to(self.device, self.dtype)
        feats = visual_features(self.model, pixel_values)
        assert feats.shape[1] == self.n_vis, f"expected {self.n_vis} visual tokens, got {feats.shape[1]}"
        s = sample_seed(seed, rec["sid"]) if mode == "random" else None
        kept = prune(feats, k, mode, s)
        text_emb = self.model.get_input_embeddings()(input_ids)
        embeds, mask, info = splice_visual(input_ids, text_emb, kept, self.image_token_id)
        self.probe.reset()
        pad = self.tok.pad_token_id if self.tok.pad_token_id is not None else self.tok.eos_token_id
        gen = self.model.generate(inputs_embeds=embeds, attention_mask=mask, do_sample=False, num_beams=1,
                                  max_new_tokens=self.max_new_tokens, temperature=None, top_p=None,
                                  pad_token_id=pad, use_cache=True)
        if self.cuda:
            torch.cuda.synchronize()
        dt = time.time() - ts
        prefill = self.probe.seq_lens[0]
        assert prefill == info["seq_len"], (prefill, info)
        n_vis_llm = prefill - info["n_text_tokens"]
        assert n_vis_llm == k, (n_vis_llm, k)
        raw = self.tok.decode(gen[0], skip_special_tokens=True)
        pred = raw.strip()
        if bench == "gqa":
            parsed = metrics.gqa_norm(pred); score = metrics.gqa_correct(pred, rec["answer"])
        elif bench == "pope":
            parsed = metrics.pope_parse(pred); score = float(parsed == rec["answer"].lower())
        else:
            parsed = metrics.evalai_process(pred); score = metrics.textvqa_acc(pred, rec["answers"])
        return dict(sid=rec["sid"], category=rec["category"], prompt=prompt, gt=rec["answers"],
                    raw_generation=raw, pred=pred, parsed=parsed, score=score,
                    n_generated_tokens=int(gen.shape[1]), visual_tokens_to_llm=n_vis_llm, llm_prefill_len=prefill,
                    n_text_tokens=info["n_text_tokens"], placeholders_from_processor=info["n_placeholders_in"], sec=dt)

    def env(self):
        return dict(torch=torch.__version__, transformers=self.transformers_version, python=platform.python_version(),
                    gpu=torch.cuda.get_device_name(0) if self.cuda else None, git=git_rev(), argv=sys.argv)

    def peak_gib(self):
        return torch.cuda.max_memory_allocated() / 2**30 if self.cuda else None


def aggregate(bench, rows):
    res = {}
    if bench == "pope":
        for c in POPE_CATS:
            sub = [x for x in rows if x["category"] == c]
            if sub:
                res[c] = metrics.pope_metrics([x["parsed"] for x in sub], [x["gt"][0] for x in sub])
                res[c]["empty_outputs"] = sum(1 for x in sub if not x["pred"])
        cats = [c for c in POPE_CATS if c in res]
        if cats:
            res["mean_f1"] = sum(res[c]["f1"] for c in cats) / len(cats)
            res["mean_acc"] = sum(res[c]["acc"] for c in cats) / len(cats)
            res["yes_ratio"] = {c: res[c]["yes_ratio"] for c in cats}
    else:
        res["acc"] = sum(x["score"] for x in rows) / max(len(rows), 1)
    res["n_empty_outputs"] = sum(1 for x in rows if not x["pred"])
    return res


def summary_dict(runner, bench, mode, k, seed, rows, extra_protocol=None):
    times = [x["sec"] for x in rows]
    proto = dict(model=runner.model_path, dtype=runner.dtype_name, decoding="greedy", max_new_tokens=runner.max_new_tokens,
                 pruning_point="after multi_modal_projector, before LLM", order="original raster order",
                 random_seeding="per-sample: sha256(f'{seed}:{sid}')" if mode == "random" else None,
                 image_preproc="expand2square(image_mean) + CLIP processor 336 (LLaVA-1.5 pad)",
                 conv="vicuna_v1 (system: 'curious user')", system_prompt=prompts.SYSTEM,
                 textvqa_ocr=bool(runner.textvqa_ocr) if bench == "textvqa" else None,
                 textvqa_question="str.capitalize()" if bench == "textvqa" else None,
                 scorer={"gqa": "rstrip('.').lower() == gold", "pope": "LLaVA eval_pope.py",
                         "textvqa": "LLaVA m4c TextVQAAccuracyEvaluator (verbatim)"}[bench],
                 harness="custom minimal loop (not lmms-eval)")
    proto.update(extra_protocol or {})
    return dict(bench=bench, arm=mode, k=k, retention=k / runner.n_vis, seed=seed if mode == "random" else None,
                n=len(rows), metrics=aggregate(bench, rows), visual_tokens_to_llm=sorted({x["visual_tokens_to_llm"] for x in rows}),
                sec_per_sample_mean=sum(times) / max(len(times), 1),
                sec_per_sample_excl_first=(sum(times[1:]) / (len(times) - 1)) if len(times) > 1 else None,
                peak_vram_gib=runner.peak_gib(), model_load_s=runner.load_s, protocol=proto, env=runner.env(),
                samples=rows)


def write_json(path, obj):
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    tmp = path + ".tmp"
    with open(tmp, "w") as f:
        json.dump(obj, f, indent=1)
    os.replace(tmp, path)


def short(summary):
    return {k: summary[k] for k in ("bench", "arm", "k", "seed", "n", "metrics", "visual_tokens_to_llm",
                                    "sec_per_sample_mean", "peak_vram_gib")}
