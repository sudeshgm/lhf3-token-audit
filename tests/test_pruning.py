import os
import sys

import pytest
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from pruning import keep_indices, prune, sample_seed  # noqa: E402
from splice import splice_visual  # noqa: E402
import metrics  # noqa: E402

N, D = 576, 8
KS = [6, 29, 58, 144]


def feats():
    # token i carries value i in every channel -> we can read back which tokens survived
    return torch.arange(N, dtype=torch.float32)[None, :, None].expand(2, N, D).clone()


def test_k_values_match_spec():
    assert [round(N * r) for r in (0.01, 0.05, 0.10, 0.25)] == KS


@pytest.mark.parametrize("k", KS)
def test_random_shape_order_unique(k):
    out = prune(feats(), k, "random", seed=0)
    assert out.shape == (2, k, D)
    ids = out[0, :, 0].long()
    assert torch.all(ids[1:] > ids[:-1])            # original order, no duplicates
    assert torch.equal(out[0], out[1])              # same index set across batch


@pytest.mark.parametrize("k", KS)
def test_random_deterministic_and_seed_dependent(k):
    a = keep_indices(N, k, "random", 0)
    assert torch.equal(a, keep_indices(N, k, "random", 0))
    assert not torch.equal(a, keep_indices(N, k, "random", 1))
    assert not torch.equal(a, keep_indices(N, k, "random", 2))


@pytest.mark.parametrize("k", KS)
def test_keep_last_k(k):
    out = prune(feats(), k, "keep_last_k")
    assert torch.equal(out[0, :, 0].long(), torch.arange(N - k, N))


def test_full_and_2d_and_errors():
    f = feats()
    assert torch.equal(prune(f, N, "full"), f)
    assert torch.equal(prune(f, N, "random", 0), f)
    assert prune(f[0], 58, "keep_last_k").shape == (58, D)
    with pytest.raises(ValueError):
        prune(f, 58, "topk_attn")
    with pytest.raises(ValueError):
        prune(f, 58, "random", None)
    with pytest.raises(ValueError):
        prune(f, 0, "keep_last_k")


def test_random_uniform_coverage():
    counts = torch.zeros(N)
    for s in range(400):
        counts[keep_indices(N, 58, "random", sample_seed(0, str(s)))] += 1
    assert counts.min() > 10 and counts.max() < 80   # expected ~40.3 per position


def test_sample_seed_stable():
    assert sample_seed(0, "abc") == sample_seed(0, "abc") != sample_seed(1, "abc")


@pytest.mark.parametrize("n_ph", [576, 1])  # expanded (new processor) and legacy single <image>
def test_splice(n_ph):
    IMG = 32000
    ids = torch.tensor([[1, 5, 6] + [IMG] * n_ph + [7, 8]])
    emb = torch.randn(1, ids.shape[1], D)
    vis = torch.randn(1, 58, D)
    e, m, info = splice_visual(ids, emb, vis, IMG)
    assert e.shape == (1, 5 + 58, D) and m.shape == (1, 63)
    assert torch.equal(e[0, 3:61], vis[0]) and torch.equal(e[0, :3], emb[0, :3])
    assert torch.equal(e[0, 61:], emb[0, -2:])
    assert info["n_visual_tokens"] == 58 and info["n_text_tokens"] == 5


def test_metrics():
    assert metrics.gqa_correct("Yes.", "yes") == 1.0
    assert metrics.pope_parse("No, there is not.") == "no"
    assert metrics.pope_parse("Yes") == "yes"
    m = metrics.pope_metrics(["yes", "no", "yes", "no"], ["yes", "no", "no", "yes"])
    assert m["acc"] == 0.5 and m["f1"] == 0.5
    gts = ["dakota"] * 6 + ["dakota digital"] * 3 + ["x"]
    assert metrics.textvqa_acc("Dakota", gts) == 1.0
    assert abs(metrics.textvqa_acc("dakota digital", gts) - 0.9) < 1e-9
    assert metrics.evalai_process("Two dogs!") == "2 dogs"


TINY = os.environ.get("LHF3_TINY_MODEL")


@pytest.mark.skipif(not TINY, reason="set LHF3_TINY_MODEL to a dir from scripts/make_tiny_llava.py")
def test_end_to_end_tiny_llava_cpu():
    """Real HF LLaVA code path with random tiny weights: the LLM must see exactly k visual tokens,
    and our projector features must equal HF's own get_image_features."""
    from transformers import AutoProcessor, LlavaForConditionalGeneration
    from PIL import Image
    import eval as ev
    import prompts

    proc = AutoProcessor.from_pretrained(TINY)
    model = LlavaForConditionalGeneration.from_pretrained(TINY, torch_dtype=torch.float32).eval()
    img = Image.new("RGB", (400, 300), (120, 30, 200))
    img = ev.expand2square(img, tuple(int(x * 255) for x in proc.image_processor.image_mean))
    enc = proc(images=img, text=prompts.full_prompt("Is there a cat?" + prompts.SHORT), return_tensors="pt")
    assert int((enc["input_ids"] == model.config.image_token_index).sum()) in (1, N)
    with torch.no_grad():
        f = ev.visual_features(model, enc["pixel_values"])
        assert f.shape[1] == N
        if hasattr(model, "get_image_features"):
            ref = model.get_image_features(pixel_values=enc["pixel_values"],
                                           vision_feature_layer=model.config.vision_feature_layer,
                                           vision_feature_select_strategy=model.config.vision_feature_select_strategy)
            ref = ref[0] if isinstance(ref, (list, tuple)) else ref
            assert torch.allclose(f, ref.reshape(f.shape), atol=1e-5)
        probe = ev.PrefillProbe(ev.find_decoder(model))
        for mode, k in [("full", N), ("random", 58), ("keep_last_k", 58), ("random", 6), ("keep_last_k", 144)]:
            kept = prune(f, k, mode, 0)
            e, m, info = splice_visual(enc["input_ids"], model.get_input_embeddings()(enc["input_ids"]),
                                       kept, model.config.image_token_index)
            probe.reset()
            out = model.generate(inputs_embeds=e, attention_mask=m, do_sample=False, max_new_tokens=3,
                                 pad_token_id=proc.tokenizer.pad_token_id)
            assert probe.seq_lens[0] == info["n_text_tokens"] + k, (mode, k, probe.seq_lens)
            assert out.shape[1] <= 3
        # full arm through our path == HF's native pixel_values path (same logits)
        e, m, _ = splice_visual(enc["input_ids"], model.get_input_embeddings()(enc["input_ids"]),
                                f, model.config.image_token_index)
        ours = model(inputs_embeds=e, attention_mask=m).logits[0, -1]
        native = model(**enc).logits[0, -1]
        assert torch.allclose(ours, native, atol=1e-4)
