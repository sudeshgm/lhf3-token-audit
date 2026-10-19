"""Build LLM inputs_embeds with exactly k visual tokens (model-version agnostic).

Works whether the HF processor already expanded <image> into 576 placeholder ids
(transformers >= 4.47 with patch_size set) or left a single <image> id (legacy).
The contiguous run of image-token ids is replaced by the k pruned feature vectors,
so the LLM sequence length == n_text_tokens + k and the attention mask has length
n_text_tokens + k as well.
"""
from __future__ import annotations

import torch


def splice_visual(input_ids: torch.Tensor, text_embeds: torch.Tensor,
                  vis_feats: torch.Tensor, image_token_id: int):
    """input_ids: [1, L]; text_embeds: [1, L, D] (embeddings of input_ids);
    vis_feats: [1, k, D]. Returns (inputs_embeds [1, L', D], attention_mask [1, L'],
    info dict)."""
    assert input_ids.shape[0] == 1, "batch size 1 only"
    pos = (input_ids[0] == image_token_id).nonzero().flatten()
    if pos.numel() == 0:
        raise ValueError("no image token in input_ids")
    start, end = int(pos[0]), int(pos[-1]) + 1
    if end - start != pos.numel():
        raise ValueError("image placeholder tokens are not contiguous")
    vis = vis_feats.to(text_embeds.dtype)
    embeds = torch.cat([text_embeds[:, :start], vis, text_embeds[:, end:]], dim=1)
    mask = torch.ones(embeds.shape[:2], dtype=torch.long, device=embeds.device)
    n_text = input_ids.shape[1] - pos.numel()
    info = {"n_placeholders_in": int(pos.numel()), "n_text_tokens": int(n_text),
            "n_visual_tokens": int(vis.shape[1]), "seq_len": int(embeds.shape[1]),
            "image_start": start}
    assert info["seq_len"] == n_text + info["n_visual_tokens"]
    return embeds, mask, info
