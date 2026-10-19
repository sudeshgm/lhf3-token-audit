"""Visual-token pruning for LHF3 Experiment -1.

Pruning point: AFTER the multi-modal projector, BEFORE the LLM.
Input feats: [B, N, D] projected visual tokens (N = 576 for llava-1.5-7b-hf).
Kept tokens are always returned in their ORIGINAL (raster) order.
"""
from __future__ import annotations

import torch

MODES = ("full", "random", "keep_last_k")


def keep_indices(n: int, k: int, mode: str, seed: int | None = None) -> torch.Tensor:
    """Return sorted (ascending) LongTensor of the k token indices kept out of n."""
    if mode not in MODES:
        raise ValueError(f"unknown mode {mode!r}; expected one of {MODES}")
    if mode == "full" or k >= n:
        return torch.arange(n)
    if k < 1:
        raise ValueError(f"k must be >= 1, got {k}")
    if mode == "keep_last_k":
        return torch.arange(n - k, n)
    # random: k distinct indices, sorted so raster order is preserved
    if seed is None:
        raise ValueError("mode='random' requires an integer seed")
    g = torch.Generator(device="cpu").manual_seed(int(seed))
    idx = torch.randperm(n, generator=g)[:k]
    return torch.sort(idx).values


def prune(feats: torch.Tensor, k: int, mode: str, seed: int | None = None) -> torch.Tensor:
    """Prune projected visual tokens.

    feats: [B, N, D] or [N, D]. Returns [B, k, D] (or [k, D]) with original order kept.
    mode: 'full' (no-op), 'random' (seeded uniform sample w/o replacement, sorted),
          'keep_last_k' (last k tokens in raster order).
    The same index set is applied to every item in the batch.
    """
    squeeze = feats.dim() == 2
    if squeeze:
        feats = feats.unsqueeze(0)
    if feats.dim() != 3:
        raise ValueError(f"feats must be [B,N,D] or [N,D], got {tuple(feats.shape)}")
    n = feats.shape[1]
    idx = keep_indices(n, k, mode, seed).to(feats.device)
    out = feats.index_select(1, idx)
    return out.squeeze(0) if squeeze else out


def sample_seed(base_seed: int, sample_key: str) -> int:
    """Deterministic per-sample seed for the random arm (stable across runs/machines).

    Each (base_seed, sample) pair gets its own random index set, so arms with
    seeds 0/1/2 are three independent random draws over the whole benchmark.
    """
    import hashlib

    h = hashlib.sha256(f"{base_seed}:{sample_key}".encode()).hexdigest()
    return int(h[:15], 16)
