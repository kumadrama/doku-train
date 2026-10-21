"""Parameter-free target attention on CPU tensors; no projections or learned scales."""

from __future__ import annotations

import torch


def history_attention(
    query: torch.Tensor, history: torch.Tensor, valid: torch.Tensor, attn_scale: float
) -> torch.Tensor:
    """Masked softmax(dot(q, h) / scale) weighted sum; empty histories return zero."""
    scores = (query.unsqueeze(1) * history).sum(-1) / attn_scale
    scores = scores.masked_fill(~valid, -torch.inf)
    # Softmax of all -inf is NaN. Empty rows use finite logits then zero weights.
    scores = torch.where(valid.any(dim=1, keepdim=True), scores, torch.zeros_like(scores))
    weights = torch.softmax(scores, dim=1).masked_fill(~valid, 0.0)
    result: torch.Tensor = (history * weights.unsqueeze(-1)).sum(1)
    return result
