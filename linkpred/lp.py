"""Link prediction on top of an unchanged backbone: cosine scores of the backbone's node outputs.

* Loss: symmetric InfoNCE with in-batch negatives (each positive pair against the other endpoints in its batch), so
  an update's negatives are its own roots and add no nodes. Temperature fixed at 0.1.
* Scoring: each query edge (u, v) is ranked against a shared pool of random present nodes, both ways (v among the
  pool for u, u among the pool for v). Reciprocal rank and Hits@20 are averaged over the two directions.
"""
from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import Tensor

from gist.models.registry import build_backbone


def build_model(name: str, in_dim: int, hidden: int, out_dim: int, heads: int):
    if name == "gat":
        return build_backbone("gat", in_dim, out_dim, hidden // heads, heads)
    return build_backbone(name, in_dim, out_dim, hidden, heads)


def embed(model, x: Tensor, sub, edge_weight: Tensor | None = None) -> Tensor:
    """Unit-norm outputs of the roots of a sampled computation graph."""
    if edge_weight is None:
        out = model(x[sub.n_id], sub.edge_index)[0]
    else:
        out = model(x[sub.n_id], sub.edge_index, edge_weight)[0]
    return F.normalize(out[: sub.num_roots], dim=-1)


def info_nce(zu: Tensor, zv: Tensor, temp: float = 0.1) -> Tensor:
    logits = zu @ zv.T / temp
    target = torch.arange(len(zu), device=zu.device)
    return 0.5 * (F.cross_entropy(logits, target) + F.cross_entropy(logits.T, target))


def rank_metrics(zu: Tensor, zv: Tensor, zp: Tensor, k: int = 20):
    """Per query edge: reciprocal rank and Hits@k, each averaged over the two directions. Ties count half."""
    pos = (zu * zv).sum(1, keepdim=True)
    out_rr, out_hit = 0.0, 0.0
    for a in (zu, zv):
        s = a @ zp.T
        rank = 1 + (s > pos).sum(1) + 0.5 * (s == pos).sum(1)
        out_rr = out_rr + 0.5 / rank
        out_hit = out_hit + 0.5 * (rank <= k).float()
    return out_rr, out_hit
