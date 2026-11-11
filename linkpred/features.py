"""Node features for graphs that have none, computed on the base snapshot G0 only (no future edges).

36 columns, standardised: log degree; mean and max log degree of the neighbours; log PageRank (x n); and a 32-dim
FastRP embedding (Chen et al., CIKM 2019: a sparse random projection propagated 1-3 hops, ``(D^-1 A)^k R``). Nodes
isolated in G0 get zeros (after standardisation, the column means). Features stay fixed while the stream runs.
"""
from __future__ import annotations

import torch

from .stream import LinkPlan


def structural_features(plan: LinkPlan, device, dim: int = 32, seed: int = 0) -> torch.Tensor:
    dev = torch.device(device)
    n = plan.num_nodes
    b = plan.key < 0
    lo, hi = plan.edges[0][b].to(dev), plan.edges[1][b].to(dev)
    src, dst = torch.cat([lo, hi]), torch.cat([hi, lo])
    deg = torch.bincount(src, minlength=n).float()
    ld = torch.log1p(deg)
    nb_mean = torch.zeros(n, device=dev).index_add_(0, dst, ld[src]) / deg.clamp(min=1)
    nb_max = torch.zeros(n, device=dev).scatter_reduce_(0, dst, ld[src], reduce="amax", include_self=True)

    inv = 1.0 / deg.clamp(min=1)
    pr = torch.full((n,), 1.0 / n, device=dev)
    for _ in range(30):
        dangling = pr[deg == 0].sum()
        pr = 0.15 / n + 0.85 * (torch.zeros(n, device=dev).index_add_(0, dst, (pr * inv)[src]) + dangling / n)
    lpr = torch.log(pr * n)

    g = torch.Generator(device=dev).manual_seed(seed)
    R = (torch.rand(n, dim, device=dev, generator=g) * 6).floor() - 2.5          # {-2.5..2.5}
    R = torch.where(R.abs() > 2, torch.sign(R), torch.zeros_like(R))              # sparse +-1, density 1/3
    E, acc = R, torch.zeros_like(R)
    for _ in range(3):
        E = torch.zeros_like(E).index_add_(0, dst, E[src]) * inv[:, None]       # D^-1 A E
        E = torch.nn.functional.normalize(E, dim=1)
        acc = acc + E
    X = torch.cat([ld[:, None], nb_mean[:, None], nb_max[:, None], lpr[:, None], acc], dim=1)
    live = deg > 0
    mu, sd = X[live].mean(0), X[live].std(0).clamp(min=1e-6)
    X = (X - mu) / sd
    X[~live] = 0.0
    return X.contiguous()
