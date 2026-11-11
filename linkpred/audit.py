"""C4 for link prediction: an update's gradient on its endpoints' induced L-hop subgraph equals the whole graph's.

The loss of an update depends on the outputs at its endpoints only (in-batch negatives are endpoints too), and an
L-layer GNN's output at a node depends only on its L-hop neighbourhood. So the induced L-hop subgraph of the
endpoints gives the same gradient as the whole visible graph (minus the update's own edges). One hop short does not.
Checked in float64 on CPU; GCN takes its normalisation from the parent graph (``gist.models.exact``).
"""
from __future__ import annotations

import copy

import torch
import torch.nn.functional as F

from gist.models.exact import parent_gcn_weights, weighted_subgraph

from .lp import info_nce


def audit_update(model, name: str, x: torch.Tensor, edge_index: torch.Tensor, num_nodes: int,
                 u: torch.Tensor, v: torch.Tensor, temp: float = 0.1, hops=(2, 1)) -> dict:
    """``edge_index``: the whole parent graph (src -> dst, both directions), the update's own edges removed."""
    m = copy.deepcopy(model).double().cpu()
    x = x.double().cpu()
    ei = edge_index.cpu()
    u, v = u.cpu(), v.cpu()
    roots, inv = torch.unique(torch.cat([u, v]), return_inverse=True)
    params = [p for p in m.parameters() if p.requires_grad]
    gcn = name == "gcn"
    if gcn:
        ei, w = parent_gcn_weights(ei, num_nodes, dtype=torch.float64)
    else:
        w = None

    def grads(xx, e, ww, pos):
        out = (m(xx, e, ww) if ww is not None else m(xx, e))[0]
        z = F.normalize(out[pos], dim=-1)
        loss = info_nce(z[inv[: len(u)]], z[inv[len(u):]], temp)
        return torch.cat([g.reshape(-1) for g in torch.autograd.grad(loss, params)])

    full = grads(x, ei, w, roots)
    scale = float(full.abs().max())
    out = dict(roots=int(len(roots)), parent_nodes=int(num_nodes), parent_edges=int(ei.shape[1]))
    for h in hops:
        subset, sub_ei, mapping, sub_w = weighted_subgraph(roots, h, ei, w, num_nodes)
        g = grads(x[subset], sub_ei, sub_w, mapping)
        out[f"rel_{h}hop"] = float((g - full).abs().max()) / max(scale, 1e-300)
        out[f"sub_nodes_{h}hop"] = int(len(subset))
    return out
