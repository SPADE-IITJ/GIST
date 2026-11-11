"""Phase 2: true induced-subgraph retraining.

Phase 1 measured the gap this closes. "Selective" retraining there was loss
reweighting: the forward and backward passes still spanned the entire sampled
batch, so targeting 3.8% of the nodes bought a 4% reduction in backward-pass
work. The proposal's scalability claim - bounded backward-pass memory, fewer
full-graph kernel invocations - rested on something that was never implemented.

This module extracts the induced subgraph that the target nodes actually
depend on and runs forward and backward over *that* alone.

**The update is exact, not an approximation.** An L-layer GNN's output at a
node depends only on its L-hop neighbourhood, so for a 2-layer GAT the 2-hop
induced subgraph carries every activation that contributes to the loss at the
target nodes. The resulting gradient equals the gradient of the same masked
loss computed over the full batch, to floating-point tolerance - which
`tests/test_phase2_subgraph.py` asserts directly. The saving is in work not
done, not in fidelity given up.
"""

from __future__ import annotations

import copy
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from torch import Tensor
from torch_geometric.utils import k_hop_subgraph


@dataclass
class SubgraphStats:
    """What the backward pass actually touched."""

    nodes_targeted: int  # nodes contributing to the loss
    nodes_in_backward: int  # nodes in the forward/backward pass
    edges_in_backward: int
    nodes_in_batch: int  # what full retraining would have used
    edges_in_batch: int

    @property
    def node_ratio(self) -> float:
        return self.nodes_in_backward / max(self.nodes_in_batch, 1)

    @property
    def edge_ratio(self) -> float:
        return self.edges_in_backward / max(self.edges_in_batch, 1)


def extract_subgraph(
    batch,
    targets: Tensor,
    num_hops: int,
    return_edge_mask: bool = False,
):
    """The induced subgraph the target nodes' activations depend on.

    Returns ``(subset, edge_index, mapping)`` where ``subset`` indexes the
    batch, ``edge_index`` is relabelled to the subgraph, and ``mapping`` gives
    the targets' positions inside it; with ``return_edge_mask``, also the mask
    of the batch's edges that were kept.
    """
    seeds = targets.nonzero(as_tuple=True)[0] if targets.dtype == torch.bool else targets
    # Callers build target masks on CPU, but a batch that came through the
    # main loop already sits on the device. k_hop_subgraph requires both on
    # the same one.
    seeds = seeds.to(batch.edge_index.device)
    subset, edge_index, mapping, edge_mask = k_hop_subgraph(
        seeds,
        num_hops,
        batch.edge_index,
        relabel_nodes=True,
        num_nodes=batch.num_nodes,
    )
    if return_edge_mask:
        return subset, edge_index, mapping, edge_mask
    return subset, edge_index, mapping


def subgraph_gradient_step(
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    batch,
    targets: Tensor,
    num_hops: int = 2,
    halo_train_mask: Optional[Tensor] = None,
    adapter=None,
) -> Optional[Tuple[float, SubgraphStats]]:
    """One optimizer step computed only on the targets' induced subgraph.

    ``halo_train_mask`` is a graph-wide boolean mask of trainable nodes. When
    given, the loss additionally covers every trainable node that the L-hop
    expansion already pulled in.

    That extra supervision is **free**: those nodes' activations are
    materialised by the forward pass whether or not they appear in the loss,
    and the backward pass spans the same subgraph either way. It buys
    rehearsal diversity - the main defence against forgetting the rest of the
    graph - without enlarging the subgraph, which is what makes scattered
    anchor nodes expensive.
    """
    if not bool(targets.any()):
        return None

    batch = with_parent_edges(model, batch)
    subset, sub_edge_index, mapping, edge_mask = extract_subgraph(
        batch, targets, num_hops, return_edge_mask=True
    )
    if subset.numel() == 0 or mapping.numel() == 0:
        return None
    # A batch may carry edge weights computed on the whole batch - the
    # parent-graph normalisation that makes GCN-family layers exact
    # (gist/models/exact.py). They are sliced, never recomputed, so the
    # subgraph keeps the parent's degrees.
    edge_weight = getattr(batch, "edge_weight", None)
    sub_edge_weight = None if edge_weight is None else edge_weight[edge_mask]

    sub_x = batch.x[subset]
    sub_y = batch.y[subset]

    loss_nodes = mapping
    if halo_train_mask is not None and hasattr(batch, "n_id"):
        trainable = halo_train_mask[batch.n_id[subset].cpu()].to(subset.device)
        trainable[mapping] = True  # seeds are targets regardless
        loss_nodes = trainable.nonzero(as_tuple=True)[0]

    # Region batches are sampled inside the retrainer and never pass through
    # the main loop's `batch.to(device)`, so they arrive on CPU while the
    # model is on the GPU. Transfer *after* extraction: only the induced
    # subgraph crosses the bus, not the whole sampled region.
    device = next(model.parameters()).device
    sub_x = sub_x.to(device)
    sub_y = sub_y.to(device)
    sub_edge_index = sub_edge_index.to(device)
    loss_nodes = loss_nodes.to(device)

    model.train()
    optimizer.zero_grad()
    if sub_edge_weight is None:
        out = model(sub_x, sub_edge_index)
    else:
        out = model(sub_x, sub_edge_index, sub_edge_weight.to(device))
    # StreamGAT returns (logits, attention, attention); other backbones logits.
    if isinstance(out, tuple):
        out = out[0]

    if adapter is None:
        loss = F.cross_entropy(out[loss_nodes], sub_y[loss_nodes])
    else:
        # Region adapter (adaptation.region_adapter): a per-node map after message passing, so exactness holds.
        node_ids = batch.n_id[subset.to(batch.n_id.device)].to(device)
        loss = F.cross_entropy(adapter.apply(out[loss_nodes], node_ids[loss_nodes]), sub_y[loss_nodes])
    loss.backward()
    optimizer.step()
    if adapter is not None:
        adapter.update(out[loss_nodes].detach(), sub_y[loss_nodes], node_ids[loss_nodes])

    stats = SubgraphStats(
        nodes_targeted=int(loss_nodes.numel()),
        nodes_in_backward=int(subset.numel()),
        edges_in_backward=int(sub_edge_index.shape[1]),
        nodes_in_batch=int(batch.num_nodes),
        edges_in_batch=int(batch.edge_index.shape[1]),
    )
    return float(loss.item()), stats


def with_parent_edges(model, batch):
    """Attach the parent batch's graph weights when the model asks for them (``parent_edges``, e.g. GCN).

    Computed once on the whole batch and later sliced with the subgraph, so a node on the subgraph boundary keeps
    its parent-batch degree; that is what makes GCN exact at depth (G3). Models without the hook, and batches that
    already carry weights, are returned unchanged.
    """
    prepare = getattr(model, "parent_edges", None)
    if prepare is None or getattr(batch, "edge_weight", None) is not None:
        return batch
    weighted = copy.copy(batch)
    weighted.edge_index, weighted.edge_weight = prepare(batch.edge_index, int(batch.num_nodes), dtype=batch.x.dtype)
    return weighted


def audit_exactness(model, x: Tensor, y: Tensor, edge_index: Tensor, num_nodes: int, seeds: Tensor,
                    num_hops: int, adapter=None) -> Dict[str, float]:
    """An update recomputed in float64 on CPU: on the whole graph vs on the seeds' induced ``num_hops``-hop subgraph.

    The parent is the whole current graph (``x``, ``y``, ``edge_index``), not a sampled batch, so a pass means the
    loss at the seeds depends on nothing outside their L-hop neighbourhood, for this model on this graph. Returns
    the gradient difference at ``num_hops`` (< 1e-8 = exact) and at ``num_hops - 1`` (the negative control, which
    must differ), two ways:

    - ``rel_diff_global``: largest absolute difference over all parameters / largest absolute gradient entry over
      all parameters. The criterion.
    - ``rel_diff``: the largest per-tensor ratio. Kept for comparison, but ill-conditioned when a tensor's true
      gradient is ~0: GAT's first-layer ``att_dst`` enters every logit of a destination equally, so the softmax
      cancels it except through the LeakyReLU, and rounding noise of 1e-17 on a 1e-16 gradient reads as 0.1.

    Loss: cross-entropy over the seeds, as in the subgraph step. The live model is not touched. With a region
    ``adapter`` (adaptation.region_adapter) the logits pass through it, as in the update, and its parameters are
    audited too (``adapter.W``, ``adapter.b``).
    """
    m = copy.deepcopy(model).to(device="cpu", dtype=torch.float64)
    m.train()
    x = x.detach().cpu().double()
    y = y.detach().cpu()
    ei = edge_index.detach().cpu()
    n = int(num_nodes)
    w = None
    prepare = getattr(m, "parent_edges", None)
    if prepare is not None:
        ei, w = prepare(ei, n, dtype=torch.float64)
    seeds = (seeds.nonzero(as_tuple=True)[0] if seeds.dtype == torch.bool else seeds).cpu()
    labels = y[seeds]

    ad = None
    if adapter is not None:
        adapter.community_ids(torch.arange(n))         # size the storage for every id first, then copy it
        ad = [adapter.W.detach().cpu().double().clone().requires_grad_(True),
              adapter.b.detach().cpu().double().clone().requires_grad_(True)]

    def grads(xx, e, ww, pos, ids):
        m.zero_grad()
        out = m(xx, e) if ww is None else m(xx, e, ww)
        out = out[0] if isinstance(out, tuple) else out
        out = out[pos]
        if ad is not None:
            ad[0].grad = ad[1].grad = None
            out = adapter.apply(out, ids[pos], ad[0], ad[1])
        F.cross_entropy(out, labels).backward()
        g = {k: p.grad.detach().clone() for k, p in m.named_parameters() if p.grad is not None}
        if ad is not None:
            g.update({"adapter.W": ad[0].grad.detach().clone(), "adapter.b": ad[1].grad.detach().clone()})
        return g

    ref = grads(x, ei, w, seeds, torch.arange(n))
    scale = max(float(g.abs().max()) for g in ref.values())
    result = dict(targets=int(seeds.numel()), parent_nodes=n, parent_edges=int(ei.shape[1]), grad_scale=scale)
    for k, key in ((num_hops, "rel_diff"), (num_hops - 1, "rel_diff_short")):
        subset, sub_ei, mapping, mask = k_hop_subgraph(seeds, k, ei, relabel_nodes=True, num_nodes=n)
        sub = grads(x[subset], sub_ei, None if w is None else w[mask], mapping, subset)
        diff = {t: float((ref[t] - sub[t]).abs().max()) for t in ref}
        per = {t: diff[t] / max(float(ref[t].abs().max()), 1e-300) for t in ref}
        worst = max(per, key=per.get)
        result[key] = per[worst]
        result[key + "_global"] = max(diff.values()) / max(scale, 1e-300)
        result[key + "_worst"] = dict(tensor=worst, abs_diff=diff[worst], grad_max=float(ref[worst].abs().max()))
        result[key.replace("rel_diff", "subgraph_nodes")] = int(subset.numel())
    return result


def full_batch_stats(batch) -> SubgraphStats:
    """Cost accounting for a retraining step that spans the whole batch.

    Note this counts *all* nodes in the sampled subgraph, not just the roots.
    The forward pass materialises activations for every sampled node, so
    charging only the roots - as Phase 1 did - understated the cost of full
    retraining and flattered the comparison.
    """
    roots = int(batch.batch_size)
    return SubgraphStats(
        nodes_targeted=roots,
        nodes_in_backward=int(batch.num_nodes),
        edges_in_backward=int(batch.edge_index.shape[1]),
        nodes_in_batch=int(batch.num_nodes),
        edges_in_batch=int(batch.edge_index.shape[1]),
    )
