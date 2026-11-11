"""Fixes that make non-local layers exact on the targets' L-hop subgraph.

G3 and G5 found which layers give the same gradient on the induced subgraph as on the whole batch, and why the
others do not. Every failure came from a quantity the layer recomputes from whatever graph it is handed:

* **GCN's symmetric normalisation** reads the degree of every node it aggregates from. A node on the subgraph's
  boundary is missing neighbours, so its degree is wrong (exact only at depth + 1). T-GCN and EvolveGCN-O use the
  same convolution.
* **ChebConv's Laplacian** (GConvGRU) is built from the subgraph's own edges, and ``sym`` normalisation also takes
  ``lambda_max`` from them, so on a sampled batch it was exact at no hop count (G5).
* **BatchNorm in training mode** normalises with statistics over every node present.

The fix is the same in all three graph cases: compute the edge weights **once, on the parent batch**, and slice
them with the subgraph. The layer is then told not to normalise again. For BatchNorm, the running statistics are
used instead of batch statistics while a subgraph step runs.

    ei, w = parent_gcn_weights(batch.edge_index, batch.num_nodes)      # GCN, T-GCN, EvolveGCN-O
    ei, w = parent_cheb_weights(model, batch.edge_index, batch.num_nodes)   # GConvGRU
    subset, sub_ei, mapping, sub_w = weighted_subgraph(targets, hops, ei, w, batch.num_nodes)

``tests/test_backbone_fixes.py`` checks each fix in float64 against the full-batch gradient. The PyG Temporal models
are imported lazily, so this module works in the main environment, which does not have PyG Temporal.
"""

from __future__ import annotations

from contextlib import contextmanager
from typing import Iterator, Optional, Tuple

import torch
from torch import Tensor, nn
from torch_geometric.nn import ChebConv, GCNConv
from torch_geometric.nn.conv.gcn_conv import gcn_norm
from torch_geometric.utils import k_hop_subgraph


# --- graph weights computed on the parent ---------------------------------------------------------------------

def parent_gcn_weights(
    edge_index: Tensor,
    num_nodes: int,
    *,
    improved: bool = False,
    add_self_loops: bool = True,
    dtype: Optional[torch.dtype] = None,
) -> Tuple[Tensor, Tensor]:
    """GCN's normalisation on the parent graph: ``(edge_index with self-loops, weights)``.

    Feed both to ``GCNConv(normalize=False)``. On the parent this reproduces the default layer exactly; sliced to a
    subgraph it keeps the parent's degrees.
    """
    return gcn_norm(edge_index, None, num_nodes, improved=improved, add_self_loops=add_self_loops, dtype=dtype)


def parent_cheb_weights(
    conv_or_model: nn.Module,
    edge_index: Tensor,
    num_nodes: int,
    *,
    lambda_max: Optional[float] = None,
    dtype: Optional[torch.dtype] = None,
) -> Tuple[Tensor, Tensor]:
    """ChebConv's scaled Laplacian ``2 L / lambda_max - I`` on the parent graph.

    ``conv_or_model`` is a ChebConv or a module containing them (all must share one normalisation, as GConvGRU's
    six do). Pass the result to a model prepared with :func:`exact_gconvgru`.
    """
    convs = [m for m in conv_or_model.modules() if isinstance(m, ChebConv)]
    if not convs:
        raise ValueError("no ChebConv found")
    norms = {c.normalization for c in convs}
    if len(norms) != 1:
        raise ValueError(f"ChebConvs disagree on normalisation: {norms}")
    if norms == {None} or norms == {"rw"}:
        if lambda_max is None:
            raise ValueError("non-symmetric normalisation needs lambda_max (compute it on the parent graph)")
    lm = None if lambda_max is None else torch.tensor(float(lambda_max), dtype=dtype)
    return ChebConv.__norm__(convs[0], edge_index, num_nodes, None, convs[0].normalization, lm, dtype=dtype)


def weighted_subgraph(
    targets: Tensor, num_hops: int, edge_index: Tensor, edge_weight: Optional[Tensor], num_nodes: int
) -> Tuple[Tensor, Tensor, Tensor, Optional[Tensor]]:
    """``k_hop_subgraph`` that carries the parent's edge weights along: ``(subset, edge_index, mapping, weight)``."""
    subset, sub_ei, mapping, edge_mask = k_hop_subgraph(
        targets, num_hops, edge_index, relabel_nodes=True, num_nodes=num_nodes
    )
    return subset, sub_ei, mapping, None if edge_weight is None else edge_weight[edge_mask]


# --- layers told not to normalise again -----------------------------------------------------------------------

class ParentNormGCN(nn.Module):
    """A GCN stack whose convolutions take precomputed weights (``normalize=False``).

    Called without weights, it normalises the graph it is given, so on the full batch it is the ordinary GCN.
    Called with :func:`parent_gcn_weights` sliced to a subgraph, it is exact at depth (G3: 1.9e-16).
    """

    def __init__(self, in_dim: int, hidden: int, out_dim: int, num_layers: int = 2, dropout: float = 0.0):
        super().__init__()
        dims = [in_dim] + [hidden] * (num_layers - 1) + [out_dim]
        self.convs = nn.ModuleList(GCNConv(a, b, normalize=False) for a, b in zip(dims[:-1], dims[1:]))
        self.dropout = dropout
        self.num_hops = num_layers

    def forward(self, x: Tensor, edge_index: Tensor, edge_weight: Optional[Tensor] = None) -> Tensor:
        if edge_weight is None:
            edge_index, edge_weight = parent_gcn_weights(edge_index, x.size(0), dtype=x.dtype)
        for i, conv in enumerate(self.convs):
            x = conv(x, edge_index, edge_weight)
            if i < len(self.convs) - 1:
                x = torch.nn.functional.dropout(torch.relu(x), self.dropout, self.training)
        return x


class _PrecomputedChebConv(ChebConv):
    """ChebConv that takes ``(edge_index, weight)`` as an already-scaled Laplacian and does not rebuild it."""

    def __norm__(self, edge_index, num_nodes, edge_weight, normalization, lambda_max=None, dtype=None, batch=None):
        if edge_weight is None:
            raise ValueError("precomputed ChebConv needs the parent's scaled Laplacian as edge_weight")
        return edge_index, edge_weight


def exact_tgcn(model: nn.Module) -> nn.Module:
    """Make every T-GCN cell in ``model`` take precomputed GCN weights.

    PyG Temporal's ``TGCN`` does not expose ``normalize``, so it is switched off on its ``conv_z``, ``conv_r`` and
    ``conv_h``. Pass :func:`parent_gcn_weights` (with the cell's ``improved`` and ``add_self_loops``). The hidden
    state ``H`` is per node: gather the subgraph's rows with ``subset`` and write them back.
    """
    from torch_geometric_temporal.nn.recurrent import TGCN

    cells = [m for m in model.modules() if isinstance(m, TGCN)]
    if not cells:
        raise ValueError("no TGCN cell found")
    for cell in cells:
        for conv in (cell.conv_z, cell.conv_r, cell.conv_h):
            conv.normalize = False
    return model


def exact_gconvgru(model: nn.Module) -> nn.Module:
    """Make every ChebConv in ``model`` (GConvGRU's six per cell) take a precomputed scaled Laplacian.

    Pass :func:`parent_cheb_weights`. Each cell is 2 hops deep when the stored state is non-zero (its reset gate is a
    1-hop convolution inside ``H * R``), so extract ``2 x cells`` hops for ``K = 2``. The state ``H`` is per node
    and, unlike T-GCN's, convolved: it carries information further with every snapshot.
    """
    convs = [m for m in model.modules() if type(m) is ChebConv]
    if not convs:
        raise ValueError("no ChebConv found")
    for conv in convs:
        conv.__class__ = _PrecomputedChebConv
    return model


def evolvegcno_exact_kwargs() -> dict:
    """``EvolveGCNO(**evolvegcno_exact_kwargs())`` accepts precomputed GCN weights; nothing else changes.

    Its evolved weight matrix is global module state, not per node: carry it across snapshots and reset it per
    stream (``model.weight = None``).
    """
    return dict(normalize=False)


# --- BatchNorm -----------------------------------------------------------------------------------------------

@contextmanager
def frozen_norm_stats(model: nn.Module) -> Iterator[nn.Module]:
    """Run BatchNorm on its running statistics, and leave them untouched, inside the block.

    In training mode BatchNorm normalises with statistics over every node present, so no subgraph of a batch can
    reproduce them (G3: 1.32 relative difference, at any hop count). With running statistics it is a per-node
    affine map and exact (G3: eval mode). Use it around **both** arms' retraining steps, so `full` and `selective`
    differ only in where they update, not in how they normalise.
    """
    norms = [m for m in model.modules() if isinstance(m, nn.modules.batchnorm._BatchNorm)]
    modes = [m.training for m in norms]
    try:
        for m in norms:
            m.eval()
        yield model
    finally:
        for m, mode in zip(norms, modes):
            m.train(mode)
