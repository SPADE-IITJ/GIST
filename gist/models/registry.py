"""Backbone registry: the model contract the layer asks for, and the models that meet it.

The layer needs from a model only:

* ``forward(x, edge_index[, edge_weight]) -> (logits, attention_1, attention_2)``. Attention is ``None`` for
  models without it, and the drift detector then runs on the other two signals.
* ``num_hops``: the receptive-field depth, which sizes the exact subgraph.
* optionally ``parent_edges(edge_index, num_nodes) -> (edge_index, edge_weight)``: graph weights computed once on
  the parent batch. ``subgraph_gradient_step`` slices them with the subgraph instead of letting the layer
  renormalise on it (GCN; see gist/models/exact.py and G3).

The triple keeps every existing call site (``out, attn1, _ = model(...)``) unchanged, so GAT's path is identical to
the one every earlier result used.
"""

from __future__ import annotations

from typing import Optional

import torch
import torch.nn as nn
from torch import Tensor
from torch_geometric.nn import GINConv, SAGEConv

from .exact import ParentNormGCN, parent_gcn_weights
from .gat import StreamGAT

BACKBONES = ("gat", "sage", "gcn", "gin")


class StreamSAGE(nn.Module):
    """Two-layer GraphSAGE with mean aggregation."""

    num_hops = 2

    def __init__(self, in_dim: int, hidden: int, out_dim: int) -> None:
        super().__init__()
        self.conv1 = SAGEConv(in_dim, hidden, aggr="mean")
        self.conv2 = SAGEConv(hidden, out_dim, aggr="mean")

    def forward(self, x: Tensor, edge_index: Tensor, edge_weight: Optional[Tensor] = None):
        return self.conv2(torch.relu(self.conv1(x, edge_index)), edge_index), None, None


class StreamGIN(nn.Module):
    """Two-layer GIN; each layer's MLP is Linear-ReLU-Linear."""

    num_hops = 2

    def __init__(self, in_dim: int, hidden: int, out_dim: int) -> None:
        super().__init__()
        mlp = lambda a, b: nn.Sequential(nn.Linear(a, hidden), nn.ReLU(), nn.Linear(hidden, b))
        self.conv1 = GINConv(mlp(in_dim, hidden))
        self.conv2 = GINConv(mlp(hidden, out_dim))

    def forward(self, x: Tensor, edge_index: Tensor, edge_weight: Optional[Tensor] = None):
        return self.conv2(torch.relu(self.conv1(x, edge_index)), edge_index), None, None


class StreamGCN(ParentNormGCN):
    """Two-layer GCN that is exact on a subgraph: its normalisation comes from the parent batch.

    Called on a whole batch without weights, it normalises that batch - the ordinary GCN. The subgraph step asks
    ``parent_edges`` for the batch's weights first and slices them, so the boundary keeps the parent's degrees.
    """

    def __init__(self, in_dim: int, hidden: int, out_dim: int) -> None:
        super().__init__(in_dim, hidden, out_dim, num_layers=2)

    def forward(self, x: Tensor, edge_index: Tensor, edge_weight: Optional[Tensor] = None):
        return super().forward(x, edge_index, edge_weight), None, None

    @staticmethod
    def parent_edges(edge_index: Tensor, num_nodes: int, dtype: Optional[torch.dtype] = None):
        return parent_gcn_weights(edge_index, num_nodes, dtype=dtype)


def build_backbone(name: str, in_dim: int, out_dim: int, hidden: int, heads: int) -> nn.Module:
    name = name.lower()
    if name == "gat":
        model = StreamGAT(in_dim, hidden, out_dim, heads=heads)
        model.num_hops = 2
        return model
    if name == "sage":
        return StreamSAGE(in_dim, hidden, out_dim)
    if name == "gcn":
        return StreamGCN(in_dim, hidden, out_dim)
    if name == "gin":
        return StreamGIN(in_dim, hidden, out_dim)
    raise ValueError(f"unknown backbone {name!r}; known: {', '.join(BACKBONES)}")


def has_attention(name: str) -> bool:
    """Whether the backbone returns attention coefficients (the detector's third signal)."""
    return name.lower() == "gat"


# --- analytic cost, one function per layer type ----------------------------------------------------------------

def backbone_flops(name: str, nodes: int, edges: int, in_dim: int, hidden: int, out_dim: int, heads: int,
                   backward: bool = True) -> int:
    """Forward (+2x for backward) FLOPs of the two-layer backbone on a materialised subgraph.

    Multiply-accumulate = 2 FLOPs. Counts the dense transforms and the per-edge aggregation; ignores activations.
    """
    from ..evaluation.profiling import model_flops

    name = name.lower()
    if name == "gat":
        return model_flops(nodes, edges, in_dim, hidden, out_dim, heads, backward=backward)
    if name == "sage":   # aggregate the input, then two linears (neighbour and root)
        layer = lambda a, b: 2 * edges * a + 2 * 2 * nodes * a * b
        total = layer(in_dim, hidden) + layer(hidden, out_dim)
    elif name == "gcn":  # linear, then aggregate the output over edges plus self-loops
        layer = lambda a, b: 2 * nodes * a * b + 2 * (edges + nodes) * b
        total = layer(in_dim, hidden) + layer(hidden, out_dim)
    elif name == "gin":  # aggregate the input, then a two-linear MLP
        layer = lambda a, b: 2 * edges * a + 2 * nodes * (a * hidden + hidden * b)
        total = layer(in_dim, hidden) + layer(hidden, out_dim)
    else:
        raise ValueError(name)
    return total * 3 if backward else total


def backbone_activation_bytes(name: str, nodes: int, hidden: int, out_dim: int, heads: int, dtype_bytes: int = 4) -> int:
    """Activations the backward pass must keep, from the materialised node count."""
    from ..evaluation.profiling import activation_bytes

    name = name.lower()
    if name == "gat":
        return activation_bytes(nodes, hidden, out_dim, heads, dtype_bytes)
    per_node = {"sage": hidden + out_dim, "gcn": hidden + out_dim, "gin": 3 * hidden + out_dim}[name]
    return nodes * per_node * dtype_bytes
