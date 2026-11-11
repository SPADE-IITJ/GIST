"""Two-layer GAT encoder that exposes its attention coefficients.

Attention weights are returned because the attention-drift signal
(D_att in the proposal) is computed from them.
"""

from __future__ import annotations

from typing import Tuple

import torch
import torch.nn as nn
from torch import Tensor
from torch_geometric.nn import GATConv

AttentionPair = Tuple[Tensor, Tensor]


class StreamGAT(nn.Module):
    def __init__(self, in_channels: int, hidden: int, out_channels: int, heads: int = 4) -> None:
        super().__init__()
        self.gat1 = GATConv(in_channels, hidden, heads=heads)
        self.gat2 = GATConv(hidden * heads, out_channels)

    def forward(self, x: Tensor, edge_index: Tensor) -> Tuple[Tensor, AttentionPair, AttentionPair]:
        x, attn1 = self.gat1(x, edge_index, return_attention_weights=True)
        x = torch.relu(x)
        x, attn2 = self.gat2(x, edge_index, return_attention_weights=True)
        return x, attn1, attn2


def build_model(in_dim: int, out_dim: int, hidden: int, heads: int) -> StreamGAT:
    return StreamGAT(in_dim, hidden, out_dim, heads=heads)
