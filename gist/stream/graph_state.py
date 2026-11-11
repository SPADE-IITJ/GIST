"""Incrementally maintained graph topology for the streaming loop."""

from __future__ import annotations

from typing import Iterable, Tuple

import torch
from torch import Tensor


class GraphState:
    """Holds the evolving edge set, exposed as a PyG ``edge_index``.

    Edges are stored symmetrically and de-duplicated. The DASS-GNN prototype
    appended raw directed pairs without de-duplication, so a repeated arrival
    silently reweighted GAT attention on that edge.

    The tensor is cached and extended by concatenation, so a step that adds
    ``k`` edges costs O(E) in the worst case but avoids re-sorting and
    re-allocating from a Python list on every access.
    """

    def __init__(self, num_nodes: int, base_edges: Tensor | None = None) -> None:
        self.num_nodes = int(num_nodes)
        self._seen: set[Tuple[int, int]] = set()
        self._cache: Tensor = torch.empty((2, 0), dtype=torch.long)
        if base_edges is not None and base_edges.numel() > 0:
            self.add_edges(base_edges)

    # -- mutation -------------------------------------------------------

    def _novel_pairs(self, pairs: Iterable[Tuple[int, int]]) -> list[Tuple[int, int]]:
        fresh: list[Tuple[int, int]] = []
        for u, v in pairs:
            if u == v:
                continue
            for a, b in ((u, v), (v, u)):
                if (a, b) not in self._seen:
                    self._seen.add((a, b))
                    fresh.append((a, b))
        return fresh

    def add_edge(self, src: int, dst: int) -> int:
        """Add one undirected edge. Returns the number of directed pairs added."""
        return self.add_edges(torch.tensor([[src], [dst]], dtype=torch.long))

    def add_edges(self, edges: Tensor) -> int:
        """Add a ``[2, k]`` block of undirected edges."""
        if edges.numel() == 0:
            return 0
        pairs = zip(edges[0].tolist(), edges[1].tolist())
        fresh = self._novel_pairs(pairs)
        if not fresh:
            return 0
        block = torch.tensor(fresh, dtype=torch.long).t().contiguous()
        self._cache = torch.cat([self._cache, block], dim=1)
        return len(fresh)

    # -- access ---------------------------------------------------------

    @property
    def edge_index(self) -> Tensor:
        return self._cache

    @property
    def num_directed_edges(self) -> int:
        return int(self._cache.shape[1])

    def __len__(self) -> int:
        return self.num_directed_edges
