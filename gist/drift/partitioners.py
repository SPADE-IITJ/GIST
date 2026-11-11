"""Graph partitioners behind the community layer.

Phase 4 (scale) measurement: networkx Louvain over OGBN-Arxiv's 1.17M edges
extrapolates to **240-390 s per refresh**. At 10 refreshes per run across 4
arms and 5 seeds that is ~16 hours of pure partitioning, so it is not a
tuning problem - it makes the experiment infeasible.

METIS, already available through ``torch_sparse``, does the same job in
**2.8-3.9 s**: a ~70-100x speedup with no new dependency.

The two backends optimise different objectives, and the difference is worth
stating plainly rather than hiding behind the word "community":

* **Louvain** maximises modularity. Communities are discovered, their number
  is emergent, and their sizes are highly skewed. This is what the proposal
  describes and what the Cora results in Phases 1-3 used.
* **METIS** minimises edge cut subject to balanced part sizes. The number of
  parts ``k`` must be supplied rather than discovered, and parts come out
  near-uniform in size.

For GIST's purpose - carve the graph into structural regions so
retraining can be confined to one - METIS is adequate and in one respect
better: balanced parts make the retraining set size predictable, where
Louvain's skew means one drifting community might be 10 nodes and the next
10,000. See docs/deviations-from-proposal.md (D18).
"""

from __future__ import annotations

from typing import List, Optional, Tuple

import torch
from torch import Tensor
from torch_geometric.utils import sort_edge_index

# Above this many edges, Louvain is impractical and METIS is selected by
# "auto". Cora (~21k directed edges) stays on Louvain; Arxiv moves to METIS.
AUTO_METIS_EDGE_THRESHOLD = 200_000


def to_csr(edge_index: Tensor, num_nodes: int) -> Tuple[Tensor, Tensor]:
    """CSR (rowptr, col) as METIS expects."""
    ei = sort_edge_index(edge_index, num_nodes=num_nodes)
    row, col = ei[0], ei[1]
    deg = torch.bincount(row, minlength=num_nodes)
    rowptr = torch.cat([torch.zeros(1, dtype=torch.long), deg.cumsum(0)])
    return rowptr.contiguous(), col.contiguous()


def metis_available() -> bool:
    return hasattr(torch.ops, "torch_sparse") and hasattr(
        torch.ops.torch_sparse, "partition"
    )


def louvain_partition(
    edge_index: Tensor, num_nodes: int, seed: int = 0, **_: object
) -> Tensor:
    """Modularity communities via networkx. Returns [N] assignment, -1 unassigned."""
    import networkx as nx
    from networkx.algorithms.community import louvain_communities

    graph = nx.Graph()
    graph.add_nodes_from(range(num_nodes))
    if edge_index.numel():
        graph.add_edges_from(edge_index.t().tolist())

    assignment = torch.full((num_nodes,), -1, dtype=torch.long)
    for cid, members in enumerate(louvain_communities(graph, seed=seed)):
        assignment[torch.as_tensor(sorted(members), dtype=torch.long)] = cid
    return assignment


def metis_partition(
    edge_index: Tensor,
    num_nodes: int,
    seed: int = 0,
    num_parts: int = 64,
    **_: object,
) -> Tensor:
    """Balanced edge-cut partition via METIS. Returns [N] assignment."""
    if edge_index.numel() == 0:
        return torch.zeros(num_nodes, dtype=torch.long)

    rowptr, col = to_csr(edge_index, num_nodes)
    cluster = torch.ops.torch_sparse.partition(
        rowptr, col, None, int(max(2, num_parts)), False
    )
    return cluster.long()


def resolve_num_parts(
    num_nodes: int, num_parts: Optional[int], target_size: int
) -> int:
    """Explicit ``num_parts`` if given, else derived from a target part size."""
    if num_parts:
        return int(num_parts)
    return int(max(2, round(num_nodes / max(target_size, 1))))


def partition_graph(
    edge_index: Tensor,
    num_nodes: int,
    *,
    backend: str = "auto",
    seed: int = 0,
    num_parts: Optional[int] = None,
    target_size: int = 2000,
) -> Tuple[Tensor, str]:
    """Partition the graph, returning ``(assignment, backend_used)``."""
    chosen = backend
    if backend == "auto":
        big = int(edge_index.shape[1]) > AUTO_METIS_EDGE_THRESHOLD
        chosen = "metis" if (big and metis_available()) else "louvain"

    if chosen == "metis":
        if not metis_available():
            raise RuntimeError(
                "METIS requested but torch_sparse.partition is unavailable. "
                "Install torch-sparse built with METIS, or set "
                "drift.partitioner: louvain."
            )
        k = resolve_num_parts(num_nodes, num_parts, target_size)
        return metis_partition(edge_index, num_nodes, seed=seed, num_parts=k), "metis"

    if chosen == "louvain":
        return louvain_partition(edge_index, num_nodes, seed=seed), "louvain"

    raise ValueError(f"Unknown partitioner backend {backend!r}")


def assignment_to_group_tensors(assignment: Tensor, min_size: int) -> List[Tensor]:
    """Node ids per community, ascending within each, communities in id order.

    One stable sort instead of one full-length comparison per community: the
    per-community loop it replaces was O(N * k) and, through ``tolist``, built
    a Python int for every node on every refresh.
    """
    valid = (assignment >= 0).nonzero(as_tuple=True)[0]
    if valid.numel() == 0:
        return []
    nodes = valid[torch.argsort(assignment[valid], stable=True)]
    _, counts = torch.unique_consecutive(assignment[nodes], return_counts=True)
    return [g for g in torch.split(nodes, counts.tolist()) if g.numel() >= min_size]


def assignment_to_groups(assignment: Tensor, min_size: int) -> List[List[int]]:
    """Group node ids by community, dropping communities below ``min_size``."""
    return [g.tolist() for g in assignment_to_group_tensors(assignment, min_size)]
