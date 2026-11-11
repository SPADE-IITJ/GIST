"""Phase 2: a stable, incrementally refreshed global community partition.

Phase 1 recomputed Louvain from scratch on every sampled batch. Three problems
that measurement exposed:

1. **Community ids meant nothing across steps.** Community 7 at step t was
   unrelated to community 7 at step t+1, so per-community drift could not be
   tracked over time and the drift-propagation and origin-attribution analyses
   inherited from DASS-GNN were meaningless.
2. **It partitioned the wrong graph.** Louvain ran on a ~2000-node sampled
   subgraph and returned ~60-70 tiny communities, not the handful of global
   partitions the proposal's ``C = {C_1..C_k}`` describes.
3. **It dominated step time** - 27-40 ms against ~3 ms for inference - so any
   throughput number would have been measuring Louvain, not the method.

This module partitions the *global* graph, refreshes only every
``refresh_interval`` steps, and carries community identity across refreshes by
matching each new community to the old one it overlaps most. Ids are therefore
stable, and per-community drift becomes a time series.
"""

from __future__ import annotations

from concurrent.futures import Future, ThreadPoolExecutor
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence

import numpy as np
import torch
from torch import Tensor

from .partitioners import assignment_to_group_tensors, partition_graph
from .signals import RunningStandardiser


@dataclass
class GlobalPartition:
    assignment: Tensor  # [N] stable community id per node, -1 if unassigned
    sizes: Dict[int, int] = field(default_factory=dict)
    refreshed_at: int = -1
    refresh_count: int = 0
    # Fraction of nodes that kept their community id across this refresh.
    # Reported rather than assumed: METIS parts are balanced, not
    # structurally anchored, so identity carry-over is not free.
    stability: float = 1.0

    @property
    def num_communities(self) -> int:
        return len(self.sizes)


@dataclass
class CommunityDriftResult:
    """Per-community drift keyed by *stable* global community id."""

    scores: Dict[int, float] = field(default_factory=dict)
    counts: Dict[int, int] = field(default_factory=dict)
    max_score: float = 0.0
    max_z: float = 0.0
    flag: bool = False
    num_communities: int = 0
    num_observed: int = 0


class GlobalCommunityTracker:
    """Maintains a stable partition of the whole graph and its drift history."""

    def __init__(
        self,
        num_nodes: int,
        *,
        refresh_interval: int = 200,
        min_size: int = 15,
        threshold: float = 2.11,
        selection_k: float = 1.5,
        ema: float = 0.1,
        warmup: int = 20,
        seed: int = 0,
        partitioner: str = "auto",
        num_parts: Optional[int] = None,
        target_size: int = 2000,
        refresh_lag: int = 0,
    ) -> None:
        self.num_nodes = int(num_nodes)
        self.refresh_interval = int(refresh_interval)
        # 0: refresh synchronously, on demand, inside the event that finds it
        # due (the original behaviour). > 0: refresh in a background thread
        # off the event path - started by ``tick`` every ``refresh_interval``
        # steps on a snapshot of the graph, installed exactly ``refresh_lag``
        # steps later. Install points depend on the step count only, never on
        # thread timing, so runs stay deterministic. G1 measured the
        # synchronous refresh at ~800 ms of METIS on Arxiv, landing on events.
        self.refresh_lag = int(refresh_lag)
        if self.refresh_lag and not 0 < self.refresh_lag < self.refresh_interval:
            raise ValueError("refresh_lag must be 0 or in (0, refresh_interval)")
        self._executor: Optional[ThreadPoolExecutor] = None
        self._pending: Optional[tuple[int, Future]] = None
        self._last_start = -1
        self.min_size = int(min_size)
        self.partitioner = partitioner
        self.num_parts = num_parts
        self.target_size = int(target_size)
        self.backend_used = partitioner
        self.threshold = threshold
        self.selection_k = selection_k
        self.ema = ema
        self.seed = seed

        self.partition = GlobalPartition(
            assignment=torch.full((num_nodes,), -1, dtype=torch.long)
        )
        self._next_id = 0
        self._gate_norm = RunningStandardiser(ema=ema, warmup=warmup)
        # Per-node EMA reference embedding, dense and allocated on first use
        # (the width is the embedding's). Replaces a dict of per-node arrays
        # walked in a Python loop, which cost ~40-110 ms per event on Arxiv.
        self._reference: Optional[np.ndarray] = None
        self._has_reference = np.zeros(self.num_nodes, dtype=bool)
        # Stable ids make this meaningful: drift per community over time.
        self.history: Dict[int, List[tuple[int, float]]] = {}

    # -- partition maintenance -----------------------------------------

    def due(self, step: int) -> bool:
        if self.partition.refreshed_at < 0:
            return True
        return (step - self.partition.refreshed_at) >= self.refresh_interval

    def refresh(self, edge_index: Tensor, step: int) -> GlobalPartition:
        """Recompute the global partition, preserving community identity."""
        self.partition = self._compute(edge_index, step, self.partition)
        return self.partition

    def _compute(
        self, edge_index: Tensor, step: int, prior: GlobalPartition
    ) -> GlobalPartition:
        """The partition that follows ``prior``. Reads no global RNG, so it may
        run on a background thread while the stream samples."""
        raw, backend = partition_graph(
            edge_index,
            self.num_nodes,
            backend=self.partitioner,
            seed=self.seed,
            num_parts=self.num_parts,
            target_size=self.target_size,
        )
        self.backend_used = backend

        found = assignment_to_group_tensors(raw, self.min_size)
        found.sort(key=lambda g: int(g.numel()), reverse=True)

        previous = prior.assignment
        assignment = torch.full((self.num_nodes,), -1, dtype=torch.long)
        sizes: Dict[int, int] = {}

        for group_idx, cid in self._match_ids(previous, found).items():
            assignment[found[group_idx]] = cid
            sizes[cid] = int(found[group_idx].numel())

        tracked = (previous >= 0) & (assignment >= 0)
        stability = (
            float((previous[tracked] == assignment[tracked]).float().mean())
            if bool(tracked.any())
            else 1.0
        )

        return GlobalPartition(
            assignment=assignment,
            sizes=sizes,
            refreshed_at=step,
            refresh_count=prior.refresh_count + 1,
            stability=stability,
        )

    def _match_ids(self, previous: Tensor, groups: List[Tensor]) -> Dict[int, int]:
        """Assign each new community the previous id it most overlaps.

        Greedy maximum-overlap bipartite matching: every (new, old) overlap is
        ranked and claimed in descending order, so a large genuine overlap is
        never lost to an earlier small one.

        This replaced a per-group "first acceptable match above 25% overlap"
        rule, which scored **0% id stability** under METIS. Balanced parts of
        a k-way partition overlap any previous part by roughly 1/k, so a fixed
        fraction threshold can never be met and every community was renumbered
        on every refresh - destroying exactly the cross-step identity Phase 2
        introduced.
        """
        overlaps: List[tuple[int, int, int]] = []
        if groups:
            # Every (new group, old id) overlap count in one pass.
            members = torch.cat(list(groups))
            group_of = torch.repeat_interleave(
                torch.arange(len(groups)),
                torch.tensor([int(g.numel()) for g in groups]),
            )
            prior = previous[members]
            seen = prior >= 0
            if bool(seen.any()):
                width = int(prior.max()) + 1
                pairs, counts = torch.unique(
                    group_of[seen] * width + prior[seen], return_counts=True
                )
                overlaps = [
                    (int(c), int(p) // width, int(p) % width)
                    for p, c in zip(pairs.tolist(), counts.tolist())
                ]

        overlaps.sort(reverse=True)
        matched: Dict[int, int] = {}
        claimed_old: set[int] = set()
        for _, group_idx, cid in overlaps:
            if group_idx in matched or cid in claimed_old:
                continue
            matched[group_idx] = cid
            claimed_old.add(cid)

        for group_idx in range(len(groups)):
            if group_idx not in matched:
                matched[group_idx] = self._next_id
                self._next_id += 1
        return matched

    def ensure(self, edge_index: Tensor, step: int) -> bool:
        if self.refresh_lag:
            # Background mode: ``tick`` owns the schedule. Only a tracker with
            # no partition yet must wait for (or build) one.
            installed = self._install(step)
            if self.partition.refreshed_at < 0:
                if self._pending is None:
                    self._start(edge_index, step)
                self._install(step, force=True)
                installed = True
            return installed
        if self.due(step):
            self.refresh(edge_index, step)
            return True
        return False

    # -- background refresh (refresh_lag > 0) ----------------------------

    def tick(self, edge_index: Tensor, step: int) -> None:
        """Advance the background schedule; call once per stream step."""
        if not self.refresh_lag:
            return
        self._install(step)
        if self._pending is None and (
            self._last_start < 0 or step - self._last_start >= self.refresh_interval
        ):
            self._start(edge_index, step)

    def _start(self, edge_index: Tensor, step: int) -> None:
        # GraphState replaces its edge tensor on every insertion rather than
        # mutating it, so holding this reference is a consistent snapshot.
        if self._executor is None:
            self._executor = ThreadPoolExecutor(max_workers=1)
        prior = self.partition
        self._pending = (step, self._executor.submit(self._compute, edge_index, step, prior))
        self._last_start = step

    def _install(self, step: int, force: bool = False) -> bool:
        if self._pending is None:
            return False
        started, future = self._pending
        if not force and step - started < self.refresh_lag:
            return False
        self.partition = future.result()  # blocks only if not finished yet
        self._pending = None
        return True

    def close(self) -> None:
        """Wait for any refresh in flight and stop the worker thread."""
        if self._pending is not None:
            self._pending[1].result()
            self._pending = None
        if self._executor is not None:
            self._executor.shutdown(wait=True)
            self._executor = None

    # -- drift ----------------------------------------------------------

    def observe(
        self, node_ids: Tensor, embeddings: Tensor, step: int
    ) -> CommunityDriftResult:
        """Aggregate node-level embedding drift by stable community id.

        Vectorised, and bit-for-bit what the per-node loop it replaced
        computed (``tests/test_community_fast.py`` checks this against that
        loop): the node norm is a stacked matmul, which reproduces
        ``np.linalg.norm`` of each row exactly, and ``np.bincount`` sums each
        community's drift in node order in float64, as the loop did.
        """
        emb = embeddings.detach().cpu().numpy()
        ids = node_ids.cpu()
        cids = self.partition.assignment[ids].numpy()
        ids = ids.numpy()
        keep = np.nonzero(cids >= 0)[0]
        ids, cids, current = ids[keep], cids[keep], emb[keep]
        if np.unique(ids).size != ids.size:
            raise ValueError("observe expects each node id at most once")

        if self._reference is None:
            self._reference = np.zeros((self.num_nodes, emb.shape[1]), dtype=emb.dtype)
        had = self._has_reference[ids]
        previous = self._reference[ids]
        diff = current - previous
        norms = np.sqrt(np.matmul(diff[:, None, :], diff[:, :, None])[:, 0, 0])
        drift = np.where(had, norms.astype(np.float64), 0.0)
        self._reference[ids] = np.where(
            had[:, None], self.ema * current + (1 - self.ema) * previous, current
        )
        self._has_reference[ids] = True

        # Communities in order of first appearance, as the loop's dict was.
        present, first = np.unique(cids, return_index=True)
        order = present[np.argsort(first, kind="stable")]
        totals = np.bincount(cids, weights=drift, minlength=int(present.max(initial=-1)) + 1)
        tallies = np.bincount(cids, minlength=totals.size)
        counts: Dict[int, int] = {int(c): int(tallies[c]) for c in order}
        scores = {int(c): float(totals[c]) / counts[int(c)] for c in order}
        for cid, score in scores.items():
            self.history.setdefault(cid, []).append((step, score))

        max_score = max(scores.values()) if scores else 0.0
        max_z = self._gate_norm.update(max_score)

        return CommunityDriftResult(
            scores=scores,
            counts=counts,
            max_score=max_score,
            max_z=max_z,
            flag=bool(self._gate_norm.ready and max_z > self.threshold),
            num_communities=self.partition.num_communities,
            num_observed=len(scores),
        )

    def drifting_communities(self, result: CommunityDriftResult) -> List[int]:
        """The tail of the observed community-drift distribution."""
        if len(result.scores) < 2:
            return list(result.scores)

        values = np.fromiter(result.scores.values(), dtype=float)
        cutoff = float(values.mean() + self.selection_k * values.std())
        selected = [cid for cid, score in result.scores.items() if score > cutoff]
        if not selected:
            selected = [max(result.scores, key=result.scores.get)]
        return selected

    def community_nodes(
        self,
        communities: Sequence[int],
        restrict_mask: Optional[Tensor] = None,
        limit: Optional[int] = None,
        generator: Optional[torch.Generator] = None,
    ) -> Tensor:
        """All nodes of the given communities, graph-wide.

        This is what "extract only the affected subgraph" means: the retraining
        set is drawn from the drifted *region of the graph*, not from whichever
        few nodes of it happen to appear in the current inference batch.
        Restricting to batch roots yields a handful of nodes per event, and an
        exact gradient on a handful of nodes destroys the model.
        """
        if not communities:
            return torch.empty(0, dtype=torch.long)

        wanted = torch.as_tensor(sorted(communities), dtype=torch.long)
        selected = torch.isin(self.partition.assignment, wanted)
        if restrict_mask is not None:
            selected = selected & restrict_mask

        nodes = selected.nonzero(as_tuple=True)[0]
        if limit is not None and nodes.numel() > limit:
            perm = torch.randperm(nodes.numel(), generator=generator)[:limit]
            nodes = nodes[perm]
        return nodes

    def node_mask_for(
        self, communities: Sequence[int], node_ids: Tensor, limit: int
    ) -> Tensor:
        """Boolean mask over the first ``limit`` entries of ``node_ids``."""
        mask = torch.zeros(limit, dtype=torch.bool)
        if not communities:
            return mask
        wanted = torch.as_tensor(sorted(communities), dtype=torch.long)
        cids = self.partition.assignment[node_ids[:limit].cpu()]
        return torch.isin(cids, wanted)
