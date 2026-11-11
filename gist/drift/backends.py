"""One interface over the two community-detection strategies.

``batch``  - Phase 1: Louvain recomputed per sampled subgraph. Community ids
             are meaningless across steps and it dominates step time. Kept so
             the Phase 1 behaviour stays reproducible for comparison.
``global`` - Phase 2: one stable partition of the whole graph, refreshed on an
             interval, with community identity carried across refreshes.

Both expose ``observe_step`` / ``drifting`` / ``root_mask``, so the runner
never branches on which one is in use.
"""

from __future__ import annotations

from typing import List, Protocol

import torch
from torch import Tensor

from .community import CommunityDriftDetector
from .global_communities import GlobalCommunityTracker


class CommunityBackend(Protocol):
    name: str

    def observe_step(self, *, batch, embeddings: Tensor, global_edge_index: Tensor, step: int): ...
    def drifting(self, observation) -> List[int]: ...
    def root_mask(self, observation, drifting: List[int], batch) -> Tensor: ...
    def tick(self, global_edge_index: Tensor, step: int) -> None: ...
    def close(self) -> None: ...


class BatchLouvainBackend:
    """Phase 1 behaviour: partition the sampled subgraph, every step."""

    name = "batch"

    def __init__(self, detector: CommunityDriftDetector) -> None:
        self.detector = detector

    def observe_step(self, *, batch, embeddings, global_edge_index, step):
        return self.detector.detect(batch.edge_index, embeddings, batch.n_id)

    def drifting(self, observation) -> List[int]:
        return self.detector.drifting_communities(observation)

    def root_mask(self, observation, drifting, batch) -> Tensor:
        """Ids here are *batch-local* indices into the sampled subgraph."""
        mask = torch.zeros(batch.batch_size, dtype=torch.bool)
        if not drifting:
            return mask
        wanted = set(drifting)
        for local_idx, cid in observation.community_map.items():
            if cid in wanted and local_idx < batch.batch_size:
                mask[local_idx] = True
        return mask

    def community_nodes(self, drifting, restrict_mask=None, limit=None, generator=None):
        """Unavailable: a per-batch partition has no graph-wide membership.

        Callers fall back to the batch roots, which is the Phase 1 behaviour.
        """
        return None

    def tick(self, global_edge_index, step) -> None:
        """Nothing runs in the background for a per-batch partition."""

    def close(self) -> None:
        pass


class GlobalLouvainBackend:
    """Phase 2 behaviour: one stable global partition, refreshed on interval."""

    name = "global"

    def __init__(self, tracker: GlobalCommunityTracker) -> None:
        self.tracker = tracker

    def observe_step(self, *, batch, embeddings, global_edge_index, step):
        # Cheap interval check; Louvain only actually runs when due.
        self.tracker.ensure(global_edge_index, step)
        return self.tracker.observe(batch.n_id, embeddings, step)

    def drifting(self, observation) -> List[int]:
        return self.tracker.drifting_communities(observation)

    def root_mask(self, observation, drifting, batch) -> Tensor:
        """Ids here are *stable global* community ids."""
        return self.tracker.node_mask_for(drifting, batch.n_id, int(batch.batch_size))

    def community_nodes(self, drifting, restrict_mask=None, limit=None, generator=None):
        return self.tracker.community_nodes(drifting, restrict_mask, limit, generator)

    def tick(self, global_edge_index, step) -> None:
        """Per-step hook for the background refresh; a no-op when it is off."""
        self.tracker.tick(global_edge_index, step)

    def close(self) -> None:
        self.tracker.close()


def build_community_backend(cfg, num_nodes: int, seed: int) -> CommunityBackend:
    mode = cfg.drift.community_mode
    if mode == "batch":
        return BatchLouvainBackend(
            CommunityDriftDetector(
                threshold=cfg.drift.community_threshold,
                ema=cfg.drift.ema,
                seed=seed,
                selection_k=cfg.drift.community_selection_k,
                warmup=cfg.drift.warmup,
            )
        )
    if mode == "global":
        return GlobalLouvainBackend(
            GlobalCommunityTracker(
                num_nodes,
                refresh_interval=cfg.drift.community_refresh_interval,
                min_size=cfg.drift.community_min_size,
                threshold=cfg.drift.community_threshold,
                selection_k=cfg.drift.community_selection_k,
                ema=cfg.drift.ema,
                warmup=cfg.drift.warmup,
                seed=seed,
                partitioner=cfg.drift.partitioner,
                num_parts=cfg.drift.num_parts,
                target_size=cfg.drift.partition_target_size,
                refresh_lag=cfg.drift.community_refresh_lag,
            )
        )
    raise ValueError(f"Unknown drift.community_mode {mode!r}")
