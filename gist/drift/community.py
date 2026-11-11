"""Community-level drift.

Phase 1 keeps the prototype's per-batch Louvain partition so the arms stay
comparable to the DASS-GNN numbers. Two known limitations are carried over
deliberately and are Phase-2 work:

* Louvain is recomputed from scratch on every sampled subgraph, so community
  ids are not stable across steps. Cross-step community tracking (drift
  propagation, origin attribution) is therefore not meaningful yet.
* Recomputing the partition per step dominates the step cost, which will
  distort any throughput measurement until it is made incremental.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List

import networkx as nx
import numpy as np
import torch
from networkx.algorithms.community import louvain_communities
from torch import Tensor

from .signals import RunningStandardiser


@dataclass
class CommunityResult:
    """Per-batch community partition and its drift scores.

    ``community_map`` and the node ids inside ``communities`` are *batch-local*
    indices into the sampled subgraph, matching the rows of the embedding
    tensor. ``max_score`` is the global community-drift indicator.
    """

    scores: Dict[int, float] = field(default_factory=dict)
    community_map: Dict[int, int] = field(default_factory=dict)
    communities: List[List[int]] = field(default_factory=list)
    max_score: float = 0.0
    max_z: float = 0.0  # standardised batch-max, what the gate reads
    flag: bool = False
    num_communities: int = 0


class CommunityDriftDetector:
    """Per-community drift, with a self-calibrating gate and a scale-free selector.

    A *static* tau_c does not survive contact with an adapting model: a
    threshold calibrated on an inert model saturates once retraining begins,
    because retraining itself shifts embeddings and so inflates community
    drift. Measured directly - a static tau_c fired on 98% of steps, which
    silently degenerated "drift-triggered" retraining into "retrain every
    cooldown steps".

    Two consequences:

    * the **gate** standardises the batch-max score against its own running
      distribution, exactly as the global signals are treated, so the
      threshold stays meaningful as the model adapts; and
    * the **selector** compares communities against each other *within the
      current batch* (score > mean + k*std), which is scale-free by
      construction and always picks the tail rather than an arbitrary share.
    """

    def __init__(
        self,
        threshold: float = 2.0,
        ema: float = 0.1,
        seed: int = 0,
        selection_k: float = 1.5,
        warmup: int = 20,
    ) -> None:
        self.threshold = threshold
        self.ema = ema
        self.seed = seed
        self.selection_k = selection_k
        self._gate_norm = RunningStandardiser(ema=ema, warmup=warmup)
        # Keyed by *global* node id so a node's history survives across batches.
        self._reference: Dict[int, np.ndarray] = {}

    def detect(self, edge_index: Tensor, embeddings: Tensor, node_ids: Tensor) -> CommunityResult:
        if edge_index.numel() == 0:
            return CommunityResult()

        graph = nx.Graph()
        graph.add_nodes_from(range(embeddings.shape[0]))
        graph.add_edges_from(edge_index.t().tolist())

        communities = [sorted(c) for c in louvain_communities(graph, seed=self.seed)]
        community_map: Dict[int, int] = {}
        for cid, members in enumerate(communities):
            for node in members:
                community_map[node] = cid

        emb = embeddings.detach().cpu().numpy()
        globals_ = node_ids.tolist()

        node_drift: Dict[int, float] = {}
        for local_idx, global_id in enumerate(globals_):
            current = emb[local_idx]
            previous = self._reference.get(global_id)
            if previous is None:
                node_drift[local_idx] = 0.0
                self._reference[global_id] = current
            else:
                node_drift[local_idx] = float(np.linalg.norm(current - previous))
                self._reference[global_id] = self.ema * current + (1 - self.ema) * previous

        scores: Dict[int, float] = {}
        for cid, members in enumerate(communities):
            valid = [node_drift[n] for n in members if n in node_drift]
            scores[cid] = float(np.mean(valid)) if valid else 0.0

        max_score = max(scores.values()) if scores else 0.0
        max_z = self._gate_norm.update(max_score)
        return CommunityResult(
            scores=scores,
            community_map=community_map,
            communities=communities,
            max_score=max_score,
            max_z=max_z,
            flag=bool(self._gate_norm.ready and max_z > self.threshold),
            num_communities=len(communities),
        )

    def drifting_communities(self, result: CommunityResult) -> List[int]:
        """The tail of the within-batch community drift distribution.

        Communities are ranked against their peers in the same batch rather
        than against a global constant, so the selected set stays a genuine
        tail as the model adapts and the absolute scores move.
        """
        if len(result.scores) < 2:
            return list(result.scores)

        values = np.fromiter(result.scores.values(), dtype=float)
        cutoff = float(values.mean() + self.selection_k * values.std())
        selected = [cid for cid, score in result.scores.items() if score > cutoff]

        # Never return an empty set once the gate has opened: fall back to the
        # single most-drifted community so a fired trigger always acts.
        if not selected:
            selected = [max(result.scores, key=result.scores.get)]
        return selected
