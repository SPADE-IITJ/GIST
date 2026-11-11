"""Typed configuration, loaded from YAML with dataclass defaults.

Replaces the hardcoded magic numbers of the DASS-GNN prototype (whose
``configs/configs.yaml`` was an empty file).
"""

from __future__ import annotations

import copy
from dataclasses import dataclass, field, fields, is_dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional

import yaml


@dataclass
class DataConfig:
    name: str = "Cora"
    root: str = "data"
    # Fraction of the dataset's edges used to build the base snapshot G_0.
    # The remainder is replayed as the edge stream. The DASS-GNN prototype
    # started from an *empty* graph, so early streaming steps evaluated a
    # model trained on the full graph against a near-empty one.
    base_edge_fraction: float = 0.5
    # Disjoint node-level split. Gradient steps only ever touch train nodes;
    # metrics are only ever read off holdout nodes.
    train_fraction: float = 0.6


@dataclass
class ModelConfig:
    hidden: int = 64
    heads: int = 4
    lr: float = 0.005
    weight_decay: float = 0.0
    warmup_epochs: int = 5
    # The graph warmup trains on. False (every result through Phase C):
    # `plan.base_edges` as stored - one direction per undirected pair, lower id
    # to higher - so each node aggregates only from its lower-id neighbours.
    # The stream itself runs on GraphState's symmetric graph, so the model
    # meets a different graph at step 0 than the one it was trained on, and
    # any retraining arm partly gains by correcting that. Found from G2a's
    # Arxiv entry-accuracy check (+9.6 pp over the probe). True trains on the
    # symmetric base graph the stream starts from.
    warmup_symmetric: bool = False
    # Which backbone the layer wraps (gist/models/registry.py): gat, sage,
    # gcn or gin. Every result before the prelim campaign used gat.
    backbone: str = "gat"


@dataclass
class StreamConfig:
    steps: int = 2000
    batch_size: int = 128
    num_neighbors: List[int] = field(default_factory=lambda: [10, 10])
    # Steps between NeighborLoader rebuilds. Constructing a loader builds a
    # CSC representation of the whole graph, which is O(E) - the prototype
    # paid that on *every* step, exactly the cost the proposal claims to
    # avoid. Rebuilding every k steps leaves the sampler at most k edges
    # stale out of |E|, which is negligible, and is the difference between
    # O(E) and O(E/k) amortised per step on a large graph.
    sampler_refresh_interval: int = 20


@dataclass
class DriftProcessConfig:
    """How drift arrives, where it lands, and what it changes.

    ``mode`` selects the arrival process:
      periodic - the prototype's fixed interval (a timer matches it exactly)
      poisson  - irregular but at a stable average rate
      regime   - bursty: calm stretches alternating with turbulent ones
    """

    mode: str = "regime"

    # regime mode
    calm_mean_gap: float = 220.0
    turbulent_mean_gap: float = 22.0
    calm_duration: List[int] = field(default_factory=lambda: [250, 550])
    turbulent_duration: List[int] = field(default_factory=lambda: [70, 170])

    # poisson mode
    poisson_mean_gap: float = 90.0

    # periodic mode
    interval: int = 200

    # event content
    kinds: List[str] = field(default_factory=lambda: ["structural", "feature", "label"])
    # "localized" targets whole communities; "diffuse" scatters the same
    # number of nodes. Diffuse is the control for GIST's core
    # assumption that drift is spatially localized.
    locality: str = "localized"
    affected_fraction: List[float] = field(default_factory=lambda: [0.03, 0.08])
    magnitude: List[float] = field(default_factory=lambda: [0.3, 0.8])
    structural_edges: int = 40
    # Preferential targeting: 0 = uniform (the reported campaign). alpha > 0 weights each node by (degree + 1)^alpha,
    # so hubs are hit more often and new structural edges attach preferentially to hubs. Localized events draw
    # communities by weighted mass and then the affected nodes from a 2x pool by weight; diffuse events draw the
    # nodes directly by weight. Uses its own rng, so event times, kinds, magnitudes and sizes match alpha = 0.
    degree_bias: float = 0.0
    # Partitioner used to *target* localized drift (not the one the detector
    # sees). Same auto rule as drift.partitioner; a direct networkx Louvain
    # here costs 240-390 s at Arxiv scale.
    partitioner: str = "auto"
    partition_target_size: int = 2000


@dataclass
class DriftConfig:
    """Composite drift metric D_total = a*z(D_emb) + b*z(D_pred) + c*z(D_att).

    Each raw signal is standardised by a running EMA mean/variance before it
    is weighted, because the three signals live on incomparable scales
    (embedding-norm deltas ~O(1), prediction-histogram L2 ~O(0.1), attention
    deltas ~O(0.01)). Without that, any choice of a/b/c is meaningless.
    """

    weight_embedding: float = 1.0 / 3.0
    weight_prediction: float = 1.0 / 3.0
    weight_attention: float = 1.0 / 3.0
    ema: float = 0.1
    # Observations to accumulate before the detector is allowed to fire.
    warmup: int = 20
    # Threshold on the standardised composite score, in sigmas.
    # Set by scripts/calibrate.py at the p95 of the null distribution.
    total_threshold: float = 2.16
    # Gate threshold on the *standardised* community batch-max, in sigmas.
    community_threshold: float = 2.11
    # Selector width: a community is "drifting" when its score exceeds the
    # within-batch mean by this many within-batch standard deviations.
    community_selection_k: float = 1.5
    # Whether community drift also participates in the retraining gate.
    # Default false: the calibrated composite D_total is the trigger, and
    # community drift decides *where* to act, not *when*. Enabling it made the
    # gate saturate (it fired on 98% of steps), which collapsed drift-triggered
    # retraining into fixed-period retraining.
    gate_uses_community: bool = False
    # Run community detection every k steps (1 = every step).
    community_stride: int = 1
    # Which signal decides *when* ("composite": D_total above; "error": training-root error rate, a two-window
    # z test, gist/drift/error_signal.py) and *where* ("embedding": per-community embedding drift; "error":
    # per-community rise in training-root error). Defaults reproduce every earlier result.
    signal: str = "composite"
    region_signal: str = "embedding"
    error_window: int = 30
    error_reference: int = 150
    error_threshold: float = 3.0
    error_halflife: float = 50.0
    error_ref_halflife: float = 300.0
    # Run the composite detector every step. Off skips the `detect` stage
    # entirely, which only arms that never read the drift signal (`baseline`,
    # `periodic`) may do: it measures what detection costs end to end. For an
    # arm that adapts on drift it would silently disable adaptation, so the
    # runner refuses it.
    detection: bool = True

    # "batch"  - Phase 1: Louvain per sampled subgraph. Ids unstable across
    #            steps, ~60-70 tiny communities, dominates step time.
    # "global" - Phase 2: one stable partition of the whole graph, refreshed
    #            on an interval, ids carried across refreshes.
    community_mode: str = "global"
    # Steps between global Louvain refreshes. This is the main scaling knob:
    # networkx Louvain over the full graph is the most expensive operation in
    # the pipeline, so on large graphs this wants to be large (or the
    # partitioner swapped for a faster one).
    community_refresh_interval: int = 200
    # 0: the refresh runs synchronously inside the retraining event that finds
    # it due (the behaviour every result before G1-fix used). k > 0: it runs
    # in a background thread, started every refresh_interval steps and
    # installed exactly k steps later, so it leaves the event path. G1 found
    # the synchronous refresh (~800 ms of METIS on Arxiv) was what made
    # selective slower than full end to end. Must be < refresh_interval.
    community_refresh_lag: int = 0
    community_min_size: int = 15
    # "auto" picks METIS above ~200k edges, Louvain below. networkx Louvain
    # over Arxiv's 1.17M edges takes 240-390 s per refresh; METIS takes ~3 s.
    partitioner: str = "auto"
    # METIS only: number of parts. None derives it from partition_target_size.
    num_parts: int | None = None
    partition_target_size: int = 2000


@dataclass
class AdaptationConfig:
    cooldown: int = 50
    # "subgraph"  - Phase 2: extract the targets' L-hop induced subgraph and
    #               run forward/backward over that alone. Exact: the gradient
    #               equals the full-batch masked gradient (see
    #               tests/test_phase2_subgraph.py).
    # "reweight"  - Phase 1 / DASS-GNN: loss reweighting only. Full-batch
    #               forward and backward; no cost reduction.
    # "mask"      - like reweight but drops non-drifting nodes from the loss.
    selective_mode: str = "subgraph"
    background_weight: float = 0.1
    # Step size for stream-time retraining. `None` falls back to `model.lr`,
    # which is what Phases 1-3 used and keeps the Cora results reproducible.
    #
    # They should not be one number. `model.lr` has to be large enough to
    # train a fresh model from scratch during warmup; a retraining step is
    # applied to an already-converged model, from a fresh Adam state, on a
    # few hundred nodes out of ~100k. Adam's update is ~lr per parameter
    # almost regardless of gradient magnitude, so at Cora's 0.005 each
    # retraining event moves every weight by roughly 10% of its own scale.
    # On Cora that was invisible because the model was nowhere near
    # converged; on Arxiv it destroys a warmed model.
    #
    # Sweeping `model.lr` cannot separate the two effects - it changes the
    # warmed model as well, so even the never-retraining baseline moves.
    retrain_lr: Optional[float] = None
    # Roots the `full`/`periodic` arms draw uniformly per retraining event.
    # 0 keeps Phase 1-3 behaviour: retrain on the step's inference batch,
    # whose size is stream.batch_size. Set it to max_retrain_nodes to match
    # the two arms' budgets, so they differ only in where their nodes sit.
    full_retrain_nodes: int = 0
    # Gradient steps taken per retraining event. One large step is a worse
    # approximation to "adapt to the drifted region" than several small ones.
    retrain_steps: int = 1
    # Cap on target nodes drawn from the drifted communities per event. Kept
    # at the inference batch size so `selective` and `full` update comparable
    # numbers of nodes, and the difference between them is *where* those nodes
    # sit - localized nodes share neighbourhoods, so their induced subgraph is
    # far smaller than that of the same number of scattered nodes.
    max_retrain_nodes: int = 128
    # Share of each retraining set drawn uniformly from the whole trainable
    # pool rather than from the drifted communities. Pure localization
    # overfits the drifted region and forgets the rest; this is explicit
    # rehearsal against that. 0.0 disables it.
    anchor_fraction: float = 0.25
    # Also compute loss on trainable nodes the L-hop expansion already pulled
    # in. Free: their activations are materialised by the forward pass either
    # way, so this buys rehearsal diversity without enlarging the subgraph.
    halo_rehearsal: bool = True
    # Give the `full` and `periodic` arms the same free supervision, so the
    # only difference between arms is locality rather than how many nodes
    # each one supervises. Turning this off makes the comparison unfair in
    # the selective arm's favour.
    supervise_halo: bool = True
    # Fixed schedule for the `periodic` control arm. Leave null to match the
    # `full` arm's realised retraining count automatically, per seed - budget
    # parity is the whole point of the control, and auto-matching removes the
    # last hand-tuned constant from the comparison.
    periodic_period: int | None = None
    # `iso_full` arm: draw as many uniform roots per event as give this many
    # nodes in the backward pass (selective's mean on the same seed), so full
    # and selective cost the same and differ only in where the nodes sit.
    iso_target_nodes: int = 0
    # Exactness audit: at the first k selective events, recompute the update
    # in float64 on CPU, full batch vs subgraph, and record the difference.
    exactness_audit: int = 0
    # Diagnostics only (C2/C3 ceilings, phase 0). Both read the injected drift events from
    # the plan, which no deployment can see. oracle_when: the drift arms fire as soon as the cooldown allows after any
    # event since their last update. oracle_where: selective trains on the nodes those events hit instead of the
    # communities the drift score picks (random_region is unchanged). Off reproduces every earlier result.
    oracle_when: bool = False
    oracle_where: bool = False
    # Region adapters: a per-community
    # correction to the logits, owned by the runtime and updated by every adapting arm on its own targets - the backbone
    # step exactly as before, then `adapter_steps` adapter-only steps (row-wise Adam `adapter_lr`, L2 `adapter_l2`).
    # Off reproduces every earlier result.
    region_adapter: bool = False
    adapter_lr: float = 1e-2
    adapter_l2: float = 1e-3
    adapter_steps: int = 5


@dataclass
class RuntimeConfig:
    """Execution environment and instrumentation."""

    # "auto" | "cpu" | "cuda". No CUDA on the development machine, so every
    # GPU-specific number is unverified until this runs on one.
    device: str = "auto"
    # Measure peak memory around retraining. Precise on CUDA; on CPU it falls
    # back to a weak tracemalloc proxy, reported as such.
    profile_memory: bool = True
    # Synchronise CUDA at every stage boundary. Required to attribute time to
    # the right stage - without it an async launch is charged to whichever
    # later stage happens to block. It also serialises CPU and GPU, which
    # costs real overlap (~30 ms/step at Arxiv scale), so end-to-end
    # throughput must be measured with this off. On: per-stage breakdown is
    # trustworthy. Off: end-to-end wall-clock is trustworthy. Not both.
    sync_stages: bool = True
    # Record the CUDA allocator peak around every retraining event. It
    # synchronises the device, so leave it off in end-to-end timing runs.
    measure_retrain_memory: bool = False


@dataclass
class ExperimentConfig:
    seeds: List[int] = field(default_factory=lambda: [0, 1, 2, 3, 4])
    arms: List[str] = field(default_factory=lambda: ["baseline", "full", "selective"])
    out_dir: str = "results"
    tag: str = "phase1"


@dataclass
class Config:
    data: DataConfig = field(default_factory=DataConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    stream: StreamConfig = field(default_factory=StreamConfig)
    drift_process: DriftProcessConfig = field(default_factory=DriftProcessConfig)
    drift: DriftConfig = field(default_factory=DriftConfig)
    runtime: RuntimeConfig = field(default_factory=RuntimeConfig)
    adaptation: AdaptationConfig = field(default_factory=AdaptationConfig)
    experiment: ExperimentConfig = field(default_factory=ExperimentConfig)

    def to_dict(self) -> Dict[str, Any]:
        return _as_dict(self)


def _as_dict(obj: Any) -> Any:
    if is_dataclass(obj):
        return {f.name: _as_dict(getattr(obj, f.name)) for f in fields(obj)}
    if isinstance(obj, (list, tuple)):
        return [_as_dict(v) for v in obj]
    return obj


def _apply(node: Any, overrides: Dict[str, Any], path: str = "") -> None:
    known = {f.name for f in fields(node)}
    for key, value in overrides.items():
        if key not in known:
            raise KeyError(f"Unknown config key: {path}{key}")
        current = getattr(node, key)
        if is_dataclass(current) and isinstance(value, dict):
            _apply(current, value, f"{path}{key}.")
        else:
            setattr(node, key, value)


def load_config(path: str | Path | None = None, **overrides: Any) -> Config:
    """Load a Config from YAML, then apply flat keyword overrides.

    Overrides use dotted paths, e.g. ``load_config(p, **{"stream.steps": 500})``.
    """
    cfg = Config()
    if path is not None:
        raw = yaml.safe_load(Path(path).read_text()) or {}
        _apply(cfg, raw)

    for dotted, value in overrides.items():
        if value is None:
            continue
        node: Any = cfg
        parts = dotted.split(".")
        for part in parts[:-1]:
            node = getattr(node, part)
        if not any(f.name == parts[-1] for f in fields(node)):
            raise KeyError(f"Unknown config key: {dotted}")
        setattr(node, parts[-1], value)

    _validate(cfg)
    return cfg


def _validate(cfg: Config) -> None:
    total = cfg.drift.weight_embedding + cfg.drift.weight_prediction + cfg.drift.weight_attention
    if total <= 0:
        raise ValueError("Drift signal weights must sum to a positive value.")
    if not 0.0 < cfg.data.base_edge_fraction < 1.0:
        raise ValueError("data.base_edge_fraction must lie in (0, 1).")
    if not 0.0 < cfg.data.train_fraction < 1.0:
        raise ValueError("data.train_fraction must lie in (0, 1).")
    if cfg.adaptation.selective_mode not in {"subgraph", "reweight", "mask"}:
        raise ValueError(f"Unknown selective_mode {cfg.adaptation.selective_mode!r}")
    if cfg.drift.signal not in {"composite", "error"}:
        raise ValueError(f"Unknown drift.signal {cfg.drift.signal!r}")
    if cfg.drift.region_signal not in {"embedding", "error"}:
        raise ValueError(f"Unknown drift.region_signal {cfg.drift.region_signal!r}")
    if cfg.drift.partitioner not in {"auto", "louvain", "metis"}:
        raise ValueError(f"Unknown drift.partitioner {cfg.drift.partitioner!r}")
    if cfg.drift.community_mode not in {"batch", "global"}:
        raise ValueError(f"Unknown drift.community_mode {cfg.drift.community_mode!r}")
    if cfg.drift_process.mode not in {"periodic", "poisson", "regime"}:
        raise ValueError(f"Unknown drift_process.mode {cfg.drift_process.mode!r}")
    if cfg.drift_process.degree_bias < 0:
        raise ValueError("drift_process.degree_bias must be >= 0")
    if cfg.drift_process.locality not in {"localized", "diffuse"}:
        raise ValueError(f"Unknown drift_process.locality {cfg.drift_process.locality!r}")
    unknown = set(cfg.drift_process.kinds) - {"structural", "feature", "label"}
    if unknown:
        raise ValueError(f"Unknown drift kinds: {sorted(unknown)}")


def clone(cfg: Config) -> Config:
    return copy.deepcopy(cfg)
