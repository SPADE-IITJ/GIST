"""Single-arm, single-seed streaming run."""

from __future__ import annotations

import copy
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

import numpy as np
import torch
from torch import Tensor
from torch_geometric.loader import NeighborLoader
from torch_geometric.sampler import NeighborSampler

from ..adaptation.controller import RetrainingController
from ..adaptation.region_adapter import RegionAdapter
from ..adaptation.retrainers import RetrainContext, RetrainInfo, build_retrainer
from ..config import Config
from ..data.loaders import Dataset
from ..drift.backends import build_community_backend
from ..drift.detector import CompositeDriftDetector, DriftReading
from ..drift.error_signal import ErrorRegionBackend, ErrorSignal
from ..evaluation.metrics import StageTimer, classification_metrics, recovery_steps
from ..evaluation.profiling import (
    StreamProfiler,
    resolve_device,
)
from ..models.registry import backbone_activation_bytes, backbone_flops, build_backbone, has_attention
from ..stream.drift_process import apply_drift
from ..stream.graph_state import GraphState
from ..stream.schedule import StreamPlan


# Recorded in place of a DriftReading when detection is off: no signal was
# computed, so every value is NaN rather than a zero that reads as "no drift".
_NO_READING = DriftReading(*([float("nan")] * 7), flag=False)

# Arms whose firing never depends on the drift signal.
_SIGNAL_FREE_ARMS = ("baseline", "periodic")


def _train_pool(train_mask: Tensor, plan: StreamPlan, step: int) -> Tensor:
    """Trainable nodes at ``step``. A temporal plan admits only nodes that have
    arrived, so no update can see a later paper's features or label; step -1 is
    the base window. Injected-drift plans have no arrival times and are unchanged."""
    arrival = getattr(plan, "arrival_step", None)
    return train_mask if arrival is None else train_mask & (arrival <= step)


@dataclass
class StepRecord:
    step: int
    drift: Dict[str, Any]
    community_max: float
    community_max_z: float
    community_flag: bool
    num_communities: int
    retrained: bool
    accuracy: Optional[float]
    f1: Optional[float]
    precision: Optional[float]
    recall: Optional[float]
    eval_support: int
    num_edges: int


@dataclass
class ArmResult:
    arm: str
    seed: int
    steps: List[StepRecord] = field(default_factory=list)
    retrain_events: List[RetrainInfo] = field(default_factory=list)
    timing: Dict[str, Dict[str, float]] = field(default_factory=dict)
    profile: Dict[str, Any] = field(default_factory=dict)
    summary: Dict[str, float] = field(default_factory=dict)


def sample_batch(data, roots: Tensor, num_neighbors: List[int], seed: int):
    """Deterministically sample one 2-hop batch around fixed root nodes.

    Determinism here is what lets every arm see byte-identical batches: the
    roots come from the shared plan and the neighbour sampling is seeded per
    step rather than drawn from an advancing global stream.
    """
    torch.manual_seed(seed)
    loader = NeighborLoader(
        data,
        num_neighbors=num_neighbors,
        input_nodes=roots,
        batch_size=int(roots.numel()),
        shuffle=False,
    )
    return next(iter(loader))


class CachedSampler:
    """Reuses a NeighborLoader's graph index across steps.

    Constructing a NeighborLoader builds a CSC representation of the whole
    graph - O(E). The prototype paid that on every single step, which is
    precisely the per-event O(|V| + |E|) cost the proposal claims to avoid,
    and on a large graph it dwarfs the GNN itself.

    Rebuilding every ``refresh_interval`` steps leaves the sampler at most
    that many edge insertions stale out of |E| - negligible for sampling -
    and turns an O(E) per-step cost into O(E/k) amortised.
    """

    def __init__(self, data, num_neighbors: List[int], refresh_interval: int) -> None:
        self.data = data
        self.num_neighbors = num_neighbors
        self.refresh_interval = max(1, int(refresh_interval))
        self._sampler: NeighborSampler | None = None
        self._built_at: int | None = None
        self.rebuilds = 0

    def _due(self, step: int) -> bool:
        return self._built_at is None or (step - self._built_at) >= self.refresh_interval

    def sample(self, roots: Tensor, seed: int, step: int):
        if self._due(step):
            # The O(E) work lives here: NeighborSampler builds the CSC index.
            self._sampler = NeighborSampler(self.data, num_neighbors=self.num_neighbors)
            self._built_at = step
            self.rebuilds += 1

        # Wrapping a cached sampler keeps loader construction cheap; a
        # DataLoader's batch_size cannot be mutated after construction.
        loader = NeighborLoader(
            self.data,
            num_neighbors=self.num_neighbors,
            input_nodes=roots,
            batch_size=int(roots.numel()),
            shuffle=False,
            neighbor_sampler=self._sampler,
        )
        torch.manual_seed(seed)
        return next(iter(loader))


@torch.no_grad()
def infer(model, batch):
    model.eval()
    return model(batch.x, batch.edge_index)


def warmup_edges(cfg: Config, num_nodes: int, plan: StreamPlan) -> Tensor:
    """The base-snapshot edges warmup trains on (see ``model.warmup_symmetric``)."""
    if cfg.model.warmup_symmetric:
        return GraphState(num_nodes, plan.base_edges).edge_index
    return plan.base_edges


def warmup_model(cfg: Config, dataset: Dataset, plan: StreamPlan, seed: int, on_epoch=None):
    """Train one model on the base snapshot G_0, using train nodes only.

    The resulting weights are deep-copied into every arm so the arms start
    from *identical* parameters. In the prototype each arm built and warmed up
    its own model off a shared advancing RNG, adding an initialisation
    confound on top of the sampling one.
    """
    torch.manual_seed(seed)
    device = resolve_device(cfg.runtime.device)
    model = build_backbone(cfg.model.backbone, dataset.in_dim, dataset.out_dim, cfg.model.hidden, cfg.model.heads)
    # Warm up on the device the run will use. This used to train on CPU and
    # only move to the GPU afterwards, which at Cora's 5 epochs was
    # invisible but on Arxiv makes each warmup epoch a full-graph forward
    # and backward on CPU - minutes per epoch, and the reason the epoch
    # count could not be raised to something adequate for 40 classes.
    model.to(device)
    optimizer = torch.optim.Adam(
        model.parameters(), lr=cfg.model.lr, weight_decay=cfg.model.weight_decay
    )

    data = copy.copy(dataset.data)
    data.edge_index = warmup_edges(cfg, dataset.num_nodes, plan)
    data.n_id = torch.arange(data.num_nodes)

    train_nodes = _train_pool(dataset.train_mask, plan, -1).nonzero(as_tuple=True)[0]
    losses = []
    for epoch in range(cfg.model.warmup_epochs):
        batch = sample_batch(
            data,
            train_nodes,
            cfg.stream.num_neighbors,
            seed=plan.sample_seeds[0] + epoch,
        )
        if device.type != "cpu":
            batch = batch.to(device)
        model.train()
        optimizer.zero_grad()
        out, _, _ = model(batch.x, batch.edge_index)
        loss = torch.nn.functional.cross_entropy(
            out[: batch.batch_size], batch.y[: batch.batch_size]
        )
        loss.backward()
        optimizer.step()
        losses.append(float(loss.item()))
        if on_epoch is not None:
            # Warmup probes score several epoch counts from one run: the first
            # k epochs of a longer warmup are exactly a k-epoch warmup.
            on_epoch(epoch + 1, model)

    # run_arm deep-copies this state into every arm and moves it to the
    # device itself, so hand back CPU tensors and keep that contract.
    model.to("cpu")
    return model, losses


def run_arm(
    arm: str,
    cfg: Config,
    dataset: Dataset,
    plan: StreamPlan,
    warm_state: Dict[str, Tensor],
) -> ArmResult:
    """Replay the shared plan under one adaptation policy."""
    if not cfg.drift.detection and arm not in _SIGNAL_FREE_ARMS:
        raise ValueError(
            f"drift.detection is off, but arm {arm!r} fires on the drift signal; "
            f"only {', '.join(_SIGNAL_FREE_ARMS)} can run without detection"
        )
    device = resolve_device(cfg.runtime.device)
    torch.manual_seed(plan.seed)
    model = build_backbone(cfg.model.backbone, dataset.in_dim, dataset.out_dim, cfg.model.hidden, cfg.model.heads)
    model.load_state_dict(copy.deepcopy(warm_state))
    model.to(device)
    # Stream-time retraining uses its own step size. `model.lr` sizes steps
    # for training a fresh model during warmup; these steps are applied to an
    # already-warmed one. Falls back to model.lr when unset, so the Cora runs
    # are unchanged.
    retrain_lr = (
        cfg.model.lr if cfg.adaptation.retrain_lr is None
        else cfg.adaptation.retrain_lr
    )
    optimizer = torch.optim.Adam(
        model.parameters(), lr=retrain_lr, weight_decay=cfg.model.weight_decay
    )

    graph = GraphState(dataset.num_nodes, plan.base_edges)

    # Each arm owns its features and labels, because feature and label drift
    # mutate them in place. The mutations depend only on the plan, never on
    # the model, so every arm still sees an identical sequence of states and
    # the arms stay paired.
    # Kept on CPU: the sampler indexes into these, and batches move to the
    # device after sampling. Holding the whole feature matrix on the GPU does
    # not scale to the graph sizes this targets.
    data = copy.copy(dataset.data)
    data.x = dataset.data.x.clone()
    data.y = dataset.data.y.clone()

    drift_detector = CompositeDriftDetector(
        dataset.out_dim, cfg.drift, use_attention=has_attention(cfg.model.backbone)
    )
    community_backend = build_community_backend(cfg, dataset.num_nodes, plan.seed)
    # Supervised error signal: training-root labels only, for the trigger and/or the region.
    error_signal = None
    if cfg.drift.signal == "error" or cfg.drift.region_signal == "error":
        error_signal = ErrorSignal(cfg.drift.error_window, cfg.drift.error_reference, cfg.drift.error_threshold,
                                   cfg.drift.error_halflife, cfg.drift.error_ref_halflife, cfg.drift.community_selection_k)
        if cfg.drift.region_signal == "error":
            community_backend = ErrorRegionBackend(community_backend, error_signal)
    controller = RetrainingController(cooldown=cfg.adaptation.cooldown)
    retrainer = build_retrainer(
        arm, community_backend, cfg.adaptation, num_hops=len(cfg.stream.num_neighbors)
    )

    # Region adapters (adaptation.region_adapter): keyed by the tracker's stable community ids, so every arm that
    # has one keeps the tracker current, whether or not its retrainer reads communities.
    adapter = adapter_tracker = None
    if cfg.adaptation.region_adapter:
        adapter_tracker = getattr(community_backend, "tracker", None)
        if adapter_tracker is None:
            raise ValueError("adaptation.region_adapter needs the global community tracker (drift.community_mode: global)")
        adapter = RegionAdapter(dataset.out_dim, lambda: adapter_tracker.partition.assignment,
                                lr=cfg.adaptation.adapter_lr, l2=cfg.adaptation.adapter_l2,
                                steps=cfg.adaptation.adapter_steps).to(device)

    # Community drift feeds the gate for every *adapting* arm, so `full` and
    # `selective` fire on exactly the same trigger signal.
    use_community = (
        cfg.drift.gate_uses_community and arm != "baseline"
    ) or retrainer.requires_community or adapter is not None

    background_refresh = use_community and cfg.drift.community_refresh_lag > 0
    # CUDA allocator peak around each retraining event (synchronises; accuracy runs only).
    measure_memory = cfg.runtime.measure_retrain_memory and device.type == "cuda"
    retrain_memory: List[tuple] = []

    sampler = CachedSampler(
        data, cfg.stream.num_neighbors, cfg.stream.sampler_refresh_interval
    )
    timer = StageTimer()
    profiler = StreamProfiler(device, sync_stages=cfg.runtime.sync_stages)
    result = ArmResult(arm=arm, seed=plan.seed)

    # Diagnostic oracles (adaptation.oracle_when / oracle_where): the injected event steps, read from the plan.
    oracle = cfg.adaptation.oracle_when or cfg.adaptation.oracle_where
    event_steps = np.asarray(plan.drift_steps, dtype=np.int64)

    profiler_run = profiler.run()
    profiler_run.__enter__()
    for step in range(plan.steps):
        with timer.stage("graph_update"), profiler.stage("graph_update"):
            graph.add_edges(plan.edges_at(step))
            for event in plan.drift_at(step):
                apply_drift(event, data.x, data.y, graph)
            data.edge_index = graph.edge_index

        # Background community refresh (drift.community_refresh_lag > 0):
        # starts and installs partitions on a fixed step schedule, so its cost
        # is whatever remains on this thread - the hand-off, and any wait for
        # a partition that is not finished when it is due.
        if background_refresh:
            with timer.stage("community_bg"), profiler.stage("community_bg"):
                community_backend.tick(graph.edge_index, step)

        with timer.stage("sample"), profiler.stage("sample"):
            batch = sampler.sample(plan.root_nodes[step], plan.sample_seeds[step], step)
            if device.type != "cpu":
                batch = batch.to(device)
        profiler.record_event()

        with timer.stage("infer"), profiler.stage("infer"):
            out, attn1, _ = infer(model, batch)
            if adapter is not None:
                if adapter_tracker.partition.refreshed_at < 0:
                    adapter_tracker.ensure(graph.edge_index, step)
                with torch.no_grad():
                    out = adapter.apply(out, batch.n_id)
            preds = out.argmax(dim=1)

        reading: DriftReading = _NO_READING
        if cfg.drift.detection:
            with timer.stage("detect"), profiler.stage("detect"):
                reading = drift_detector.update(out, preds, attn1)

        # Community detection is only consulted to decide *where* to retrain,
        # which happens ~30 times in 2000 steps. Running it every step cost
        # ~17 ms/step to save ~3 s of retraining - the detection overhead
        # swamped the saving and made the selective arm the slowest of all.
        # When it does not feed the gate, it is computed on demand instead.
        eager_community = use_community and cfg.drift.gate_uses_community
        community = None
        if eager_community and step % cfg.drift.community_stride == 0:
            with timer.stage("community"), profiler.stage("community"):
                community = community_backend.observe_step(
                    batch=batch,
                    embeddings=out,
                    global_edge_index=graph.edge_index,
                    step=step,
                )

        # --- evaluation: holdout roots only ---------------------------
        with timer.stage("evaluate"), profiler.stage("evaluate"):
            # Masks live on CPU; batch ids may be on the device.
            root_ids = batch.n_id[: batch.batch_size].cpu()
            eval_roots = dataset.eval_mask[root_ids]
            metrics = classification_metrics(
                preds[: batch.batch_size][eval_roots],
                batch.y[: batch.batch_size][eval_roots],
            )

        if error_signal is not None:
            tr = dataset.train_mask[root_ids]
            wrong = (preds[: batch.batch_size] != batch.y[: batch.batch_size]).cpu()[tr].numpy()
            cids = None
            if cfg.drift.region_signal == "error":
                tracker = getattr(community_backend, "tracker", None)
                if tracker is not None:
                    tracker.ensure(graph.edge_index, step)
                    assign = tracker.partition.assignment
                    cids = assign[root_ids[tr]].numpy() if assign is not None and assign.numel() else None
            z = error_signal.update(wrong, cids)
            if cfg.drift.signal == "error":
                reading = DriftReading(0.0, 0.0, 0.0, 0.0, 0.0, 0.0, total=z, flag=error_signal.flag)

        # --- adaptation ----------------------------------------------
        drift_signal = reading.flag or bool(
            cfg.drift.gate_uses_community and community is not None and community.flag
        )
        oracle_nodes = None
        if oracle:
            if cfg.adaptation.oracle_when:
                # Fire as soon as the cooldown allows after any event since this arm's last update.
                last = -1 if controller.last_retrain is None else controller.last_retrain
                drift_signal = bool(((event_steps > last) & (event_steps <= step)).any())
            if cfg.adaptation.oracle_where:
                # The nodes hit in the last 200 steps (the anatomy diagnostic's "recent" window).
                recent = event_steps[(event_steps > step - 200) & (event_steps <= step)]
                if recent.size:
                    oracle_nodes = torch.unique(torch.cat([e.nodes.long() for t in recent for e in plan.drift_at(int(t))]))
        info: Optional[RetrainInfo] = None
        if retrainer.should_fire(step, drift_signal, controller):
            if community is None and use_community:
                with timer.stage("community"), profiler.stage("community"):
                    community = community_backend.observe_step(
                        batch=batch,
                        embeddings=out,
                        global_edge_index=graph.edge_index,
                        step=step,
                    )
            if measure_memory:
                torch.cuda.synchronize(device)
                mem_before = torch.cuda.memory_allocated(device)
                torch.cuda.reset_peak_memory_stats(device)
            with timer.stage("retrain"), profiler.stage("retrain"):
                ctx = RetrainContext(
                    step=step,
                    batch=batch,
                    train_roots=dataset.train_mask[root_ids],
                    community=community,
                    backend=community_backend,
                    sampler=sampler,
                    train_mask=_train_pool(dataset.train_mask, plan, step),
                    seed=plan.seed,
                    oracle_nodes=oracle_nodes,
                    adapter=adapter,
                )
                # Every arm takes the same number of steps per event, so the
                # arms still differ only in *where* the update is computed.
                info = None
                for _ in range(max(1, cfg.adaptation.retrain_steps)):
                    step_info = retrainer.apply(model, optimizer, ctx)
                    if step_info is None:
                        break
                    info = step_info
            if measure_memory:
                torch.cuda.synchronize(device)
                peak = torch.cuda.max_memory_allocated(device)
                if info is not None:
                    retrain_memory.append((peak, peak - mem_before))
            if info is not None:
                controller.commit(step)
                result.retrain_events.append(info)
                # Analytic cost of the subgraph this arm actually materialised.
                # Hardware-independent, so unlike wall-clock it stays
                # meaningful at Cora's scale and transfers to larger graphs.
                profiler.record_cost(
                    backbone_flops(
                        cfg.model.backbone,
                        info.nodes_in_backward,
                        info.edges_in_backward,
                        dataset.in_dim,
                        cfg.model.hidden,
                        dataset.out_dim,
                        cfg.model.heads,
                    ),
                    backbone_activation_bytes(
                        cfg.model.backbone,
                        info.nodes_in_backward,
                        cfg.model.hidden,
                        dataset.out_dim,
                        cfg.model.heads,
                    ),
                )

        result.steps.append(
            StepRecord(
                step=step,
                drift=reading.as_dict(),
                community_max=community.max_score if community else 0.0,
                community_max_z=community.max_z if community else 0.0,
                community_flag=bool(community.flag) if community else False,
                num_communities=community.num_communities if community else 0,
                retrained=info is not None,
                accuracy=metrics.accuracy if metrics else None,
                f1=metrics.f1 if metrics else None,
                precision=metrics.precision if metrics else None,
                recall=metrics.recall if metrics else None,
                eval_support=metrics.support if metrics else 0,
                num_edges=graph.num_directed_edges,
            )
        )

    if background_refresh:
        # A refresh still in flight is charged to this run, not leaked.
        with timer.stage("community_bg"), profiler.stage("community_bg"):
            community_backend.close()
    profiler_run.__exit__(None, None, None)
    result.profile = profiler.summary()
    if measure_memory and retrain_memory:
        peaks = np.array([p for p, _ in retrain_memory], dtype=float)
        deltas = np.array([d for _, d in retrain_memory], dtype=float)
        result.profile["retrain_memory"] = dict(
            peak_bytes_mean=float(peaks.mean()), peak_bytes_max=float(peaks.max()),
            delta_bytes_mean=float(deltas.mean()), delta_bytes_max=float(deltas.max()), events=len(retrain_memory))
    result.timing = timer.summary()
    result.summary = summarise(result, plan, sampler.rebuilds)
    return result


def _nanmean(values: List[Optional[float]]) -> float:
    arr = np.array([v for v in values if v is not None], dtype=float)
    return float(arr.mean()) if arr.size else float("nan")


def summarise(result: ArmResult, plan: StreamPlan, sampler_rebuilds: int = 0) -> Dict[str, float]:
    steps = result.steps
    accuracies = [s.accuracy for s in steps]

    # Recovery adaptivity, now measurable because drift arrives as discrete
    # irregular events rather than on a fixed period.
    recoveries = recovery_steps(accuracies, plan.drift_steps)
    recovered = [r for r in recoveries if r is not None]

    return {
        "recovery_mean_steps": float(np.mean(recovered)) if recovered else float("nan"),
        "recovery_rate": len(recovered) / len(recoveries) if recoveries else float("nan"),
        "drift_events_injected": len(plan.drift_events),
        "arm": result.arm,
        "seed": result.seed,
        "steps": len(steps),
        "accuracy": _nanmean([s.accuracy for s in steps]),
        "precision": _nanmean([s.precision for s in steps]),
        "recall": _nanmean([s.recall for s in steps]),
        "f1": _nanmean([s.f1 for s in steps]),
        "final_accuracy": _nanmean([s.accuracy for s in steps[-100:]]),
        "drift_flags": sum(1 for s in steps if s.drift["flag"]),
        "community_flags": sum(1 for s in steps if s.community_flag),
        "retrain_events": len(result.retrain_events),
        "nodes_targeted_total": sum(e.nodes_targeted for e in result.retrain_events),
        "nodes_updated_total": sum(e.nodes_updated for e in result.retrain_events),
        "nodes_in_backward_total": sum(e.nodes_in_backward for e in result.retrain_events),
        "mean_communities_selected": _nanmean(
            [float(e.communities_selected) for e in result.retrain_events]
        ),
        "mean_communities_total": _nanmean(
            [float(e.communities_total) for e in result.retrain_events]
        ),
        "edges_in_backward_total": sum(e.edges_in_backward for e in result.retrain_events),
        "sampler_rebuilds": sampler_rebuilds,
        "mean_drift_total": _nanmean([s.drift["total"] for s in steps]),
        "max_drift_total": max((s.drift["total"] for s in steps), default=float("nan")),
        "mean_eval_support": _nanmean([float(s.eval_support) for s in steps]),
        "final_edges": steps[-1].num_edges if steps else 0,
        "injected_drift_bursts": len(plan.drift_steps),
    }
