"""Phase 2 tests: induced-subgraph retraining and stable global communities.

The central claim is that localized retraining is a *cost* optimisation, not
an approximation: the gradient computed on the targets' induced subgraph must
equal the gradient of the same masked loss computed over the whole batch.
If that holds, every saving is work not done rather than fidelity given up.
"""

from __future__ import annotations

import sys
from pathlib import Path

import torch
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from gist.adaptation.subgraph import (
    extract_subgraph,
    full_batch_stats,
    subgraph_gradient_step,
)
from gist.config import load_config
from gist.data.loaders import load_dataset
from gist.drift.global_communities import GlobalCommunityTracker
from gist.experiments.runner import sample_batch
from gist.models.gat import build_model
from gist.seeding import derive, seed_everything
from gist.stream.graph_state import GraphState
from gist.stream.schedule import build_plan

CONFIG = "configs/default.yaml"
SEED = 0
NUM_HOPS = 2  # must match the GAT's depth


def _batch_and_model(steps: int = 60):
    cfg = load_config(CONFIG, **{"stream.steps": steps})
    seed_everything(SEED)
    dataset = load_dataset(
        cfg.data.name, cfg.data.root, cfg.data.train_fraction, split_seed=derive(SEED, 0x51)
    )
    plan = build_plan(
        dataset.data.edge_index,
        dataset.num_nodes,
        seed=SEED,
        steps=steps,
        batch_size=cfg.stream.batch_size,
        base_edge_fraction=cfg.data.base_edge_fraction,
        drift_cfg=cfg.drift_process,
        x=dataset.data.x,
        y=dataset.data.y,
        num_classes=dataset.out_dim,
    )

    graph = GraphState(dataset.num_nodes, plan.base_edges)
    data = dataset.data
    data.edge_index = graph.edge_index
    batch = sample_batch(data, plan.root_nodes[0], cfg.stream.num_neighbors, plan.sample_seeds[0])

    torch.manual_seed(SEED)
    model = build_model(dataset.in_dim, dataset.out_dim, cfg.model.hidden, cfg.model.heads)
    return cfg, dataset, plan, graph, batch, model


def _grads(model) -> dict[str, torch.Tensor]:
    return {n: p.grad.detach().clone() for n, p in model.named_parameters() if p.grad is not None}


def _targets(batch, every: int = 7) -> torch.Tensor:
    mask = torch.zeros(batch.batch_size, dtype=torch.bool)
    mask[torch.arange(0, batch.batch_size, every)] = True
    return mask


# --- the exactness claim -----------------------------------------------


def test_subgraph_gradient_equals_full_batch_gradient():
    """The whole Phase 2 argument: same gradient, less work."""
    _, _, _, _, batch, model = _batch_and_model()
    targets = _targets(batch)

    # Reference: masked loss over the full batch.
    model.zero_grad()
    out, _, _ = model(batch.x, batch.edge_index)
    F.cross_entropy(out[: batch.batch_size][targets], batch.y[: batch.batch_size][targets]).backward()
    reference = _grads(model)

    # Subgraph: same loss, computed on the induced 2-hop neighbourhood only.
    subset, sub_edge_index, mapping = extract_subgraph(batch, targets, NUM_HOPS)
    model.zero_grad()
    sub_out, _, _ = model(batch.x[subset], sub_edge_index)
    F.cross_entropy(sub_out[mapping], batch.y[subset][mapping]).backward()
    localized = _grads(model)

    assert set(reference) == set(localized)
    for name, grad in reference.items():
        assert torch.allclose(grad, localized[name], atol=1e-5, rtol=1e-4), (
            f"gradient mismatch on {name}: "
            f"max |delta| = {(grad - localized[name]).abs().max():.3e}"
        )


def test_subgraph_forward_matches_full_forward_at_targets():
    """Outputs at the targets must be identical, not merely close in loss."""
    _, _, _, _, batch, model = _batch_and_model()
    targets = _targets(batch)

    with torch.no_grad():
        out, _, _ = model(batch.x, batch.edge_index)
        subset, sub_edge_index, mapping = extract_subgraph(batch, targets, NUM_HOPS)
        sub_out, _, _ = model(batch.x[subset], sub_edge_index)

    assert torch.allclose(
        out[: batch.batch_size][targets], sub_out[mapping], atol=1e-5, rtol=1e-4
    )


def test_insufficient_hops_would_break_exactness():
    """Guards the choice of num_hops against silent under-extraction.

    A 1-hop subgraph omits activations the second GAT layer needs, so it must
    *not* reproduce the full-batch output. If this ever passes, num_hops has
    been decoupled from the model depth and the exactness claim is void.
    """
    _, _, _, _, batch, model = _batch_and_model()
    targets = _targets(batch)

    with torch.no_grad():
        out, _, _ = model(batch.x, batch.edge_index)
        subset, sub_edge_index, mapping = extract_subgraph(batch, targets, 1)
        sub_out, _, _ = model(batch.x[subset], sub_edge_index)

    assert not torch.allclose(
        out[: batch.batch_size][targets], sub_out[mapping], atol=1e-5, rtol=1e-4
    ), "a 1-hop subgraph reproduced a 2-layer model's output; check the model depth"


# --- the cost claim -----------------------------------------------------


def test_subgraph_is_materially_smaller_than_the_batch():
    _, _, _, _, batch, model = _batch_and_model()
    targets = _targets(batch)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

    result = subgraph_gradient_step(model, optimizer, batch, targets, NUM_HOPS)
    assert result is not None
    _, stats = result

    assert stats.nodes_in_backward < stats.nodes_in_batch
    assert stats.edges_in_backward < stats.edges_in_batch
    assert stats.nodes_targeted == int(targets.sum())


def test_full_batch_stats_charge_every_sampled_node():
    """Phase 1 charged only the roots, understating full retraining's cost."""
    _, _, _, _, batch, _ = _batch_and_model()
    stats = full_batch_stats(batch)
    assert stats.nodes_in_backward == batch.num_nodes
    assert stats.nodes_in_backward > batch.batch_size


def test_no_targets_is_a_noop():
    _, _, _, _, batch, model = _batch_and_model()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    empty = torch.zeros(batch.batch_size, dtype=torch.bool)
    assert subgraph_gradient_step(model, optimizer, batch, empty, NUM_HOPS) is None


# --- stable global communities -----------------------------------------


def test_global_partition_is_coarser_than_per_batch_louvain():
    """Phase 1's per-batch Louvain returned ~60-70 tiny communities."""
    _, dataset, _, graph, _, _ = _batch_and_model()
    tracker = GlobalCommunityTracker(dataset.num_nodes, min_size=15, seed=SEED)
    partition = tracker.refresh(graph.edge_index, step=0)

    assert partition.num_communities > 0
    assert partition.num_communities < 40, (
        f"expected a coarse global partition, got {partition.num_communities}"
    )


def test_community_ids_survive_a_refresh():
    """Stable ids are what make per-community drift a time series."""
    _, dataset, plan, graph, _, _ = _batch_and_model()
    tracker = GlobalCommunityTracker(dataset.num_nodes, min_size=15, seed=SEED)
    first = tracker.refresh(graph.edge_index, step=0)

    for step in range(200):
        graph.add_edges(plan.edges_at(step % plan.steps))
    second = tracker.refresh(graph.edge_index, step=200)

    before, after = first.assignment, second.assignment
    tracked = (before >= 0) & (after >= 0)
    agreement = float((before[tracked] == after[tracked]).float().mean())
    assert agreement > 0.7, f"only {agreement:.1%} of nodes kept their community id"


def test_refresh_is_interval_gated():
    """Louvain dominated step time in Phase 1; it must not run every step."""
    _, dataset, _, graph, _, _ = _batch_and_model()
    tracker = GlobalCommunityTracker(dataset.num_nodes, refresh_interval=200, seed=SEED)

    assert tracker.ensure(graph.edge_index, step=0) is True
    assert tracker.ensure(graph.edge_index, step=50) is False
    assert tracker.ensure(graph.edge_index, step=199) is False
    assert tracker.ensure(graph.edge_index, step=200) is True
    assert tracker.partition.refresh_count == 2


def test_community_drift_is_keyed_by_stable_id_and_tracked_over_time():
    _, dataset, plan, graph, batch, model = _batch_and_model()
    tracker = GlobalCommunityTracker(dataset.num_nodes, seed=SEED)
    tracker.refresh(graph.edge_index, step=0)

    with torch.no_grad():
        out, _, _ = model(batch.x, batch.edge_index)

    first = tracker.observe(batch.n_id, out, step=0)
    second = tracker.observe(batch.n_id, out * 1.5, step=1)

    assert first.num_observed > 0
    assert set(second.scores).issubset(set(tracker.partition.sizes))
    # A node seen twice has a drift history under a persistent id.
    assert any(len(v) >= 2 for v in tracker.history.values())
