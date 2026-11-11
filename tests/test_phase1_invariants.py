"""Tests for the Phase-1 validity guarantees.

Phase 1's whole claim is that a measured difference between arms is
attributable to the adaptation policy. These tests check the properties that
claim depends on, rather than taking the design on trust.

Run with:  python -m pytest tests/ -v
"""

from __future__ import annotations

import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from gist.adaptation.controller import RetrainingController
from gist.config import load_config
from gist.data.loaders import load_dataset, make_node_split
from gist.drift.detector import CompositeDriftDetector
from gist.drift.signals import RunningStandardiser
from gist.experiments.runner import run_arm, sample_batch, warmup_model
from gist.seeding import derive, seed_everything
from gist.stream.graph_state import GraphState
from gist.stream.schedule import build_plan

CONFIG = "configs/default.yaml"
SEED = 0


def _setup(steps: int = 60):
    cfg = load_config(CONFIG, **{"stream.steps": steps})
    seed_everything(SEED)
    dataset = load_dataset(
        cfg.data.name, cfg.data.root, cfg.data.train_fraction, split_seed=derive(SEED, 0x51)
    )
    plan = build_plan(
        dataset.data.edge_index,
        dataset.num_nodes,
        seed=SEED,
        steps=cfg.stream.steps,
        batch_size=cfg.stream.batch_size,
        base_edge_fraction=cfg.data.base_edge_fraction,
        drift_cfg=cfg.drift_process,
        x=dataset.data.x,
        y=dataset.data.y,
        num_classes=dataset.out_dim,
    )
    return cfg, dataset, plan


# --- split integrity ---------------------------------------------------


def test_train_and_eval_masks_are_disjoint_and_cover_all_nodes():
    train, evalm = make_node_split(1000, 0.6, seed=7)
    assert not bool((train & evalm).any()), "a node is both trained on and scored"
    assert int((train | evalm).sum()) == 1000
    assert int(train.sum()) == 600


def test_split_is_reproducible():
    a = make_node_split(500, 0.6, seed=3)
    b = make_node_split(500, 0.6, seed=3)
    assert torch.equal(a[0], b[0]) and torch.equal(a[1], b[1])


# --- the shared stream plan --------------------------------------------


def test_plan_is_deterministic_for_a_seed():
    _, _, p1 = _setup()
    _, _, p2 = _setup()
    assert torch.equal(p1.stream_edges, p2.stream_edges)
    assert torch.equal(p1.root_nodes, p2.root_nodes)
    assert torch.equal(p1.base_edges, p2.base_edges)
    assert p1.sample_seeds == p2.sample_seeds
    assert p1.drift_steps == p2.drift_steps
    for e1, e2 in zip(p1.drift_events, p2.drift_events):
        assert (e1.step, e1.kind, e1.regime) == (e2.step, e2.kind, e2.regime)
        assert torch.equal(e1.nodes, e2.nodes)


def test_base_snapshot_and_stream_are_disjoint():
    _, _, plan = _setup()

    def pairs(t):
        return {tuple(sorted(p)) for p in t.t().tolist()}

    assert not (pairs(plan.base_edges) & pairs(plan.stream_edges)), (
        "an edge appears in both G_0 and the stream"
    )


def test_base_snapshot_is_non_empty():
    """The prototype started from an empty graph; G_0 must be populated."""
    _, _, plan = _setup()
    assert plan.base_edges.shape[1] > 0


# --- determinism of sampling -------------------------------------------


def test_identical_batches_across_arms():
    """Two independent samples with the same plan seed must match exactly."""
    cfg, dataset, plan = _setup()
    data = dataset.data
    data.edge_index = plan.base_edges

    b1 = sample_batch(data, plan.root_nodes[0], cfg.stream.num_neighbors, plan.sample_seeds[0])
    b2 = sample_batch(data, plan.root_nodes[0], cfg.stream.num_neighbors, plan.sample_seeds[0])
    assert torch.equal(b1.n_id, b2.n_id)
    assert torch.equal(b1.edge_index, b2.edge_index)


# --- graph state --------------------------------------------------------


def test_graph_state_deduplicates_and_symmetrises():
    g = GraphState(10)
    g.add_edge(1, 2)
    added_again = g.add_edge(1, 2)
    reversed_add = g.add_edge(2, 1)
    assert g.num_directed_edges == 2, "expected exactly (1,2) and (2,1)"
    assert added_again == 0 and reversed_add == 0


def test_graph_state_ignores_self_loops():
    g = GraphState(10)
    assert g.add_edge(3, 3) == 0
    assert g.num_directed_edges == 0


# --- drift detector -----------------------------------------------------


def test_standardiser_approaches_unit_variance():
    import numpy as np

    rng = np.random.default_rng(0)
    norm = RunningStandardiser(ema=0.05, warmup=50)
    scores = [norm.update(float(v)) for v in rng.normal(5.0, 2.0, size=4000)]
    tail = np.array(scores[1000:])
    assert abs(tail.mean()) < 0.3, f"mean drifted to {tail.mean()}"
    assert 0.6 < tail.std() < 1.6, f"std was {tail.std()}"


def test_threshold_is_invariant_to_weight_spread():
    """Equal weights must not be implicitly stricter than a single signal.

    Without the 1/||w||_2 rescaling, spreading weight over three signals
    shrinks the composite's variance and silently raises the effective
    threshold, which would corrupt any alpha/beta/gamma ablation.
    """
    cfg = load_config(CONFIG)

    cfg.drift.weight_embedding, cfg.drift.weight_prediction, cfg.drift.weight_attention = 1, 1, 1
    equal = CompositeDriftDetector(7, cfg.drift)

    cfg.drift.weight_embedding, cfg.drift.weight_prediction, cfg.drift.weight_attention = 1, 0, 0
    single = CompositeDriftDetector(7, cfg.drift)

    # A unit z on every signal must map to the same composite as a unit z on
    # the only active signal.
    assert abs(equal._scale - (3 * (1 / 3) ** 2) ** 0.5) < 1e-9
    assert abs(single._scale - 1.0) < 1e-9
    assert abs((1.0 / equal._scale) * (1 / 3) * 3 - 1.7320508) < 1e-4


def test_detector_stays_silent_during_warmup():
    cfg = load_config(CONFIG)
    det = CompositeDriftDetector(7, cfg.drift)
    torch.manual_seed(0)
    for _ in range(cfg.drift.warmup):
        emb = torch.randn(32, 7) * 100  # wildly unstable input
        reading = det.update(emb, emb.argmax(1), (None, torch.rand(64, 1)))
        assert not reading.flag, "detector fired before it had a reference distribution"


# --- controller ---------------------------------------------------------


def test_cooldown_is_not_consumed_by_a_noop():
    """A permitted-but-unused retraining slot must not arm the cooldown."""
    c = RetrainingController(cooldown=50)
    assert c.should_retrain(True, step=10)
    # Selective policy found nothing to do -> no commit.
    assert c.should_retrain(True, step=11), "cooldown was armed by a no-op"
    c.commit(11)
    assert not c.should_retrain(True, step=30)
    assert c.should_retrain(True, step=62)


def test_no_drift_means_no_retrain():
    c = RetrainingController(cooldown=10)
    assert not c.should_retrain(False, step=500)


# --- end-to-end ---------------------------------------------------------


def test_baseline_never_retrains_and_arms_are_reproducible():
    cfg, dataset, plan = _setup(steps=40)
    model, _ = warmup_model(cfg, dataset, plan, seed=derive(SEED, 0x77))
    warm = {k: v.detach().clone() for k, v in model.state_dict().items()}

    baseline = run_arm("baseline", cfg, dataset, plan, warm)
    assert baseline.retrain_events == []

    again = run_arm("baseline", cfg, dataset, plan, warm)
    assert [s.accuracy for s in baseline.steps] == [s.accuracy for s in again.steps]


def test_arms_diverge_only_after_first_retrain():
    """Before any arm adapts, all arms must be numerically identical."""
    cfg, dataset, plan = _setup(steps=120)
    model, _ = warmup_model(cfg, dataset, plan, seed=derive(SEED, 0x77))
    warm = {k: v.detach().clone() for k, v in model.state_dict().items()}

    baseline = run_arm("baseline", cfg, dataset, plan, warm)
    full = run_arm("full", cfg, dataset, plan, warm)

    first = full.retrain_events[0].step if full.retrain_events else len(full.steps)
    for i in range(first + 1):
        assert baseline.steps[i].accuracy == full.steps[i].accuracy, (
            f"arms differ at step {i}, before the first retrain at {first}"
        )


def test_retraining_gate_does_not_saturate():
    """A gate that fires on nearly every step is not a drift detector.

    The first Phase-1 run had a static community threshold that fired on 98%
    of steps, which made "drift-triggered" retraining indistinguishable from
    retraining every `cooldown` steps. Guard against regressing to that.
    """
    cfg, dataset, plan = _setup(steps=300)
    model, _ = warmup_model(cfg, dataset, plan, seed=derive(SEED, 0x77))
    warm = {k: v.detach().clone() for k, v in model.state_dict().items()}

    result = run_arm("selective", cfg, dataset, plan, warm)
    scored = result.steps[cfg.drift.warmup:]
    flag_rate = sum(s.drift["flag"] for s in scored) / len(scored)
    community_rate = sum(s.community_flag for s in scored) / len(scored)

    assert 0.0 < flag_rate < 0.30, f"composite gate fired on {flag_rate:.1%} of steps"
    assert community_rate < 0.30, f"community gate fired on {community_rate:.1%} of steps"


def test_selective_targets_a_minority_of_communities():
    """"Localized" must mean localized - check it against the actual partition."""
    cfg, dataset, plan = _setup(steps=300)
    model, _ = warmup_model(cfg, dataset, plan, seed=derive(SEED, 0x77))
    warm = {k: v.detach().clone() for k, v in model.state_dict().items()}

    result = run_arm("selective", cfg, dataset, plan, warm)
    events = [e for e in result.retrain_events if e.communities_total > 0]
    assert events, "selective arm never retrained; cannot assess localization"

    share = sum(e.communities_selected for e in events) / sum(
        e.communities_total for e in events
    )
    assert share < 0.5, f"selected {share:.1%} of communities - not localized"


def test_periodic_arm_ignores_the_drift_signal():
    """The control arm must fire on its schedule regardless of drift."""
    from gist.adaptation.retrainers import PeriodicRetrainer

    r = PeriodicRetrainer(period=10)
    assert r.should_fire(10, drift_signal=False, controller=None)
    assert r.should_fire(20, drift_signal=False, controller=None)
    assert not r.should_fire(15, drift_signal=True, controller=None)
    assert not r.should_fire(0, drift_signal=True, controller=None)


def test_evaluation_never_scores_a_training_node():
    """The contamination the prototype had must be structurally impossible."""
    cfg, dataset, plan = _setup(steps=20)
    overlap = dataset.train_mask & dataset.eval_mask
    assert not bool(overlap.any())


# --- Phase 1.5: the drift process --------------------------------------


def _plan_with(mode: str, steps: int = 2000, locality: str = "localized", seed: int = SEED):
    cfg = load_config(CONFIG, **{"stream.steps": steps})
    cfg.drift_process.mode = mode
    cfg.drift_process.locality = locality
    seed_everything(seed)
    dataset = load_dataset(
        cfg.data.name, cfg.data.root, cfg.data.train_fraction, split_seed=derive(seed, 0x51)
    )
    plan = build_plan(
        dataset.data.edge_index,
        dataset.num_nodes,
        seed=seed,
        steps=steps,
        batch_size=cfg.stream.batch_size,
        base_edge_fraction=cfg.data.base_edge_fraction,
        drift_cfg=cfg.drift_process,
        x=dataset.data.x,
        y=dataset.data.y,
        num_classes=dataset.out_dim,
    )
    return cfg, dataset, plan


def test_regime_drift_is_burstier_than_poisson_which_beats_periodic():
    """The whole point of Phase 1.5: drift must not arrive on a timer.

    Coefficient of variation of the inter-arrival gaps is 0 for a fixed
    period (which a periodic retrainer matches exactly), ~1 for a Poisson
    process, and >1 when arrivals cluster into bursts.
    """
    cvs = {}
    for mode in ["periodic", "poisson", "regime"]:
        _, _, plan = _plan_with(mode)
        cvs[mode] = plan.describe()["gap_cv"]

    assert cvs["periodic"] == 0.0, "periodic drift should be perfectly regular"
    assert cvs["poisson"] > 0.6, f"poisson CV was {cvs['poisson']}"
    assert cvs["regime"] > cvs["poisson"], (
        f"regime drift ({cvs['regime']}) should be burstier than poisson "
        f"({cvs['poisson']})"
    )


def test_regime_drift_has_both_calm_and_turbulent_periods():
    _, _, plan = _plan_with("regime")
    regimes = plan.describe()["regimes"]
    assert regimes.get("calm", 0) > 0 and regimes.get("turbulent", 0) > 0


def test_all_three_drift_kinds_are_generated():
    _, _, plan = _plan_with("regime")
    kinds = plan.describe()["kinds"]
    assert set(kinds) == {"structural", "feature", "label"}, kinds


def test_localized_drift_is_spatially_concentrated():
    """Localized events must hit far fewer distinct nodes than diffuse ones."""
    _, _, local = _plan_with("regime", locality="localized")
    _, _, diffuse = _plan_with("regime", locality="diffuse")

    def spread(plan):
        seen = set()
        for e in plan.drift_events:
            seen.update(e.nodes.tolist())
        return len(seen) / max(sum(e.nodes.numel() for e in plan.drift_events), 1)

    assert spread(local) < spread(diffuse), (
        "localized events should revisit the same regions; "
        f"got {spread(local):.3f} vs {spread(diffuse):.3f}"
    )


def test_feature_drift_actually_changes_features():
    from gist.stream.drift_process import apply_drift

    _, dataset, plan = _plan_with("regime")
    event = next(e for e in plan.drift_events if e.kind == "feature")
    x, y = dataset.data.x.clone(), dataset.data.y.clone()
    before = x[event.nodes].clone()
    apply_drift(event, x, y, GraphState(dataset.num_nodes))
    assert not torch.allclose(before, x[event.nodes]), "feature drift was a no-op"
    untouched = torch.ones(x.shape[0], dtype=torch.bool)
    untouched[event.nodes] = False
    assert torch.allclose(x[untouched], dataset.data.x[untouched]), (
        "feature drift leaked outside its target set"
    )


def test_label_drift_is_a_real_concept_change():
    from gist.stream.drift_process import apply_drift

    _, dataset, plan = _plan_with("regime")
    event = next(e for e in plan.drift_events if e.kind == "label")
    x, y = dataset.data.x.clone(), dataset.data.y.clone()
    before = y[event.nodes].clone()
    apply_drift(event, x, y, GraphState(dataset.num_nodes))
    assert not torch.equal(before, y[event.nodes]), "label drift was a no-op"
    assert int(y.max()) < dataset.out_dim, "label drift produced an invalid class"


def test_structural_drift_breaks_homophily():
    """Injected edges should mostly join nodes of differing classes."""
    _, dataset, plan = _plan_with("regime")
    y = dataset.data.y
    same, total = 0, 0
    for event in plan.drift_events:
        if event.kind != "structural" or event.edges is None:
            continue
        src, dst = event.edges[0], event.edges[1]
        same += int((y[src] == y[dst]).sum())
        total += int(src.numel())
    assert total > 0
    assert same / total < 0.2, f"{same / total:.1%} of injected edges were homophilous"


def test_drift_application_is_model_independent():
    """Two arms must see identical feature/label states, preserving pairing."""
    from gist.stream.drift_process import apply_drift

    _, dataset, plan = _plan_with("regime", steps=300)

    def replay():
        x, y = dataset.data.x.clone(), dataset.data.y.clone()
        graph = GraphState(dataset.num_nodes, plan.base_edges)
        for step in range(300):
            graph.add_edges(plan.edges_at(step))
            for event in plan.drift_at(step):
                apply_drift(event, x, y, graph)
        return x, y, graph.edge_index

    x1, y1, e1 = replay()
    x2, y2, e2 = replay()
    assert torch.equal(x1, x2) and torch.equal(y1, y2) and torch.equal(e1, e2)
