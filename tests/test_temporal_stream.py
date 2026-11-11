"""G2b: the year-ordered stream. Built on Cora with synthetic years (fast, deterministic); the same code runs on
ogbn-arxiv's real node_year.

The leakage test is the one that matters: at step t no node, edge or label that arrives later may appear in an
inference batch or in any retraining batch, including the batches retrainers sample for themselves.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from gist.config import load_config
from gist.data.loaders import load_dataset
from gist.experiments import runner as runner_mod
from gist.experiments.runner import run_arm, warmup_model
from gist.seeding import derive, seed_everything
from gist.stream.temporal import build_temporal_plan, year_windows

CONFIG = "configs/default.yaml"
SEED = 100
STEPS = 120


def _years(n, seed=7):
    # 2000-2009, skewed towards later years like a growing archive.
    return torch.from_numpy(numpy_years(n, seed))


def numpy_years(n, seed):
    rng = np.random.default_rng(seed)
    return np.sort(rng.choice(np.arange(2000, 2010), size=n, p=np.linspace(1, 3, 10) / np.linspace(1, 3, 10).sum()))[rng.permutation(n)]


def _setup(steps=STEPS, years=None, **overrides):
    cfg = load_config(CONFIG, **{"stream.steps": steps, "runtime.device": "cpu", "adaptation.periodic_period": 30,
                                 "adaptation.supervise_halo": False, "adaptation.halo_rehearsal": False, **overrides})
    seed_everything(SEED)
    ds = load_dataset(cfg.data.name, cfg.data.root, cfg.data.train_fraction, split_seed=derive(SEED, 0x51))
    years = _years(ds.num_nodes) if years is None else years
    plan = build_temporal_plan(ds.data.edge_index, years, ds.num_nodes, seed=SEED, steps=steps, batch_size=cfg.stream.batch_size)
    model, _ = warmup_model(cfg, ds, plan, seed=derive(SEED, 0x77))
    warm = {k: v.detach().clone() for k, v in model.state_dict().items()}
    return cfg, ds, plan, warm


class Recorder:
    """Wraps every neighbour-sampling call to record, per call, the latest arrival among the sampled nodes."""

    def __init__(self, plan):
        self.plan, self.calls = plan, []

    def __enter__(self):
        self._orig = runner_mod.CachedSampler.sample
        rec = self

        def sample(sampler, roots, seed, step):
            batch = rec._orig(sampler, roots, seed, step)
            rec.calls.append((step, int(rec.plan.arrival_step[batch.n_id].max()), batch.n_id.clone()))
            return batch

        runner_mod.CachedSampler.sample = sample
        return self

    def __exit__(self, *exc):
        runner_mod.CachedSampler.sample = self._orig


def test_plan_is_time_ordered():
    _, ds, plan, _ = _setup()
    arr, years = plan.arrival_step, plan.node_year
    stream = arr >= 0
    # Nodes sharing a step may straddle a year boundary; across steps, years never go backwards.
    a, y = arr[stream], years[stream]
    per_step_max = torch.full((plan.steps,), -1).scatter_reduce(0, a, y, "amax")
    per_step_min = torch.full((plan.steps,), 10**6).scatter_reduce(0, a, y, "amin")
    used = per_step_max >= 0
    assert torch.all(per_step_max[used][:-1] <= per_step_min[used][1:]), "a later step admitted an earlier year"
    assert torch.all(years[~stream] <= plan.windows["base"][1]) and torch.all(years[stream] > plan.windows["base"][1])
    for t in range(plan.steps):
        e = plan.edges_at(t)
        if e.numel():
            both = torch.maximum(arr[e[0]], arr[e[1]])
            assert torch.all(both == t), f"step {t} released an edge whose later endpoint arrives at another step"
    base = plan.base_edges
    assert torch.all(arr[base[0]] < 0) and torch.all(arr[base[1]] < 0)
    assert torch.all(torch.diff(plan.step_year) >= 0)


def test_all_arms_replay_identical_plan():
    cfg, ds, plan, warm = _setup()
    again = build_temporal_plan(ds.data.edge_index, plan.node_year, ds.num_nodes, seed=SEED, steps=STEPS,
                                batch_size=cfg.stream.batch_size)
    for name in ("root_nodes", "stream_edges", "edge_offsets", "arrival_step", "base_edges"):
        assert torch.equal(getattr(plan, name), getattr(again, name)), name
    seen = {}
    for arm in ("baseline", "periodic", "full"):
        with Recorder(plan) as rec:
            run_arm(arm, cfg, ds, plan, warm)
        # The first sampling call of each step is the inference batch.
        first = {}
        for step, _, n_id in rec.calls:
            first.setdefault(step, n_id)
        seen[arm] = first
    for arm in ("periodic", "full"):
        assert all(torch.equal(seen["baseline"][s], seen[arm][s]) for s in range(STEPS)), arm


@pytest.mark.parametrize("arm", ["baseline", "periodic", "full", "selective"])
def test_no_future_leakage(arm):
    # max_retrain_nodes makes full/periodic draw their own training batch from the pool (_global_batch), and
    # selective draws its region and anchors (_region_batch): the paths that would read the future.
    cfg, ds, plan, warm = _setup(**{"adaptation.max_retrain_nodes": 64, "adaptation.full_retrain_nodes": 64})
    with Recorder(plan) as rec:
        result = run_arm(arm, cfg, ds, plan, warm)
    assert rec.calls, "no sampling call was recorded"
    late = [(s, a) for s, a, _ in rec.calls if a > s]
    assert not late, f"{arm}: batches contained nodes from later steps: {late[:5]}"
    if arm != "baseline":
        assert result.retrain_events, f"{arm} never retrained, so its retraining batches were not exercised"
        # Retraining calls exist beyond the one inference call per step.
        assert len(rec.calls) > STEPS


def test_warmup_uses_only_base_window_nodes(monkeypatch):
    cfg, ds, plan, _ = _setup()
    seen = []
    orig = runner_mod.sample_batch

    def spy(data, roots, num_neighbors, seed):
        b = orig(data, roots, num_neighbors, seed)
        seen.append(int(plan.arrival_step[b.n_id].max()))
        return b

    monkeypatch.setattr(runner_mod, "sample_batch", spy)
    warmup_model(cfg, ds, plan, seed=derive(SEED, 0x77))
    assert seen and max(seen) < 0


def test_calibration_and_evaluation_windows_are_disjoint():
    w = year_windows(np.array([2000] * 30 + [2001] * 10 + [2002] * 10 + [2003] * 10 + [2004] * 40))
    assert w == {"base": (2000, 2000), "calibration": (2001, 2002), "evaluation": (2003, 2004)}
    _, _, plan, _ = _setup()
    (b0, b1), (c0, c1), (e0, e1) = plan.windows["base"], plan.windows["calibration"], plan.windows["evaluation"]
    assert b1 < c0 <= c1 < e0 <= e1
    years = plan.node_year
    sizes = [int(((years >= lo) & (years <= hi)).sum()) for lo, hi in plan.windows.values()]
    assert sum(sizes) == plan.num_nodes, "windows must cover every node exactly once"


def test_single_year_is_a_static_graph():
    cfg, ds, plan, warm = _setup(steps=30, years=torch.full((2708,), 2010))
    assert plan.stream_edges.shape[1] == 0 and torch.all(plan.arrival_step == -1)
    result = run_arm("periodic", cfg, ds, plan, warm)
    assert result.summary["steps"] == 30
