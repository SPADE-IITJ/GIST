"""The layer runs any registered backbone end to end (C1), its update is exact for each (C4), and the two C3
controls do what they claim. CPU, Cora, short streams.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from gist.adaptation.subgraph import audit_exactness, subgraph_gradient_step
from gist.config import load_config
from gist.data.loaders import load_dataset
from gist.drift.detector import CompositeDriftDetector
from gist.experiments.runner import run_arm, sample_batch, warmup_model
from gist.models.registry import BACKBONES, build_backbone, has_attention
from gist.seeding import derive, seed_everything
from gist.stream.graph_state import GraphState
from gist.stream.schedule import build_plan

SEED = 100
STEPS = 150


def _setup(backbone="gat", **extra):
    over = {"stream.steps": STEPS, "runtime.device": "cpu", "model.backbone": backbone, "model.warmup_epochs": 3,
            "model.warmup_symmetric": True, "adaptation.supervise_halo": False, "adaptation.halo_rehearsal": False,
            "drift.total_threshold": 0.5, "adaptation.cooldown": 20, **extra}
    cfg = load_config("configs/default.yaml", **over)
    seed_everything(SEED)
    ds = load_dataset(cfg.data.name, cfg.data.root, cfg.data.train_fraction, split_seed=derive(SEED, 0x51))
    plan = build_plan(ds.data.edge_index, ds.num_nodes, seed=SEED, steps=STEPS, batch_size=cfg.stream.batch_size,
                      base_edge_fraction=cfg.data.base_edge_fraction, drift_cfg=cfg.drift_process,
                      x=ds.data.x, y=ds.data.y, num_classes=ds.out_dim)
    model, _ = warmup_model(cfg, ds, plan, seed=derive(SEED, 0x77))
    return cfg, ds, plan, {k: v.detach().clone() for k, v in model.state_dict().items()}


@pytest.fixture(scope="module", params=BACKBONES)
def cell(request):
    return request.param, _setup(request.param, **{"adaptation.exactness_audit": 2})


def test_every_backbone_runs_every_arm(cell):
    name, (cfg, ds, plan, warm) = cell
    events = {}
    for arm in ("baseline", "full", "selective", "random_region"):
        r = run_arm(arm, cfg, ds, plan, warm)
        assert len(r.steps) == STEPS
        events[arm] = len(r.retrain_events)
        if not has_attention(name):
            assert all(s.drift["attention_z"] == 0.0 for s in r.steps)
    assert events["baseline"] == 0 and events["full"] > 0 and events["selective"] > 0
    # Same detector, so the trigger fires at the same steps whatever the region choice.
    sel = run_arm("selective", cfg, ds, plan, warm)
    rnd = run_arm("random_region", cfg, ds, plan, warm)
    assert [e.step for e in sel.retrain_events][:1] == [e.step for e in rnd.retrain_events][:1]


def test_in_stream_audit_is_exact(cell):
    name, (cfg, ds, plan, warm) = cell
    r = run_arm("selective", cfg, ds, plan, warm)
    audits = [e.audit for e in r.retrain_events if e.audit is not None]
    assert len(audits) == 2, name
    for a in audits:
        assert a["rel_diff_global"] < 1e-8, (name, a)
        assert a["rel_diff_short_global"] > 1e-3, (name, a)
        assert a["rel_diff_global"] <= a["rel_diff"]          # global scale never exceeds the per-tensor ratio


@pytest.mark.parametrize("name", BACKBONES)
def test_subgraph_step_gradient_equals_full_batch(name):
    """The production step (not just the audit): its gradient equals the full-batch masked gradient."""
    cfg = load_config("configs/default.yaml", **{"stream.steps": 10})
    seed_everything(0)
    ds = load_dataset(cfg.data.name, cfg.data.root, cfg.data.train_fraction, split_seed=derive(0, 0x51))
    plan = build_plan(ds.data.edge_index, ds.num_nodes, seed=0, steps=10, batch_size=cfg.stream.batch_size,
                      base_edge_fraction=cfg.data.base_edge_fraction, drift_cfg=cfg.drift_process,
                      x=ds.data.x, y=ds.data.y, num_classes=ds.out_dim)
    data = ds.data
    data.edge_index = GraphState(ds.num_nodes, plan.base_edges).edge_index
    batch = sample_batch(data, plan.root_nodes[0], cfg.stream.num_neighbors, plan.sample_seeds[0])
    batch.x = batch.x.double()
    torch.manual_seed(0)
    model = build_backbone(name, ds.in_dim, ds.out_dim, 16, 2).double()
    targets = torch.zeros(batch.batch_size, dtype=torch.bool)
    targets[::7] = True
    model.zero_grad()
    out = model(batch.x, batch.edge_index)[0]
    torch.nn.functional.cross_entropy(out[targets.nonzero(as_tuple=True)[0]], batch.y[targets.nonzero(as_tuple=True)[0]]).backward()
    ref = {k: p.grad.clone() for k, p in model.named_parameters()}
    opt = torch.optim.SGD(model.parameters(), lr=0.0)
    res = subgraph_gradient_step(model, opt, batch, targets, num_hops=2)
    assert res is not None and res[1].nodes_in_backward < batch.num_nodes
    rel = max(float((ref[k] - p.grad).abs().max() / ref[k].abs().max()) for k, p in model.named_parameters())
    assert rel < 1e-8, (name, rel)


def test_detector_without_attention_uses_two_signals():
    cfg = load_config("configs/default.yaml")
    det = CompositeDriftDetector(7, cfg.drift, use_attention=False)
    assert det._w_att == 0.0 and abs(det._w_emb + det._w_pred - 1.0) < 1e-12
    out = torch.randn(50, 7)
    r = det.update(out, out.argmax(1), None)
    assert r.attention_raw == 0.0 and r.attention_z == 0.0
    with_att = CompositeDriftDetector(7, cfg.drift)
    assert abs(with_att._w_att - 1 / 3) < 1e-12           # GAT's three-signal weighting is unchanged


def test_iso_full_matches_its_node_target():
    cfg, ds, plan, warm = _setup("sage", **{"adaptation.full_retrain_nodes": 128})
    full = run_arm("full", cfg, ds, plan, warm)
    target = int(sum(e.nodes_in_backward for e in full.retrain_events) / len(full.retrain_events) * 0.6)
    cfg.adaptation.iso_target_nodes = target
    iso = run_arm("iso_full", cfg, ds, plan, warm)
    assert iso.retrain_events
    got = sum(e.nodes_in_backward for e in iso.retrain_events) / len(iso.retrain_events)
    assert abs(got - target) / target < 0.15, (got, target)
    assert all(e.roots_drawn < 128 for e in iso.retrain_events)
    # Same trigger until the first update; after it the models differ, and so do the detector's readings.
    assert iso.retrain_events[0].step == full.retrain_events[0].step


def test_random_region_changes_where_not_when():
    cfg, ds, plan, warm = _setup("gat")
    sel = run_arm("selective", cfg, ds, plan, warm)
    rnd = run_arm("random_region", cfg, ds, plan, warm)
    # Identical up to the first update: same step, same number of communities (only which ones differs).
    assert sel.retrain_events[0].step == rnd.retrain_events[0].step
    assert sel.retrain_events[0].communities_selected == rnd.retrain_events[0].communities_selected
