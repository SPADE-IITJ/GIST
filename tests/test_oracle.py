"""Diagnostic oracles for the C2/C3 ceilings (adaptation.oracle_when / oracle_where; phase 0)."""
from __future__ import annotations

import torch

from gist.adaptation.retrainers import RetrainContext, SelectiveRetrainer


def _small_run(**extra):
    from gist.config import load_config
    from gist.data.loaders import load_dataset
    from gist.experiments.runner import run_arm, warmup_model
    from gist.seeding import derive, seed_everything
    from gist.stream.schedule import build_plan
    over = {"stream.steps": 400, "runtime.device": "cpu", "model.warmup_epochs": 3, "adaptation.cooldown": 20, **extra}
    cfg = load_config("configs/default.yaml", **over)
    seed_everything(100)
    ds = load_dataset(cfg.data.name, cfg.data.root, cfg.data.train_fraction, split_seed=derive(100, 0x51))
    plan = build_plan(ds.data.edge_index, ds.num_nodes, seed=100, steps=400, batch_size=cfg.stream.batch_size,
                      base_edge_fraction=cfg.data.base_edge_fraction, drift_cfg=cfg.drift_process,
                      x=ds.data.x, y=ds.data.y, num_classes=ds.out_dim)
    model, _ = warmup_model(cfg, ds, plan, seed=derive(100, 0x77))
    warm = {k: v.detach().clone() for k, v in model.state_dict().items()}
    return cfg, ds, plan, warm, run_arm


def test_oracle_when_fires_only_after_a_fresh_event():
    cfg, ds, plan, warm, run_arm = _small_run(**{"adaptation.oracle_when": True})
    r = run_arm("full", cfg, ds, plan, warm)
    fired = [e.step for e in r.retrain_events]
    events = plan.drift_steps
    assert fired and events, "need at least one event and one update"
    prev = -1
    for t in fired:
        assert any(prev < s <= t for s in events), f"update at {t} with no event since {prev}"
        assert prev < 0 or t - prev > cfg.adaptation.cooldown
        prev = t
    assert fired[0] >= events[0]


class _Backend:
    def community_nodes(self, drifting, restrict_mask=None, limit=None, generator=None):
        raise AssertionError("oracle_where must not consult the drift score's communities")


class _Sampler:
    def sample(self, nodes, seed, step):
        self.nodes = nodes
        return nodes


def test_oracle_where_targets_only_drifted_trainable_nodes():
    train = torch.zeros(100, dtype=torch.bool)
    train[::2] = True                                   # even ids are trainable
    drifted = torch.arange(10, 40)
    sampler = _Sampler()
    sel = SelectiveRetrainer(_Backend(), max_nodes=8, anchor_fraction=0.0)
    ctx = RetrainContext(step=5, batch=None, train_roots=None, sampler=sampler, train_mask=train, seed=1,
                         oracle_nodes=drifted)
    sel._region_batch(ctx, drifting=[0])
    picked = sampler.nodes
    assert picked.numel() == 8
    assert bool(train[picked].all()) and bool(((picked >= 10) & (picked < 40)).all())
