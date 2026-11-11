"""The supervised error signal: flags a sustained error rise, not noise, and ranks the
community whose error rose; the runner accepts it for every arm."""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from gist.drift.error_signal import ErrorSignal


def test_flags_a_rise_and_not_noise():
    rng = np.random.default_rng(0)
    sig = ErrorSignal(window=30, reference=150, threshold=3.0)
    flags = []
    for t in range(600):
        p = 0.2 if t < 400 else 0.3
        flags.append(sig.update(rng.random(64) < p) > 3.0)
    assert not any(flags[:400])                 # stationary: no alarm
    assert any(flags[400:450])                  # +10 pp: alarm within the window


def test_ranks_the_community_whose_error_rose():
    rng = np.random.default_rng(1)
    sig = ErrorSignal(halflife=20, ref_halflife=300)
    for t in range(400):
        cids = rng.integers(0, 10, 64)
        p = np.where((cids == 7) & (t > 300), 0.7, 0.2)
        sig.update(rng.random(64) < p, cids)
    assert sig.drifting() == [7]


def test_runner_accepts_error_signal():
    from gist.config import load_config
    from gist.data.loaders import load_dataset
    from gist.experiments.runner import run_arm, warmup_model
    from gist.seeding import derive, seed_everything
    from gist.stream.schedule import build_plan
    over = {"stream.steps": 260, "runtime.device": "cpu", "model.warmup_epochs": 3, "drift.signal": "error",
            "drift.region_signal": "error", "drift.error_threshold": 1.0, "adaptation.cooldown": 20}
    cfg = load_config("configs/default.yaml", **over)
    seed_everything(100)
    ds = load_dataset(cfg.data.name, cfg.data.root, cfg.data.train_fraction, split_seed=derive(100, 0x51))
    plan = build_plan(ds.data.edge_index, ds.num_nodes, seed=100, steps=260, batch_size=cfg.stream.batch_size,
                      base_edge_fraction=cfg.data.base_edge_fraction, drift_cfg=cfg.drift_process,
                      x=ds.data.x, y=ds.data.y, num_classes=ds.out_dim)
    model, _ = warmup_model(cfg, ds, plan, seed=derive(100, 0x77))
    warm = {k: v.detach().clone() for k, v in model.state_dict().items()}
    for arm in ("full", "selective", "random_region"):
        r = run_arm(arm, cfg, ds, plan, warm)
        assert len(r.steps) == 260
        assert all(not s.retrained for s in r.steps[:180])     # no trigger before window + reference are full
    assert r.retrain_events, "the error trigger never fired"
