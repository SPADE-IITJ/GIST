"""G1: the switch that skips drift detection for arms that never read the signal.

Skipping detection must change nothing but cost for `baseline` and `periodic`, and must be refused for the arms
that fire on drift, where it would silently turn adaptation off.
"""

from __future__ import annotations

import math
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from gist.config import load_config
from gist.data.loaders import load_dataset
from gist.experiments.runner import run_arm, warmup_model
from gist.seeding import derive, seed_everything
from gist.stream.schedule import build_plan

CONFIG = "configs/default.yaml"
SEED = 100
STEPS = 150


def _setup(detection: bool):
    cfg = load_config(CONFIG, **{"stream.steps": STEPS, "runtime.device": "cpu",
                                 "adaptation.periodic_period": 40, "drift.detection": detection})
    seed_everything(SEED)
    dataset = load_dataset(cfg.data.name, cfg.data.root, cfg.data.train_fraction, split_seed=derive(SEED, 0x51))
    plan = build_plan(dataset.data.edge_index, dataset.num_nodes, seed=SEED, steps=STEPS,
                      batch_size=cfg.stream.batch_size, base_edge_fraction=cfg.data.base_edge_fraction,
                      drift_cfg=cfg.drift_process, x=dataset.data.x, y=dataset.data.y, num_classes=dataset.out_dim)
    model, _ = warmup_model(cfg, dataset, plan, seed=derive(SEED, 0x77))
    warm = {k: v.detach().clone() for k, v in model.state_dict().items()}
    return cfg, dataset, plan, warm


@pytest.fixture(scope="module")
def periodic_runs():
    return {d: run_arm("periodic", *_setup(d)) for d in (True, False)}


def test_detection_is_on_by_default():
    assert load_config(CONFIG).drift.detection is True


def test_periodic_without_detection_matches_with_detection(periodic_runs):
    on, off = periodic_runs[True], periodic_runs[False]
    assert [e.step for e in on.retrain_events] == [e.step for e in off.retrain_events] == [40, 80, 120]
    assert on.summary["accuracy"] == off.summary["accuracy"]
    assert [s.accuracy for s in on.steps] == [s.accuracy for s in off.steps]


def test_detection_off_skips_the_stage(periodic_runs):
    on, off = periodic_runs[True], periodic_runs[False]
    assert on.timing["detect"]["calls"] == STEPS
    assert "detect" not in off.timing and "detect" not in off.profile["latency"]
    assert all(math.isnan(s.drift["total"]) and s.drift["flag"] is False for s in off.steps)


def test_baseline_runs_without_detection():
    result = run_arm("baseline", *_setup(False))
    assert "detect" not in result.timing and not result.retrain_events


@pytest.mark.parametrize("arm", ["full", "selective"])
def test_detection_off_is_refused_for_drift_triggered_arms(arm):
    with pytest.raises(ValueError, match="fires on the drift signal"):
        run_arm(arm, *_setup(False))
