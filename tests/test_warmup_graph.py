"""Warmup trains on the graph the stream starts from (``model.warmup_symmetric``).

``plan.base_edges`` stores each undirected pair once, lower id to higher. Warmup used it as is, so every node
aggregated only from lower-id neighbours, while the stream runs on GraphState's symmetric graph. Found from G2a's
Arxiv entry-accuracy check (+9.6 pp over the probe). The default keeps the old behaviour, so every result through
Phase C reproduces.
"""

from __future__ import annotations

import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from gist.config import load_config
from gist.data.loaders import load_dataset
from gist.experiments.runner import warmup_edges
from gist.seeding import derive, seed_everything
from gist.stream.graph_state import GraphState
from gist.stream.schedule import build_plan

SEED = 100


def _setup(**overrides):
    cfg = load_config("configs/default.yaml", **{"stream.steps": 30, **overrides})
    seed_everything(SEED)
    ds = load_dataset(cfg.data.name, cfg.data.root, cfg.data.train_fraction, split_seed=derive(SEED, 0x51))
    plan = build_plan(ds.data.edge_index, ds.num_nodes, seed=SEED, steps=30, batch_size=cfg.stream.batch_size,
                      base_edge_fraction=cfg.data.base_edge_fraction, drift_cfg=cfg.drift_process,
                      x=ds.data.x, y=ds.data.y, num_classes=ds.out_dim)
    return cfg, ds, plan


def _pairs(ei):
    return set(zip(ei[0].tolist(), ei[1].tolist()))


def test_base_edges_are_one_directional():
    # The premise: one direction per pair, lower id first.
    _, _, plan = _setup()
    assert bool((plan.base_edges[0] < plan.base_edges[1]).all())


def test_default_warmup_graph_is_unchanged():
    cfg, ds, plan = _setup()
    assert cfg.model.warmup_symmetric is False
    assert torch.equal(warmup_edges(cfg, ds.num_nodes, plan), plan.base_edges)


def test_symmetric_warmup_graph_is_the_streams_starting_graph():
    cfg, ds, plan = _setup(**{"model.warmup_symmetric": True})
    ei = warmup_edges(cfg, ds.num_nodes, plan)
    assert torch.equal(ei, GraphState(ds.num_nodes, plan.base_edges).edge_index)
    pairs = _pairs(ei)
    assert pairs == {(v, u) for u, v in pairs}                          # symmetric
    assert pairs == _pairs(plan.base_edges) | _pairs(plan.base_edges.flip(0))
    assert ei.shape[1] == 2 * plan.base_edges.shape[1]
