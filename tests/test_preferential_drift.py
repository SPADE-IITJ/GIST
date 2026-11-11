"""Preferential drift targeting (drift_process.degree_bias)."""
from __future__ import annotations

import numpy as np
import pytest
import torch


def _schedule(**extra):
    from gist.config import load_config
    from gist.data.loaders import load_dataset
    from gist.stream.drift_process import build_drift_schedule
    cfg = load_config("configs/default.yaml", **extra)
    ds = load_dataset(cfg.data.name, cfg.data.root, cfg.data.train_fraction, split_seed=1)
    ev = build_drift_schedule(x=ds.data.x, y=ds.data.y, full_edge_index=ds.data.edge_index, num_nodes=ds.num_nodes,
                              num_classes=ds.out_dim, steps=1000, seed=100, cfg=cfg.drift_process)
    deg = np.bincount(ds.data.edge_index[0].numpy(), minlength=ds.num_nodes)
    return ev, deg


@pytest.mark.parametrize("locality", ["localized", "diffuse"])
def test_degree_bias_hits_hubs_and_keeps_the_event_sequence(locality):
    uni, deg = _schedule(**{"drift_process.locality": locality})
    pa, _ = _schedule(**{"drift_process.locality": locality, "drift_process.degree_bias": 1.0})
    assert [(e.step, e.kind, e.magnitude, e.nodes.numel(), e.label_shift) for e in uni] == \
           [(e.step, e.kind, e.magnitude, e.nodes.numel(), e.label_shift) for e in pa]
    mean_deg = lambda evs: np.mean([deg[e.nodes.numpy()].mean() for e in evs])
    assert mean_deg(pa) > 1.2 * mean_deg(uni)
    for e in pa:
        assert e.nodes.unique().numel() == e.nodes.numel()
        if e.kind == "structural":
            assert torch.isin(e.edges[0], e.nodes).all()


def test_degree_bias_zero_is_the_default_schedule():
    a, _ = _schedule()
    b, _ = _schedule(**{"drift_process.degree_bias": 0.0})
    for x, y in zip(a, b):
        assert torch.equal(x.nodes, y.nodes)
        assert (x.edges is None and y.edges is None) or torch.equal(x.edges, y.edges)
