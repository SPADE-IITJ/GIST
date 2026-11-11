"""The G3/G5 "needs work" layers, made exact on the L-hop subgraph by gist/models/exact.py.

Same harness and threshold as tests/test_exactness_backbones.py (float64, relative gradient difference < 1e-8, one
Cora batch, targets = every 7th root). Each fix is checked three ways: exact with the fix at the stated hop count,
not exact without it (so the test can fail), and not exact one hop short (the negative control).

GCN and BatchNorm run in the main environment. EvolveGCN-O, T-GCN and GConvGRU need PyG Temporal and run in
env-backbones (``$B -m pytest tests/test_backbone_fixes.py``); elsewhere they are skipped.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from gist.adaptation.subgraph import subgraph_gradient_step
from gist.models.exact import (
    ParentNormGCN,
    evolvegcno_exact_kwargs,
    exact_gconvgru,
    exact_tgcn,
    frozen_norm_stats,
    parent_cheb_weights,
    parent_gcn_weights,
    weighted_subgraph,
)
from test_exactness_backbones import DTYPE, EXACT, HIDDEN, SAGE, SEED, _batch, _rel, _targets

FEAT = 16   # snapshot models: features projected by a fixed random matrix, per node, as in G5


@pytest.fixture(scope="module")
def cora():
    return _batch()


def _grads(model, run, y, pos):
    model.zero_grad()
    F.cross_entropy(run()[pos], y).backward()
    return {n: p.grad.detach().clone() for n, p in model.named_parameters() if p.grad is not None}


def _compare(model, forward, batch, ei, w, hops, x=None):
    """Relative gradient difference, full batch vs the targets' ``hops``-hop subgraph (weights sliced along)."""
    x = batch.x if x is None else x
    targets = _targets(batch)
    y = batch.y[targets]
    ref = _grads(model, lambda: forward(x, ei, w, None), y, targets)
    subset, sub_ei, mapping, sub_w = weighted_subgraph(targets, hops, ei, w, batch.num_nodes)
    return _rel(ref, _grads(model, lambda: forward(x[subset], sub_ei, sub_w, subset), y, mapping))


# --- GCN ------------------------------------------------------------------------------------------------------

def test_parent_norm_gcn_is_the_ordinary_gcn_on_the_full_batch(cora):
    batch, i, o = cora
    torch.manual_seed(SEED)
    plain = nn.ModuleList([GCNConv(i, HIDDEN), GCNConv(HIDDEN, o)]).to(DTYPE)
    fixed = ParentNormGCN(i, HIDDEN, o).to(DTYPE)
    fixed.convs.load_state_dict(plain.state_dict())
    ref = plain[1](torch.relu(plain[0](batch.x, batch.edge_index)), batch.edge_index)
    assert torch.allclose(fixed(batch.x, batch.edge_index), ref, rtol=0, atol=1e-12)


@pytest.mark.parametrize("hops,exact", [(1, False), (2, True), (3, True)])
def test_parent_norm_gcn_exact_at_depth(cora, hops, exact):
    batch, i, o = cora
    torch.manual_seed(SEED)
    model = ParentNormGCN(i, HIDDEN, o).to(DTYPE)
    ei, w = parent_gcn_weights(batch.edge_index, batch.num_nodes, dtype=DTYPE)
    d = _compare(model, lambda x, e, ww, s: model(x, e, ww), batch, ei, w, hops)
    assert (d < EXACT) == exact, f"{hops} hops: {d:.2e}"


def test_gcn_without_parent_weights_is_not_exact(cora):
    # Same model, but each graph normalises itself: the G3 failure (5.5e-2), so the fix is what makes it pass.
    batch, i, o = cora
    torch.manual_seed(SEED)
    model = ParentNormGCN(i, HIDDEN, o).to(DTYPE)
    d = _compare(model, lambda x, e, ww, s: model(x, e), batch, batch.edge_index, None, 2)
    assert d > 1e-2


def test_production_subgraph_step_uses_parent_weights(cora):
    """subgraph_gradient_step slices a batch's edge_weight with the subgraph: its gradient is the full-batch one."""
    batch, i, o = cora
    torch.manual_seed(SEED)
    model = ParentNormGCN(i, HIDDEN, o).to(DTYPE)
    weighted = batch.clone()
    weighted.edge_index, weighted.edge_weight = parent_gcn_weights(batch.edge_index, batch.num_nodes, dtype=DTYPE)
    targets = _targets(batch)
    mask = torch.zeros(batch.batch_size, dtype=torch.bool)
    mask[targets] = True
    ref = _grads(model, lambda: model(weighted.x, weighted.edge_index, weighted.edge_weight), batch.y[targets], targets)
    opt = torch.optim.SGD(model.parameters(), lr=0.0)
    out = subgraph_gradient_step(model, opt, weighted, mask, num_hops=2)
    assert out is not None
    sub = {n: p.grad.detach().clone() for n, p in model.named_parameters()}
    assert _rel(ref, sub) < EXACT
    assert out[1].nodes_in_backward < batch.num_nodes


# --- BatchNorm ------------------------------------------------------------------------------------------------

def test_batchnorm_with_frozen_stats_is_exact_and_leaves_state_alone(cora):
    batch, i, o = cora
    torch.manual_seed(SEED)
    model = SAGE(i, HIDDEN, o, norm=True).to(DTYPE).train()
    with torch.no_grad():                       # non-trivial running statistics
        model.bn.running_mean.normal_()
        model.bn.running_var.uniform_(0.5, 2.0)
    before = {k: v.clone() for k, v in model.bn.state_dict().items()}
    fwd = lambda x, e, ww, s: model(x, e)
    with frozen_norm_stats(model):
        exact2 = _compare(model, fwd, batch, batch.edge_index, None, 2)
        short1 = _compare(model, fwd, batch, batch.edge_index, None, 1)
    assert exact2 < EXACT and short1 > 1e-2
    assert model.training and model.bn.training                     # mode restored
    for k, v in model.bn.state_dict().items():
        assert torch.equal(v, before[k]), k                         # statistics untouched
    assert _compare(model, fwd, batch, batch.edge_index, None, 2) > 1e-2   # without it: the G3 failure


# --- PyG Temporal snapshot models (env-backbones) -------------------------------------------------------------

def _snapshot(cora):
    batch, _, o = cora
    g = torch.Generator().manual_seed(SEED)
    proj = torch.randn(batch.x.size(1), FEAT, generator=g, dtype=DTYPE) / batch.x.size(1) ** 0.5
    state = [torch.randn(batch.num_nodes, FEAT, generator=g, dtype=DTYPE) for _ in range(2)]
    return batch, o, batch.x @ proj, state


class TwoCells(nn.Module):
    """Two stacked recurrent graph cells and a linear head, as G5 probed them."""

    def __init__(self, cells, o, stateful):
        super().__init__()
        self.cells, self.head, self.stateful = nn.ModuleList(cells), nn.Linear(FEAT, o), stateful

    def forward(self, x, ei, w, state=None, **kw):
        for cell, h in zip(self.cells, state or [None, None]):
            if hasattr(cell, "weight") and not self.stateful:
                cell.weight = None          # EvolveGCN-O: every pass evolves from the same initial weight
            x = torch.relu(cell(x, ei, w, h, **kw) if self.stateful else cell(x, ei, w))
        return self.head(x)


def _run(model, batch, x, state, ei, w, hops, **kw):
    fwd = lambda xx, e, ww, s: model(xx, e, ww, None if state is None else [h if s is None else h[s] for h in state], **kw)
    return _compare(model, fwd, batch, ei, w, hops, x=x)


@pytest.mark.parametrize("fixed,hops,exact", [(True, 1, False), (True, 2, True), (True, 3, True), (False, 2, False)])
def test_evolvegcno_exact_with_parent_weights(cora, fixed, hops, exact):
    rec = pytest.importorskip("torch_geometric_temporal.nn.recurrent")
    batch, o, x, _ = _snapshot(cora)
    torch.manual_seed(SEED)
    kw = evolvegcno_exact_kwargs() if fixed else {}
    model = TwoCells([rec.EvolveGCNO(FEAT, **kw), rec.EvolveGCNO(FEAT, **kw)], o, stateful=False).to(DTYPE)
    ei, w = parent_gcn_weights(batch.edge_index, batch.num_nodes, dtype=DTYPE) if fixed else (batch.edge_index, None)
    d = _run(model, batch, x, None, ei, w, hops)
    assert (d < EXACT) == exact, f"fixed={fixed} {hops} hops: {d:.2e}"


@pytest.mark.parametrize("fixed,hops,exact", [(True, 1, False), (True, 2, True), (True, 3, True), (False, 2, False)])
def test_tgcn_exact_with_parent_weights(cora, fixed, hops, exact):
    rec = pytest.importorskip("torch_geometric_temporal.nn.recurrent")
    batch, o, x, state = _snapshot(cora)
    torch.manual_seed(SEED)
    model = TwoCells([rec.TGCN(FEAT, FEAT), rec.TGCN(FEAT, FEAT)], o, stateful=True).to(DTYPE)
    if fixed:
        exact_tgcn(model)
        assert all(not c.normalize for cell in model.cells for c in (cell.conv_z, cell.conv_r, cell.conv_h))
    ei, w = parent_gcn_weights(batch.edge_index, batch.num_nodes, dtype=DTYPE) if fixed else (batch.edge_index, None)
    d = _run(model, batch, x, state, ei, w, hops)
    assert (d < EXACT) == exact, f"fixed={fixed} {hops} hops: {d:.2e}"


@pytest.mark.parametrize("normalization", ["sym", None])
@pytest.mark.parametrize("fixed,hops,exact", [(True, 3, False), (True, 4, True), (True, 5, True), (False, 4, False), (False, 6, False)])
def test_gconvgru_exact_on_the_sampled_batch_with_a_parent_laplacian(cora, normalization, fixed, hops, exact):
    """G5: never exact on a sampled batch at any hop count. With the parent's Laplacian: exact at 2 hops per cell."""
    rec = pytest.importorskip("torch_geometric_temporal.nn.recurrent")
    batch, o, x, state = _snapshot(cora)
    torch.manual_seed(SEED)
    model = TwoCells([rec.GConvGRU(FEAT, FEAT, K=2, normalization=normalization) for _ in range(2)], o,
                     stateful=True).to(DTYPE)
    # A fixed lambda_max for the combinatorial Laplacian, from the parent: twice its largest weighted degree.
    lam = None if normalization == "sym" else float(2 * torch.bincount(batch.edge_index[0], minlength=batch.num_nodes).max())
    kw = {} if lam is None else dict(lambda_max=torch.tensor(lam, dtype=DTYPE))
    if fixed:
        exact_gconvgru(model)
        ei, w = parent_cheb_weights(model, batch.edge_index, batch.num_nodes, lambda_max=lam, dtype=DTYPE)
        kw = {}
    else:
        ei, w = batch.edge_index, None
    d = _run(model, batch, x, state, ei, w, hops, **kw)
    assert (d < EXACT) == exact, f"{normalization} fixed={fixed} {hops} hops: {d:.2e}"


def test_parent_cheb_weights_needs_lambda_max_when_not_symmetric():
    rec = pytest.importorskip("torch_geometric_temporal.nn.recurrent")
    cell = rec.GConvGRU(FEAT, FEAT, K=2, normalization=None)
    with pytest.raises(ValueError):
        parent_cheb_weights(cell, torch.tensor([[0, 1], [1, 0]]), 2)
