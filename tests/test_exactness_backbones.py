"""G3: for which layer types is the subgraph gradient equal to the full-batch gradient?

Extends tests/test_phase2_subgraph.py from the repository's GAT to the four PyG layer types the layer is meant to
wrap, plus two variants: GCN with its normalisation computed on the parent batch, and a model with BatchNorm
between layers. Everything runs in float64, so "exact" means a relative gradient difference below 1e-8, not a
float32 tolerance.

On one sampled Cora batch, the gradient of the loss masked to a strict subset of the roots is computed (a) over
the whole batch and (b) over the targets' induced k-hop subgraph. k = 2 matches the 2-layer models; k = 1 is the
negative control (it must differ for every model), k = 3 the superset control (it must stay exact wherever 2 hops
is exact).

Predictions, written before measuring: GAT, SAGE and GIN exact; GCN with default
normalisation not exact; GCN with parent-graph weights exact; BatchNorm not exact. All five held. Also measured:
GCN with default normalisation becomes exact at 3 hops (depth + 1), and BatchNorm is exact in eval mode.

    python tests/test_exactness_backbones.py      # writes results/g3_exactness.json
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GATConv, GCNConv, GINConv, SAGEConv
from torch_geometric.nn.conv.gcn_conv import gcn_norm
from torch_geometric.utils import k_hop_subgraph

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from gist.config import load_config
from gist.data.loaders import load_dataset
from gist.experiments.runner import sample_batch
from gist.models.gat import build_model
from gist.seeding import derive, seed_everything
from gist.stream.graph_state import GraphState
from gist.stream.schedule import build_plan

CONFIG = "configs/default.yaml"
SEED = 0
HIDDEN = 32
EXACT = 1e-8
DTYPE = torch.float64


class GAT(nn.Module):
    def __init__(self, i, h, o):
        super().__init__()
        self.c1, self.c2 = GATConv(i, h, heads=4), GATConv(4 * h, o)

    def forward(self, x, ei, w=None):
        return self.c2(torch.relu(self.c1(x, ei)), ei)


class SAGE(nn.Module):
    def __init__(self, i, h, o, norm=False):
        super().__init__()
        self.c1, self.c2 = SAGEConv(i, h, aggr="mean"), SAGEConv(h, o, aggr="mean")
        self.bn = nn.BatchNorm1d(h) if norm else nn.Identity()

    def forward(self, x, ei, w=None):
        return self.c2(torch.relu(self.bn(self.c1(x, ei))), ei)


class GCN(nn.Module):
    """``parent_norm``: normalisation computed once on the parent batch and passed as edge weights."""

    def __init__(self, i, h, o, parent_norm=False):
        super().__init__()
        self.parent_norm = parent_norm
        kw = dict(normalize=False) if parent_norm else {}
        self.c1, self.c2 = GCNConv(i, h, **kw), GCNConv(h, o, **kw)

    def forward(self, x, ei, w=None):
        return self.c2(torch.relu(self.c1(x, ei, w)), ei, w)


class GIN(nn.Module):
    def __init__(self, i, h, o):
        super().__init__()
        mlp = lambda a, b: nn.Sequential(nn.Linear(a, h), nn.ReLU(), nn.Linear(h, b))
        self.c1, self.c2 = GINConv(mlp(i, h)), GINConv(mlp(h, o))

    def forward(self, x, ei, w=None):
        return self.c2(torch.relu(self.c1(x, ei)), ei)


class RepoGAT(nn.Module):
    """gist.models.gat.StreamGAT, the model tests/test_phase2_subgraph.py checks."""

    def __init__(self, i, h, o):
        super().__init__()
        self.m = build_model(i, o, h, 4)

    def forward(self, x, ei, w=None):
        return self.m(x, ei)[0]


# name -> (constructor, train mode). BatchNorm is measured in both modes: in training it normalises with
# statistics over whatever nodes are present; in eval it uses fixed running statistics.
MODELS = {
    "GAT": (lambda i, o: GAT(i, HIDDEN, o), True),
    "SAGE (mean)": (lambda i, o: SAGE(i, HIDDEN, o), True),
    "GCN (default normalisation)": (lambda i, o: GCN(i, HIDDEN, o), True),
    "GCN (parent-batch normalisation)": (lambda i, o: GCN(i, HIDDEN, o, parent_norm=True), True),
    "GIN": (lambda i, o: GIN(i, HIDDEN, o), True),
    "SAGE + BatchNorm (train mode)": (lambda i, o: SAGE(i, HIDDEN, o, norm=True), True),
    "SAGE + BatchNorm (eval mode)": (lambda i, o: SAGE(i, HIDDEN, o, norm=True), False),
    "GAT (repository StreamGAT)": (lambda i, o: RepoGAT(i, HIDDEN, o), True),
}
PREDICTED_EXACT = {"GAT": True, "SAGE (mean)": True, "GIN": True, "GCN (default normalisation)": False,
                   "GCN (parent-batch normalisation)": True, "SAGE + BatchNorm (train mode)": False}


def _batch():
    cfg = load_config(CONFIG, **{"stream.steps": 60})
    seed_everything(SEED)
    dataset = load_dataset(cfg.data.name, cfg.data.root, cfg.data.train_fraction, split_seed=derive(SEED, 0x51))
    plan = build_plan(dataset.data.edge_index, dataset.num_nodes, seed=SEED, steps=60,
                      batch_size=cfg.stream.batch_size, base_edge_fraction=cfg.data.base_edge_fraction,
                      drift_cfg=cfg.drift_process, x=dataset.data.x, y=dataset.data.y, num_classes=dataset.out_dim)
    data = dataset.data
    data.edge_index = GraphState(dataset.num_nodes, plan.base_edges).edge_index
    batch = sample_batch(data, plan.root_nodes[0], cfg.stream.num_neighbors, plan.sample_seeds[0])
    batch.x = batch.x.to(DTYPE)
    return batch, dataset.in_dim, dataset.out_dim


def _targets(batch, every: int = 7):
    return torch.arange(0, batch.batch_size, every)


def _grad(model, x, ei, w, y, pos):
    model.zero_grad()
    F.cross_entropy(model(x, ei, w)[pos], y).backward()
    return {n: p.grad.detach().clone() for n, p in model.named_parameters() if p.grad is not None}


def _rel(ref, sub):
    """Largest per-tensor relative difference, max|delta| / max|reference|."""
    assert set(ref) == set(sub)
    return max(float((ref[n] - sub[n]).abs().max() / ref[n].abs().max().clamp_min(1e-300)) for n in ref)


def measure(name: str, batch=None) -> dict:
    batch, i, o = batch or _batch()
    make, train = MODELS[name]
    torch.manual_seed(SEED)
    model = make(i, o).to(DTYPE).train(train)
    targets = _targets(batch)
    y = batch.y[targets]
    ei, w = batch.edge_index, None
    if getattr(model, "parent_norm", False):
        ei, w = gcn_norm(batch.edge_index, None, batch.num_nodes, add_self_loops=True, dtype=DTYPE)
    ref = _grad(model, batch.x, ei, w, y, targets)
    row = dict(model=name, train_mode=train, dtype=str(DTYPE).replace("torch.", ""), batch_nodes=batch.num_nodes,
               batch_roots=batch.batch_size, targets=len(targets))
    for k in (1, 2, 3):
        subset, sub_ei, mapping, edge_mask = k_hop_subgraph(targets, k, ei, relabel_nodes=True, num_nodes=batch.num_nodes)
        sub_w = None if w is None else w[edge_mask]
        row[f"rel_diff_{k}hop"] = _rel(ref, _grad(model, batch.x[subset], sub_ei, sub_w, y, mapping))
        row[f"subgraph_nodes_{k}hop"] = int(subset.numel())
    row["exact_2hop"] = row["rel_diff_2hop"] < EXACT
    row["exact_3hop"] = row["rel_diff_3hop"] < EXACT
    return row


def measure_all() -> list[dict]:
    b = _batch()
    return [measure(n, b) for n in MODELS]


# --- assertions: what was measured (2026-10-15), not what was hoped ----------------------------------------------

# Exact at 2 hops (= model depth), as measured. All five predictions held.
MEASURED_EXACT = {**PREDICTED_EXACT, "SAGE + BatchNorm (eval mode)": True, "GAT (repository StreamGAT)": True}


@pytest.fixture(scope="module")
def rows():
    return {r["model"]: r for r in measure_all()}


@pytest.mark.parametrize("name", list(MODELS))
def test_exactness_at_model_depth(rows, name):
    r = rows[name]
    assert r["exact_2hop"] == MEASURED_EXACT[name], f"{name}: 2-hop relative difference {r['rel_diff_2hop']:.2e}"


@pytest.mark.parametrize("name", list(MODELS))
def test_one_hop_is_not_enough_for_any_model(rows, name):
    # Negative control. Measured minimum 0.29; if this ever passes at 1 hop, the targets or the graph are too easy.
    assert rows[name]["rel_diff_1hop"] > 1e-2


@pytest.mark.parametrize("name", [n for n, e in MEASURED_EXACT.items() if e])
def test_extra_nodes_do_not_change_an_exact_gradient(rows, name):
    # Superset control: a 3-hop subgraph adds nodes the targets do not depend on.
    assert rows[name]["exact_3hop"], f"{name}: 3-hop relative difference {rows[name]['rel_diff_3hop']:.2e}"


def test_gcn_default_normalisation_needs_one_extra_hop(rows):
    # Measured, not predicted: GCN's symmetric normalisation reads the degree of the 2-hop nodes, which is only
    # complete once their own neighbours (3 hops) are in the subgraph. Exact at depth + 1.
    r = rows["GCN (default normalisation)"]
    assert not r["exact_2hop"] and r["exact_3hop"]


def test_batchnorm_in_training_is_not_fixed_by_more_hops(rows):
    # Batch statistics depend on every node present, so no subgraph of the batch reproduces them.
    assert rows["SAGE + BatchNorm (train mode)"]["rel_diff_3hop"] > 1e-2


def test_repository_gat_agrees_with_phase2(rows):
    # Anchor: tests/test_phase2_subgraph.py asserts the same model is exact at 2 hops in float32.
    assert rows["GAT (repository StreamGAT)"]["exact_2hop"]


def test_targets_and_subgraph_are_non_trivial(rows):
    for r in rows.values():
        assert r["targets"] < r["batch_roots"]
        assert r["subgraph_nodes_2hop"] < r["batch_nodes"]


def test_comparisons_ran_in_float64(rows):
    assert all(r["dtype"] == "float64" for r in rows.values())


if __name__ == "__main__":
    rows = measure_all()
    for r in rows:
        print(f"{r['model']:<34} 2-hop {r['rel_diff_2hop']:.2e}  1-hop {r['rel_diff_1hop']:.2e}  3-hop {r['rel_diff_3hop']:.2e}"
              f"  exact@2 {r['exact_2hop']}  nodes {r['subgraph_nodes_1hop']}/{r['subgraph_nodes_2hop']}/{r['subgraph_nodes_3hop']} of {r['batch_nodes']}")
    out = Path(__file__).resolve().parents[1] / "results" / "g3_exactness.json"
    out.write_text(json.dumps(dict(dtype="float64", threshold=EXACT, dataset="Cora (configs/default.yaml), seed 0, first stream batch",
                                   predicted_exact=PREDICTED_EXACT, rows=rows), indent=2))
    print(f"wrote {out}")
