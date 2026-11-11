"""Shared plumbing for experiments: paths, seeds, stream variants, calibration, warm states, and the arm registry.

Everything streams through the prelim campaign's driver (`prelim/scripts/prelim.py`), imported read-only, so a
`full` or `selective` arm here is the same code and the same hyperparameters as the paper's prelim numbers. The new
arms (arms.py) plug in by wrapping `gist.experiments.runner.build_retrainer` at runtime; no file outside
experiments/ changes.
"""
from __future__ import annotations

import copy
import json
import os
import sys
from pathlib import Path

import numpy as np
import torch

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "prelim" / "scripts"))

import prelim as P  # noqa: E402
from gist.experiments import runner as R  # noqa: E402
from gist.seeding import derive  # noqa: E402

# GIST_SMOKE=1 sends every output to results/smoke/, so smoke runs never mix with the real results.
# GIST_RESULTS=<dir> (set by experiments.final for Phase D) sends them to that directory instead.
RESULTS = (Path(os.environ["GIST_RESULTS"]) if os.environ.get("GIST_RESULTS")
           else HERE / "results" / "smoke" if os.environ.get("GIST_SMOKE") else HERE / "results")
CAL = HERE / "calibration"
CACHE = HERE / "cache"
ARMS_HP = (RESULTS if os.environ.get("GIST_SMOKE") else CAL) / "arms_hp.json"  # baselines' tuned hyperparameters
CAL_SEEDS = (100, 101, 102)
REPORTED_SEEDS = (110, 111, 112, 113, 114)
MODELS = ("gat", "sage", "gcn", "gin")

# ogbn-products: a scale stream with injected drift, Arxiv's model and stream settings.
P.DATASETS.setdefault("products", dict(config="configs/arxiv.yaml", name="ogbn-products", temporal=False, steps=1000,
                                       budget=8192, epochs=[100, 200, 400, 800]))
PRODUCTS_PT = ROOT / "data" / "ogbn_products_pyg" / "data.pt"   # converted from DGL's GraphBolt copy
WARMUP_ROOTS_BIG = 65536            # sampled training roots per warmup epoch on graphs over 1M nodes


def _install_products_loader():
    from gist.data import loaders as Lm
    if getattr(Lm, "_introval_products", False):
        return
    orig = Lm.load_ogb

    def load_ogb(name, root):
        if name == "ogbn-products" and PRODUCTS_PT.exists():
            d = torch.load(PRODUCTS_PT, weights_only=False)
            return d["data"], int(d["in_dim"]), int(d["out_dim"])
        return orig(name, root)
    Lm.load_ogb = load_ogb
    Lm._introval_products = True


def _install_big_warmup():
    """Warmup on a graph over 1M nodes: each epoch is one step on WARMUP_ROOTS_BIG sampled training roots, not on all of
    them (a full epoch over ogbn-products' training roots does not fit on one GPU). Smaller graphs are unchanged."""
    if getattr(R, "_introval_big_warmup", False):
        return
    orig = R.warmup_model

    def warmup_model(cfg, dataset, plan, seed, on_epoch=None):
        if dataset.num_nodes <= 1_000_000:
            return orig(cfg, dataset, plan, seed, on_epoch=on_epoch)
        torch.manual_seed(seed)
        device = R.resolve_device(cfg.runtime.device)
        model = R.build_backbone(cfg.model.backbone, dataset.in_dim, dataset.out_dim, cfg.model.hidden, cfg.model.heads)
        model.to(device)
        opt = torch.optim.Adam(model.parameters(), lr=cfg.model.lr, weight_decay=cfg.model.weight_decay)
        data = copy.copy(dataset.data)
        data.edge_index = R.warmup_edges(cfg, dataset.num_nodes, plan)
        data.n_id = torch.arange(data.num_nodes)
        pool = R._train_pool(dataset.train_mask, plan, -1).nonzero(as_tuple=True)[0]
        sampler = R.CachedSampler(data, cfg.stream.num_neighbors, refresh_interval=10 ** 9)
        g = torch.Generator().manual_seed(seed)
        losses = []
        for epoch in range(cfg.model.warmup_epochs):
            roots = pool[torch.randperm(pool.numel(), generator=g)[:WARMUP_ROOTS_BIG]]
            batch = sampler.sample(roots, plan.sample_seeds[0] + epoch, 0).to(device)
            model.train(); opt.zero_grad()
            out, _, _ = model(batch.x, batch.edge_index)
            loss = torch.nn.functional.cross_entropy(out[: batch.batch_size], batch.y[: batch.batch_size])
            loss.backward(); opt.step()
            losses.append(float(loss.item()))
            if on_epoch is not None:
                on_epoch(epoch + 1, model)
        model.to("cpu")
        return model, losses
    R.warmup_model = warmup_model
    P.warmup_model = warmup_model
    R._introval_big_warmup = True


_install_products_loader()
_install_big_warmup()


class FastGraphState:
    """GraphState for graphs over 1M nodes: the same edge set, in the same order (symmetric, de-duplicated, self-loops
    dropped, first occurrence first), but the base snapshot is built with numpy instead of a Python set of tuples
    (ogbn-products: ~31M base pairs). Later per-step additions are small and use binary search + a set."""

    def __init__(self, num_nodes, base_edges=None):
        self.num_nodes = int(num_nodes)
        n = self.num_nodes
        if base_edges is None or base_edges.numel() == 0:
            self._cache = torch.empty((2, 0), dtype=torch.long)
            self._base = np.empty(0, dtype=np.int64)
        else:
            u, v = base_edges[0].numpy().astype(np.int64), base_edges[1].numpy().astype(np.int64)
            keep = u != v
            u, v = u[keep], v[keep]
            a = np.empty(2 * u.size, dtype=np.int64); b = np.empty_like(a)
            a[0::2], b[0::2], a[1::2], b[1::2] = u, v, v, u          # (u,v), (v,u) in arrival order
            key = a * n + b
            _, first = np.unique(key, return_index=True)
            first.sort()
            self._cache = torch.from_numpy(np.stack([a[first], b[first]]))
            self._base = np.sort(key[first])
        self._extra = set()

    def _known(self, k):
        i = np.searchsorted(self._base, k)
        return (i < self._base.size and self._base[i] == k) or k in self._extra

    def add_edges(self, edges):
        if edges.numel() == 0:
            return 0
        n, fresh = self.num_nodes, []
        for u, v in zip(edges[0].tolist(), edges[1].tolist()):
            if u == v:
                continue
            for a, b in ((u, v), (v, u)):
                k = a * n + b
                if not self._known(k):
                    self._extra.add(k)
                    fresh.append((a, b))
        if fresh:
            self._cache = torch.cat([self._cache, torch.tensor(fresh, dtype=torch.long).t().contiguous()], dim=1)
        return len(fresh)

    def add_edge(self, src, dst):
        return self.add_edges(torch.tensor([[src], [dst]], dtype=torch.long))

    @property
    def edge_index(self):
        return self._cache

    @property
    def num_directed_edges(self):
        return int(self._cache.shape[1])

    def __len__(self):
        return self.num_directed_edges


def _install_fast_graph_state():
    from gist.stream import graph_state as GS
    if getattr(GS, "_introval_fast", False):
        return
    Slow = GS.GraphState

    def GraphState(num_nodes, base_edges=None):
        return (FastGraphState if int(num_nodes) > 1_000_000 else Slow)(num_nodes, base_edges)
    GS.GraphState = GraphState
    R.GraphState = GraphState
    GS._introval_fast = True


_install_fast_graph_state()

# Stream variants: a prelim dataset plus config overrides. `label` streams carry pure concept drift: a cyclic class
# shift on a community's nodes, with the graph and the features untouched (drift kind "label" in drift_process.py).
VARIANTS = {
    **{d: dict(base=d, over={}) for d in P.DATASETS},
    "cora_label": dict(base="cora", over={"drift_process.kinds": ["label"]}),
    "pubmed_label": dict(base="pubmed", over={"drift_process.kinds": ["label"]}),
    "arxiv_label": dict(base="arxiv", over={"drift_process.kinds": ["label"]}),
    "cora_struct": dict(base="cora", over={"drift_process.kinds": ["structural"]}),
    "arxiv_struct": dict(base="arxiv", over={"drift_process.kinds": ["structural"]}),
}


def eval_seeds_allowed() -> bool:
    """Phase D only: seeds 0-9 run when experiments.final set GIST_EVAL=1."""
    return os.environ.get("GIST_EVAL") == "1"


def guard(seeds, allowed=None):
    """Seeds 0-9 run only in Phase D (eval_seeds_allowed); calibration jobs run on 100-102 only."""
    bad = sorted(set(seeds) & set(range(10)))
    if bad and not eval_seeds_allowed():
        raise SystemExit(f"seeds {bad} are the paper's evaluation seeds (0-9); only "
                         f"python -m experiments.final ... runs them")
    if allowed is not None and not set(seeds) <= set(allowed):
        raise SystemExit(f"this job uses seeds {sorted(allowed)} only, got {sorted(seeds)}")


def base_of(variant: str) -> str:
    return VARIANTS[variant]["base"]


def config(model: str, variant: str, **extra):
    v = VARIANTS[variant]
    return P.config(model, v["base"], **{**v["over"], **extra})


def stream(cfg, variant: str, seed: int):
    return P.stream(cfg, base_of(variant), seed)


def calibration(model: str, variant: str) -> dict:
    """Warmup epochs and drift threshold: experiments's own file if present, else the prelim campaign's (seeds 100-102).

    A label / structural variant shares its base dataset's warmup (the base snapshot G0 does not depend on the drift
    kind) but gets its own threshold, calibrated on its own null stream (calibrate.py)."""
    own = CAL / f"{model}_{variant}.json"
    cal = json.loads(own.read_text()) if own.exists() else {}
    pre = P.cal_path(model, base_of(variant))
    pre = json.loads(pre.read_text()) if pre.exists() else {}
    out = {**({k: pre[k] for k in ("warmup_epochs", "probe") if k in pre}),
           **({"total_threshold": pre["total_threshold"]} if variant == base_of(variant) and "total_threshold" in pre else {}),
           **cal}
    out["source"] = "experiments" if own.exists() else "prelim"
    return out


def save_calibration(model: str, variant: str, **fields):
    p = CAL / f"{model}_{variant}.json"
    cal = json.loads(p.read_text()) if p.exists() else {}
    cal.update(fields)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(cal, indent=2, sort_keys=True))


def run_config(model: str, variant: str, **extra):
    """The prelim `run` configuration: calibrated warmup and threshold, synced stage timers, retrain memory."""
    cal = calibration(model, variant)
    if "drift.total_threshold_override" in extra:  # review round: a threshold at another null flag rate
        cal["total_threshold"] = float(extra.pop("drift.total_threshold_override"))
    if "drift.total_threshold" in extra:          # smoke cells pass a threshold instead of a calibration
        cal.setdefault("total_threshold", extra.pop("drift.total_threshold"))
    if "model.warmup_epochs" in extra:            # ... and may pass a warmup length (E5's warmup sweep does too)
        cal["warmup_epochs"] = extra.pop("model.warmup_epochs")
    if "warmup_epochs" not in cal or "total_threshold" not in cal:
        raise SystemExit(f"{model}/{variant} is not calibrated (run calibrate.py on seeds 100-102)")
    return config(model, variant, **{"model.warmup_epochs": cal["warmup_epochs"],
                                     "drift.total_threshold": cal["total_threshold"],
                                     "runtime.sync_stages": True, "runtime.measure_retrain_memory": True, **extra})


def warm_state(cfg, ds, plan, seed: int, key: str):
    """Warm weights for (config, seed), cached on disk: the warmup is identical across arms and across experiments
    that share model, base dataset, warmup settings and seed."""
    tag = (f"{cfg.model.backbone}_{cfg.data.name}_e{cfg.model.warmup_epochs}_lr{cfg.model.lr}"
           f"_sym{int(cfg.model.warmup_symmetric)}_h{cfg.model.hidden}_s{seed}_{key}").replace("/", "-")
    p = CACHE / "warm" / f"{tag}.pt"
    if p.exists():
        return torch.load(p, map_location="cpu")
    model, _ = R.warmup_model(cfg, ds, plan, seed=derive(seed, 0x77))
    st = {k: v.detach().clone() for k, v in model.state_dict().items()}
    p.parent.mkdir(parents=True, exist_ok=True)
    torch.save(st, p)
    return st


def install_arms():
    """Route experiments's arm names through runner.run_arm (idempotent)."""
    from . import arms  # noqa: F401  (registers on import)
    return arms


def write_json(path: Path, obj) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(obj))
    os.replace(tmp, path)


def read_json(path: Path, default=None):
    return json.loads(path.read_text()) if path.exists() else default
