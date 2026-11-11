"""E3a: what one update costs under each method, as the drifted region and the graph grow (intro Gap 2).

    python -m experiments.e3a_cost --graph arxiv --model gat [--budget 2048] [--seeds 110 ...]

No stream: one update event per (method, region size r, graph fraction), on the graph with random features of the
stream model's width (cost does not depend on feature values). Per update: nodes and edges materialised, analytic
FLOPs (gist's counts), CUDA allocator peak and wall ms of forward+backward on the GPU.

| method | batch it materialises |
|---|---|
| full, ewc, twp | N uniform roots, fanout [25, 15] (EWC/TWP add an O(|theta|) penalty, no nodes) |
| ergnn | N uniform roots + a buffer of N/4 |
| selective | 0.75 N roots from the region, 0.25 N uniform anchors (GIST) |
| global (D3-GNN, InstantGNN) | the whole graph, full-batch |
| roland | the whole graph, `roland_inner` steps |

Region = random METIS communities (k = clamp(n/1000, 16, 256)) until they hold r % of the nodes. Graph fractions are
BFS balls from a random node (they keep local structure; random node subsampling would thin every neighbourhood).
"""
from __future__ import annotations

import argparse
import time

import numpy as np
import torch
import torch.nn.functional as F

from . import common as C
from .e1_census import csr, load_graph
from . import arms as A

R = C.R
DIMS = dict(arxiv=(128, 256, 40, 8), default=(128, 256, 40, 8))


def bfs_ball(a, n_target, rng):
    seen = np.zeros(a.shape[0], np.bool_)
    start = int(rng.integers(a.shape[0]))
    seen[start] = True
    frontier = seen.copy()
    while seen.sum() < n_target:
        nxt = ((a @ frontier.astype(np.int8)) > 0) & ~seen
        if not nxt.any():
            rest = np.nonzero(~seen)[0]
            nxt[int(rng.choice(rest))] = True
        if seen.sum() + nxt.sum() > n_target:
            idx = np.nonzero(nxt)[0]
            nxt[:] = False
            nxt[rng.choice(idx, n_target - int(seen.sum()), replace=False)] = True
        seen |= nxt
        frontier = nxt
    return np.nonzero(seen)[0]


def measure(model, name, x, ei, roots, n, dev, dims, inner=1):
    """Forward + backward on the sampled batch around `roots` (None: the whole graph)."""
    from torch_geometric.data import Data
    from torch_geometric.loader import NeighborLoader
    in_dim, hidden, out_dim, heads = dims
    if roots is None:
        bx, bei, bs = x, ei, n
    else:
        torch.manual_seed(0)
        loader = NeighborLoader(Data(x=x, edge_index=ei, num_nodes=n), num_neighbors=[25, 15],
                                input_nodes=torch.as_tensor(roots).long(), batch_size=len(roots), shuffle=False)
        b = next(iter(loader))
        bx, bei, bs = b.x, b.edge_index, b.batch_size
    nodes, edges = int(bx.shape[0]), int(bei.shape[1])
    bx, bei = bx.to(dev), bei.to(dev)
    y = torch.zeros(bs, dtype=torch.long, device=dev)
    torch.cuda.synchronize(dev)
    torch.cuda.reset_peak_memory_stats(dev)
    base = torch.cuda.memory_allocated(dev)
    flops = R.backbone_flops(name, nodes, edges, in_dim, hidden, out_dim, heads) * inner
    try:                                   # one untimed pass: kernel selection and allocator warm-up
        model.zero_grad()
        out, _, _ = model(bx, bei)
        F.cross_entropy(out[:bs], y).backward()
        torch.cuda.synchronize(dev)
        del out
        torch.cuda.reset_peak_memory_stats(dev)
        base = torch.cuda.memory_allocated(dev)
    except torch.OutOfMemoryError:
        model.zero_grad(set_to_none=True)
        torch.cuda.empty_cache()
        return dict(nodes=nodes * inner, edges=edges * inner, gflops=flops / 1e9, peak_mb=None, ms=None, oom=True)
    t = time.perf_counter()
    try:
        for _ in range(inner):
            model.zero_grad()
            out, _, _ = model(bx, bei)
            F.cross_entropy(out[:bs], y).backward()
        torch.cuda.synchronize(dev)
    except torch.OutOfMemoryError:     # does not fit on this GPU: a result, recorded as such
        model.zero_grad(set_to_none=True)
        out = None
        torch.cuda.empty_cache()
        return dict(nodes=nodes * inner, edges=edges * inner, gflops=flops / 1e9, peak_mb=None, ms=None, oom=True)
    ms = 1000 * (time.perf_counter() - t)
    peak = torch.cuda.max_memory_allocated(dev) - base
    del out
    torch.cuda.empty_cache()
    return dict(nodes=nodes * inner, edges=edges * inner, gflops=flops / 1e9, peak_mb=peak / 2**20, ms=ms, oom=False)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--graph", required=True)
    ap.add_argument("--model", required=True, choices=C.MODELS)
    ap.add_argument("--budget", type=int, default=2048)
    ap.add_argument("--regions", type=float, nargs="+", default=[0.5, 1, 2, 5, 10, 25])
    ap.add_argument("--fractions", type=float, nargs="+", default=[0.25, 0.5, 1.0])
    ap.add_argument("--seeds", type=int, nargs="+", default=list(C.REPORTED_SEEDS))
    a = ap.parse_args()
    C.guard(a.seeds)
    dev = torch.device("cuda")
    dims = DIMS.get(a.graph, DIMS["default"])
    ei_full, n_full, family = load_graph(a.graph)
    A_full = csr(ei_full, n_full)
    from gist.drift.partitioners import metis_partition
    from gist.models.registry import build_backbone
    from torch_geometric.utils import subgraph
    rows = []
    t0 = time.time()
    und = torch.from_numpy(np.vstack(A_full.nonzero())).long()
    for frac in a.fractions:
        g_rng = np.random.default_rng([a.seeds[0], 0xE3A, int(frac * 100)])
        keep = np.arange(n_full) if frac >= 1 else bfs_ball(A_full, int(frac * n_full), g_rng)
        keep_t = torch.from_numpy(np.sort(keep)).long()
        ei, _ = subgraph(keep_t, und, relabel_nodes=True, num_nodes=n_full)
        n = int(keep_t.numel())
        x = torch.randn(n, dims[0], generator=torch.Generator().manual_seed(0))
        k = int(np.clip(n // 1000, 16, 256))
        cid = metis_partition(ei, n, num_parts=k).numpy()
        sizes = np.bincount(cid, minlength=k)
        for seed in a.seeds:
            rng = np.random.default_rng([seed, 0xE3A])
            torch.manual_seed(seed)
            model = build_backbone(a.model, dims[0], dims[2], dims[1], dims[3]).to(dev)
            N = min(a.budget, n // 4)
            uni = rng.choice(n, N, replace=False)
            base = dict(seed=seed, fraction=frac, num_nodes=n, num_edges=int(ei.shape[1] // 2), budget=N)
            full = measure(model, a.model, x, ei, uni, n, dev, dims)
            buf = rng.choice(n, max(1, int(N * A.HP["ergnn_buffer_frac"])), replace=False)
            rows.append({**base, "method": "full", "region_pct": None, **full})
            rows.append({**base, "method": "ewc", "region_pct": None, **full,
                         "penalty_params": sum(p.numel() for p in model.parameters())})
            rows.append({**base, "method": "ergnn", "region_pct": None,
                         **measure(model, a.model, x, ei, np.unique(np.r_[uni, buf]), n, dev, dims)})
            g = measure(model, a.model, x, ei, None, n, dev, dims)
            rows.append({**base, "method": "global", "region_pct": None, **g})
            rows.append({**base, "method": "roland", "region_pct": None,
                         **measure(model, a.model, x, ei, None, n, dev, dims, inner=A.HP["roland_inner"])})
            sel = []
            for r in a.regions:
                order = rng.permutation(k)
                take = order[: max(1, int(np.searchsorted(np.cumsum(sizes[order]), r / 100 * n) + 1))]
                pool = np.nonzero(np.isin(cid, take))[0]
                n_t = int(round(0.75 * N))
                tgt = rng.choice(pool, min(n_t, pool.size), replace=False)
                anc = rng.choice(n, N - tgt.size, replace=False)
                m = measure(model, a.model, x, ei, np.unique(np.r_[tgt, anc]), n, dev, dims)
                rows.append({**base, "method": "selective", "region_pct": r, "region_nodes": int(pool.size), **m})
                sel.append(m["nodes"])
            print(f"  {a.graph}/{a.model} frac {frac} seed {seed}: n {n:,}  full {full['nodes']:,}  global {g['nodes']:,}  "
                  f"selective r={a.regions}: {sel}", flush=True)
            del model
            torch.cuda.empty_cache()
    C.write_json(C.RESULTS / "E3a" / f"{a.graph}_{a.model}.json",
                 dict(graph=a.graph, family=family, model=a.model, dims=dims, rows=rows, wall_s=round(time.time() - t0, 1)))
    print(f"wrote results/E3a/{a.graph}_{a.model}.json")


if __name__ == "__main__":
    main()
