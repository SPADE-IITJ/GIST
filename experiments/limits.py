"""D7: the limits of an update on one 48 GB L40S, and 3-layer models.

    python -m experiments.limits memory --variant arxiv|products --model gat|sage --seeds 0 ... 4
    python -m experiments.limits deep --variant cora|pubmed|arxiv --model gat|sage --seeds 0 ... 9

memory: one update on the stream's graph (after the base snapshot) with the dataset's real features, for
  hidden in {256, 512, 1024} x roots N in {2048, 8192, 32768}:
  - full (N uniform roots, the config's fanout);
  - selective (0.75 N roots from METIS communities + 0.25 N uniform anchors, same fanout);
  - global (the whole graph).
  Peak CUDA memory above resident, forward + backward ms, or OOM. Finds where each method stops fitting.
deep: 3-layer GAT / GraphSAGE (receptive field 3).
  - float64 gradient on the whole graph vs the targets' 3-hop and 2-hop induced subgraphs (exact at 3, one hop short at 2);
  - one GPU update's nodes / memory / ms, full vs selective (fanout [15, 10, 5]).
"""
from __future__ import annotations

import argparse
import time

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GATConv, SAGEConv
from torch_geometric.utils import k_hop_subgraph

from . import common as C

R = C.R


def graph_after_base(variant, seed):
    cfg = C.config("gat", variant, **{"model.warmup_epochs": 1})
    ds, plan = C.stream(cfg, variant, seed)
    from gist.stream.graph_state import GraphState
    ei = GraphState(ds.num_nodes, plan.base_edges).edge_index
    return cfg, ds, plan, ei


def region_roots(ei, n, N, rng, train):
    from gist.drift.partitioners import metis_partition
    k = int(np.clip(n // 1000, 16, 256))
    cid = metis_partition(ei, n, num_parts=k).numpy()
    order = rng.permutation(k)
    sizes = np.bincount(cid, minlength=k)
    take = order[: max(1, int(np.searchsorted(np.cumsum(sizes[order]), 0.75 * N * 2) + 1))]
    pool = np.nonzero(np.isin(cid, take) & train)[0]
    tgt = rng.choice(pool, min(int(0.75 * N), pool.size), replace=False)
    tr = np.nonzero(train)[0]
    anc = rng.choice(tr, N - tgt.size, replace=False)
    return np.unique(np.r_[tgt, anc])


def step_cost(model, bx, bei, bs, y, dev):
    """Forward + backward of one update: peak MB above resident, ms; None on out-of-memory."""
    torch.cuda.synchronize(dev); torch.cuda.empty_cache()
    base = torch.cuda.memory_allocated(dev); torch.cuda.reset_peak_memory_stats(dev)
    try:
        bx, bei, y = bx.to(dev), bei.to(dev), y.to(dev)
        t = time.perf_counter()
        model.zero_grad()
        out = model(bx, bei)[0]
        F.cross_entropy(out[:bs], y[:bs]).backward()
        torch.cuda.synchronize(dev)
        ms = 1000 * (time.perf_counter() - t)
        peak = (torch.cuda.max_memory_allocated(dev) - base) / 2 ** 20
        return dict(peak_mb=peak, ms=ms, oom=False)
    except torch.OutOfMemoryError:
        model.zero_grad(set_to_none=True)
        torch.cuda.empty_cache()
        return dict(peak_mb=None, ms=None, oom=True)


def memory(a):
    dev = torch.device("cuda")
    out_p = C.RESULTS / "D7" / f"memory_{a.variant}_{a.model}.json"
    res = C.read_json(out_p, dict(variant=a.variant, model=a.model, rows=[]))
    done = {(r["seed"], r["hidden"], r["N"], r["method"]) for r in res["rows"]}
    for seed in a.seeds:
        cfg, ds, plan, ei = graph_after_base(a.variant, seed)
        n, x, y = ds.num_nodes, ds.data.x, ds.data.y
        data = ds.data.clone(); data.edge_index = ei; data.n_id = torch.arange(n)
        sampler = R.CachedSampler(data, cfg.stream.num_neighbors, refresh_interval=10 ** 9)
        rng = np.random.default_rng([seed, 0xD7])
        train = ds.train_mask.numpy()
        for hidden in (256, 512, 1024):
            heads = cfg.model.heads
            torch.manual_seed(seed)
            model = R.build_backbone(a.model, ds.in_dim, ds.out_dim, hidden, heads).to(dev)
            for N in (2048, 8192, 32768):
                for method in ("full", "selective", "global"):
                    if (seed, hidden, N, method) in done or (method == "global" and N != 2048):
                        continue
                    if method == "global":
                        bx, bei, bs, nodes = x, ei, n, n
                    else:
                        roots = (rng.choice(np.nonzero(train)[0], N, replace=False) if method == "full"
                                 else region_roots(ei, n, N, rng, train))
                        b = sampler.sample(torch.from_numpy(roots).long(), seed, 0)
                        bx, bei, bs, nodes = b.x, b.edge_index, b.batch_size, int(b.num_nodes)
                    r = step_cost(model, bx, bei, bs, y if method == "global" else b.y, dev)
                    res["rows"].append(dict(seed=seed, hidden=hidden, N=N, method=method, nodes=nodes,
                                            edges=int(bei.shape[1]), **r))
                    C.write_json(out_p, res)
                    print(f"  D7 memory {a.variant}/{a.model} seed {seed} h{hidden} N{N} {method:<9} nodes {nodes:>9,} "
                          + ("OOM" if r["oom"] else f"{r['peak_mb']:>8.0f} MB {r['ms']:>7.1f} ms"), flush=True)
            del model; torch.cuda.empty_cache()
    print(f"wrote {out_p}")


class Deep(nn.Module):
    """3-layer GAT or GraphSAGE (receptive field 3), the registry's 2-layer models with one more hidden layer."""

    def __init__(self, kind, i, h, o, heads=4):
        super().__init__()
        self.num_hops = 3
        if kind == "gat":
            self.l = nn.ModuleList([GATConv(i, h, heads=heads), GATConv(h * heads, h, heads=heads), GATConv(h * heads, o)])
        else:
            self.l = nn.ModuleList([SAGEConv(i, h), SAGEConv(h, h), SAGEConv(h, o)])

    def forward(self, x, ei):
        for k, layer in enumerate(self.l):
            x = layer(x, ei)
            if k < len(self.l) - 1:
                x = torch.relu(x)
        return x, None, None


def deep(a):
    out_p = C.RESULTS / "D7" / f"deep_{a.variant}_{a.model}.json"
    res = C.read_json(out_p, dict(variant=a.variant, model=a.model, seeds={}))
    for seed in a.seeds:
        if str(seed) in res["seeds"]:
            continue
        cfg, ds, plan, ei = graph_after_base(a.variant, seed)
        n = ds.num_nodes
        hidden = 64 if n < 50000 else 128
        torch.manual_seed(seed)
        model = Deep(a.model, ds.in_dim, hidden, ds.out_dim).double()
        x, y = ds.data.x.double(), ds.data.y
        g = torch.Generator().manual_seed(seed + 0xD7)
        pool = ds.train_mask.nonzero(as_tuple=True)[0]
        roots = pool[torch.randperm(pool.numel(), generator=g)[:64]]

        def grads(xx, e, pos, yy):
            model.zero_grad()
            F.cross_entropy(model(xx, e)[0][pos], yy).backward()
            return torch.cat([p.grad.reshape(-1) for p in model.parameters()]).clone()
        ref = grads(x, ei, roots, y[roots])
        row = dict(nodes=n, hidden=hidden)
        for hops in (3, 2):
            sub, sei, mp, _ = k_hop_subgraph(roots, hops, ei, relabel_nodes=True, num_nodes=n)
            gsub = grads(x[sub], sei, mp, y[roots])
            row[f"rel_{hops}hop"] = float((gsub - ref).abs().max() / ref.abs().max())
            row[f"nodes_{hops}hop"] = int(sub.numel())
        if torch.cuda.is_available():                  # one GPU update, full vs selective, float32
            dev = torch.device("cuda")
            m32 = Deep(a.model, ds.in_dim, hidden, ds.out_dim).to(dev)
            data = ds.data.clone(); data.edge_index = ei; data.n_id = torch.arange(n)
            sampler = R.CachedSampler(data, [15, 10, 5], refresh_interval=10 ** 9)
            rng = np.random.default_rng([seed, 0xD70])
            N = min(cfg.adaptation.max_retrain_nodes, int(ds.train_mask.sum()) // 4)
            for method in ("full", "selective"):
                rr = (rng.choice(pool.numpy(), N, replace=False) if method == "full"
                      else region_roots(ei, n, N, rng, ds.train_mask.numpy()))
                b = sampler.sample(torch.from_numpy(rr).long(), seed, 0)
                row[f"gpu_{method}"] = dict(nodes=int(b.num_nodes), **step_cost(m32, b.x, b.edge_index, b.batch_size, b.y, dev))
        res["seeds"][str(seed)] = row
        C.write_json(out_p, res)
        print(f"  D7 deep {a.variant}/{a.model} seed {seed}: 3-hop {row['rel_3hop']:.1e} (exact)  2-hop {row['rel_2hop']:.1e}"
              + (f"  | GPU full {row['gpu_full']['nodes']:,} vs selective {row['gpu_selective']['nodes']:,} nodes"
                 if "gpu_full" in row else ""), flush=True)
    print(f"wrote {out_p}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("what", choices=["memory", "deep"])
    ap.add_argument("--variant", required=True, choices=list(C.VARIANTS))
    ap.add_argument("--model", required=True, choices=("gat", "sage"))
    ap.add_argument("--seeds", type=int, nargs="+", default=list(C.REPORTED_SEEDS))
    a = ap.parse_args()
    C.guard(a.seeds)
    (memory if a.what == "memory" else deep)(a)


if __name__ == "__main__":
    main()
