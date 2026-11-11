"""E1: how much of the graph a parameter update must materialise (intro, "The cost of a parameter update").

    python -m experiments.e1_census --graph arxiv [--seeds 110 ...] [--budgets 128 512 ...]

For N roots and depth L, the exact L-hop expansion (sparse BFS on the undirected graph), and what neighbour sampling
materialises (PyG NeighborLoader, fanouts [15, 10] and [25, 15]). Roots are either uniform over the graph (a
conventional update) or drawn from METIS communities (k = clamp(n / 1000, 16, 256)), taking random communities until
they hold N nodes (a region update of the same size). Writes results/E1/<graph>.json.
"""
from __future__ import annotations

import argparse
import time

import numpy as np
import scipy.sparse as sp
import torch

from . import common as C

NC = {"cora": "Cora", "pubmed": "PubMed", "photo": "Photo", "arxiv": "ogbn-arxiv", "elliptic": "elliptic",
      "products": "ogbn-products"}
FANOUTS = ([15, 10], [25, 15])


def load_graph(name):
    if name in NC:
        from gist.data.loaders import load_dataset
        ds = load_dataset(NC[name], C.P.DATA_ROOT, 0.6, split_seed=0)
        return ds.data.edge_index, ds.num_nodes, "node-classification"
    from linkpred.data import GRAPHS, load
    g = load(name)
    return g["edges"], g["num_nodes"], GRAPHS[name]["family"]


def csr(ei, n):
    src, dst = ei[0].numpy(), ei[1].numpy()
    keep = src != dst
    a = sp.coo_matrix((np.ones(int(keep.sum()), np.bool_), (src[keep], dst[keep])), shape=(n, n)).tocsr()
    return ((a + a.T) > 0).astype(np.int8).tocsr()


def expansion(a, roots, L):
    seen = np.zeros(a.shape[0], np.bool_)
    seen[roots] = True
    frontier = seen.copy()
    sizes = []
    for _ in range(L):
        nxt = (a @ frontier.astype(np.int8)) > 0
        frontier = nxt & ~seen
        seen |= nxt
        sizes.append(int(seen.sum()))
    return sizes


def region_roots(cid, N, rng):
    order = rng.permutation(int(cid.max()) + 1)
    take, total = [], 0
    for c in order:
        take.append(c)
        total += int((cid == c).sum())
        if total >= N:
            break
    pool = np.nonzero(np.isin(cid, take))[0]
    return rng.choice(pool, min(N, pool.size), replace=False), len(take)


def sampled_nodes(ei_und, n, roots, fanout, seed):
    from torch_geometric.data import Data
    from torch_geometric.loader import NeighborLoader
    torch.manual_seed(seed)
    loader = NeighborLoader(Data(edge_index=ei_und, num_nodes=n), num_neighbors=fanout,
                            input_nodes=torch.from_numpy(roots).long(), batch_size=len(roots), shuffle=False)
    b = next(iter(loader))
    return int(b.num_nodes), int(b.edge_index.shape[1])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--graph", required=True)
    ap.add_argument("--seeds", type=int, nargs="+", default=list(C.REPORTED_SEEDS))
    ap.add_argument("--budgets", type=int, nargs="+", default=[128, 512, 2048, 8192, 32768])
    ap.add_argument("--depths", type=int, nargs="+", default=[1, 2, 3])
    a = ap.parse_args()
    C.guard(a.seeds)
    t0 = time.time()
    ei, n, family = load_graph(a.graph)
    A = csr(ei, n)
    ei_und = torch.from_numpy(np.vstack(A.nonzero())).long()
    deg = np.asarray(A.sum(1)).ravel()
    from gist.drift.partitioners import metis_partition
    k = int(np.clip(n // 1000, 16, 256))
    cid = metis_partition(ei_und, n, num_parts=k).numpy()
    rows = []
    for seed in a.seeds:
        rng = np.random.default_rng([seed, 0xE1])
        for N in [b for b in a.budgets if b <= n // 2]:
            for kind in ("uniform", "region"):
                if kind == "uniform":
                    roots, ncomm = rng.choice(n, N, replace=False), 0
                else:
                    roots, ncomm = region_roots(cid, N, rng)
                ex = expansion(A, roots, max(a.depths))
                row = dict(seed=seed, N=N, roots=kind, communities=ncomm,
                           exact={str(L): ex[L - 1] for L in a.depths})
                for f in FANOUTS:
                    nn_, ne = sampled_nodes(ei_und, n, roots, f, seed)
                    row[f"fanout_{f[0]}_{f[1]}"] = dict(nodes=nn_, edges=ne)
                rows.append(row)
                print(f"  {a.graph} seed {seed} N {N:>6} {kind:<7} 2-hop {ex[1] / n:6.1%} of {n:,}  "
                      f"fanout15/10 {row['fanout_15_10']['nodes'] / n:6.1%}", flush=True)
    out = dict(graph=a.graph, family=family, num_nodes=n, num_edges=int(A.nnz // 2), mean_degree=float(deg.mean()),
               max_degree=int(deg.max()), metis_parts=k, rows=rows, wall_s=round(time.time() - t0, 1))
    C.write_json(C.RESULTS / "E1" / f"{a.graph}.json", out)
    print(f"wrote results/E1/{a.graph}.json ({out['wall_s']} s)")


if __name__ == "__main__":
    main()
