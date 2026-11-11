"""E1 with GIST's root rule (review experiments): the 2 x 2 of root choice x parent graph.

    python -m experiments.e1_mix --graph arxiv [--seeds ...] [--budgets 2048 8192]

For N roots: uniform (Full's rule); one region (METIS communities until N nodes, as E1); and GIST's rule, (1 - rho) N
from one region plus rho N uniform rehearsal roots (rho = 0.25), counted as distinct roots. For each root set, the
nodes an update materialises with the whole graph as parent (exact 2-hop expansion) and with a fan-out-limited parent
(PyG NeighborLoader [25, 15], the streams' fan-out). Static graph, no training. Writes results/E1mix/<graph>.json.
"""
from __future__ import annotations

import argparse
import time

import numpy as np
import torch

from . import common as C
from .e1_census import csr, expansion, load_graph, region_roots, sampled_nodes

RHO = 0.25


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--graph", required=True)
    ap.add_argument("--seeds", type=int, nargs="+", default=list(C.REPORTED_SEEDS))
    ap.add_argument("--budgets", type=int, nargs="+", default=[2048, 8192])
    a = ap.parse_args()
    C.guard(a.seeds)
    t0 = time.time()
    ei, n, family = load_graph(a.graph)
    A = csr(ei, n)
    ei_und = torch.from_numpy(np.vstack(A.nonzero())).long()
    from gist.drift.partitioners import metis_partition
    k = int(np.clip(n // 1000, 16, 256))
    cid = metis_partition(ei_und, n, num_parts=k).numpy()
    rows = []
    for seed in a.seeds:
        rng = np.random.default_rng([seed, 0xE1, 0x2])
        for N in [b for b in a.budgets if b <= n // 2]:
            uni = rng.choice(n, N, replace=False)
            reg, _ = region_roots(cid, N, rng)
            r_part, _ = region_roots(cid, int(round((1 - RHO) * N)), rng)
            gist = np.unique(np.concatenate([r_part, rng.choice(n, int(round(RHO * N)), replace=False)]))
            for kind, roots in (("uniform", uni), ("region", reg), ("gist", gist)):
                ex = expansion(A, roots, 2)
                fn, fe = sampled_nodes(ei_und, n, roots, [25, 15], seed)
                rows.append(dict(seed=seed, N=N, roots=kind, n_roots=int(len(roots)), exact_nodes=ex[1],
                                 fanout_25_15_nodes=fn, fanout_25_15_edges=fe))
                print(f"  {a.graph} seed {seed} N {N:>5} {kind:<7} |S| {len(roots):>5}  exact {ex[1] / n:6.1%}  "
                      f"fan-out {fn / n:6.1%}", flush=True)
    out = dict(graph=a.graph, family=family, num_nodes=n, rho=RHO, metis_parts=k, rows=rows,
               wall_s=round(time.time() - t0, 1))
    C.write_json(C.RESULTS / "E1mix" / f"{a.graph}.json", out)
    print(f"wrote E1mix/{a.graph}.json ({out['wall_s']} s)")


if __name__ == "__main__":
    main()
