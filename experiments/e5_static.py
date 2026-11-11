"""E5, the parts that need no training: the placeholders `\\res{d}`, `\\res{e}` and `\\res{f}` in intro.tex.

    python -m experiments.e5_static coverage --variant arxiv [--seeds 110 ...]
    python -m experiments.e5_static halo --variant arxiv [--seeds 110 ...]

coverage: for every injected drift event of the prelim stream, the trainable nodes it affects, and the share a
    fixed 128-root update could cover (128 / affected, capped at 1), next to the matched budget N of the run.
halo: in a training batch as the prelim pipeline samples it (fanout of the config, 1,024 roots), the share of
    supervised nodes (trainable nodes anywhere in the batch, which halo supervision trains on) whose receptive field
    the sampler truncated: a node is truncated when fewer of its true in-neighbours are in the batch's edges than it
    has, and "blind" when it has none at all (the outermost hop). Roots-only supervision trains on the roots alone.
"""
from __future__ import annotations

import argparse

import numpy as np
import torch

from . import common as C


def coverage(variant, seeds):
    cal = C.calibration("gat", variant)
    cfg = C.config("gat", variant, **{"model.warmup_epochs": cal.get("warmup_epochs", 1)})
    rows = []
    for seed in seeds:
        ds, plan = C.stream(cfg, variant, seed)
        for ev in plan.drift_events:
            aff = ev.nodes.long()
            tr = int(ds.train_mask[aff].sum())
            rows.append(dict(seed=seed, step=int(ev.step), kind=ev.kind, affected=int(aff.numel()), affected_train=tr,
                             cover_128=min(1.0, 128 / max(tr, 1)),
                             cover_budget=min(1.0, cfg.adaptation.max_retrain_nodes / max(tr, 1))))
    return dict(budget=cfg.adaptation.max_retrain_nodes, num_nodes=ds.num_nodes, rows=rows)


def halo(variant, seeds, roots_n=1024):
    from torch_geometric.utils import degree
    cal = C.calibration("gat", variant)
    cfg = C.config("gat", variant, **{"model.warmup_epochs": cal.get("warmup_epochs", 1)})
    rows = []
    for seed in seeds:
        ds, plan = C.stream(cfg, variant, seed)
        from gist.stream.graph_state import GraphState
        g = GraphState(ds.num_nodes, plan.base_edges)
        ei = g.edge_index
        data = ds.data.clone()
        data.edge_index = ei
        true_in = degree(ei[1], ds.num_nodes).long()
        pool = ds.train_mask.nonzero(as_tuple=True)[0]
        gen = torch.Generator().manual_seed(seed + 0xE5)
        roots = pool[torch.randperm(pool.numel(), generator=gen)[:roots_n]]
        b = C.R.sample_batch(data, roots, cfg.stream.num_neighbors, seed=seed)
        nid = b.n_id
        in_batch = degree(b.edge_index[1], int(b.num_nodes)).long()
        sup = ds.train_mask[nid]
        trunc = in_batch < true_in[nid]
        blind = (in_batch == 0) & (true_in[nid] > 0)   # has neighbours, none of them in the batch
        root = torch.zeros(int(b.num_nodes), dtype=torch.bool)
        root[: b.batch_size] = True
        rows.append(dict(seed=seed, batch_nodes=int(b.num_nodes), roots=int(b.batch_size),
                         supervised_halo=int(sup.sum()), supervised_roots=int((sup & root).sum()),
                         truncated_share=float(trunc[sup].float().mean()),
                         isolated_share=float((true_in[nid] == 0)[sup].float().mean()),
                         blind_share=float(blind[sup].float().mean()),
                         truncated_share_roots=float(trunc[sup & root].float().mean()),
                         fanout=list(cfg.stream.num_neighbors)))
    return dict(rows=rows)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("what", choices=["coverage", "halo"])
    ap.add_argument("--variant", required=True, choices=list(C.VARIANTS))
    ap.add_argument("--seeds", type=int, nargs="+", default=list(C.REPORTED_SEEDS))
    a = ap.parse_args()
    C.guard(a.seeds)
    out = coverage(a.variant, a.seeds) if a.what == "coverage" else halo(a.variant, a.seeds)
    C.write_json(C.RESULTS / "E5" / f"{a.what}_{a.variant}.json", out)
    r = out["rows"]
    if a.what == "coverage":
        c = [x["cover_128"] for x in r]
        print(f"  coverage {a.variant}: {len(r)} events, affected train {np.median([x['affected_train'] for x in r]):.0f} (median), "
              f"128-root cover {np.percentile(c, 10):.1%}..{np.percentile(c, 90):.1%} (p10..p90)")
    else:
        print(f"  halo {a.variant}: truncated {np.mean([x['truncated_share'] for x in r]):.1%} of halo-supervised nodes, "
              f"blind {np.mean([x['blind_share'] for x in r]):.1%}; roots truncated {np.mean([x['truncated_share_roots'] for x in r]):.1%}")


if __name__ == "__main__":
    main()
