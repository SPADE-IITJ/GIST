"""D2 backbone coverage (C4 / G3 / G5) on the evaluation seeds: is the induced-subgraph gradient exact, per backbone,
with and without the fixes, at the exact depth and one hop short?

    $B -m experiments.backbones --seeds 0 1 ... 9      (env-backbones: PyG Temporal)    -> <results>/D2_backbones.json

It reuses the repository's own exactness tests (tests/test_backbone_fixes.py, tests/test_exactness_backbones.py),
re-seeding their module-level SEED for every seed, so these numbers come from exactly the code the tests assert on.
Two metrics: the tests' per-tensor max-relative difference, and the global scale (max |delta| over all parameters /
max |gradient| over all parameters), the C4 criterion.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import torch
import torch.nn as nn

from . import common as C

sys.path.insert(0, str(C.ROOT / "tests"))
import test_exactness_backbones as TE  # noqa: E402
import test_backbone_fixes as TB  # noqa: E402
from gist.models.exact import (ParentNormGCN, exact_gconvgru, exact_tgcn, evolvegcno_exact_kwargs,  # noqa: E402
                                      frozen_norm_stats, parent_cheb_weights, parent_gcn_weights)


def _global(ref, sub):
    num = max(float((ref[n] - sub[n]).abs().max()) for n in ref)
    den = max(float(ref[n].abs().max()) for n in ref)
    return num / max(den, 1e-300)


def compare2(model, forward, batch, ei, w, hops, x=None):
    """Both metrics for one full-batch vs `hops`-hop subgraph comparison (TB._compare, extended)."""
    x = batch.x if x is None else x
    targets = TE._targets(batch)
    y = batch.y[targets]
    ref = TB._grads(model, lambda: forward(x, ei, w, None), y, targets)
    subset, sub_ei, mapping, sub_w = TB.weighted_subgraph(targets, hops, ei, w, batch.num_nodes)
    sub = TB._grads(model, lambda: forward(x[subset], sub_ei, sub_w, subset), y, mapping)
    return dict(per_tensor=TE._rel(ref, sub), global_scale=_global(ref, sub))


def one_seed(seed):
    TE.SEED = TB.SEED = seed
    batch, i, o = TE._batch()
    DT, H = TE.DTYPE, TE.HIDDEN
    rows = []

    def add(backbone, fixed, hops, d, note=""):
        rows.append(dict(seed=seed, backbone=backbone, fixed=fixed, hops=hops, **d, note=note))

    for name in ("gat", "sage", "gin"):                       # exact as they are
        torch.manual_seed(seed)
        m = dict(gat=TE.GAT, sage=TE.SAGE, gin=getattr(TE, "GIN", None))[name]
        if m is None:
            continue
        model = m(i, H, o).to(DT)
        fwd = lambda x, e, ww, s, model=model: model(x, e, None) if "w" in model.forward.__code__.co_varnames else model(x, e)
        for hops in (1, 2):
            add(name, None, hops, compare2(model, lambda x, e, ww, s, model=model: model(x, e), batch, batch.edge_index, None, hops))
    torch.manual_seed(seed)
    model = ParentNormGCN(i, H, o).to(DT)
    ei, w = parent_gcn_weights(batch.edge_index, batch.num_nodes, dtype=DT)
    for hops in (1, 2, 3):
        add("gcn", True, hops, compare2(model, lambda x, e, ww, s: model(x, e, ww), batch, ei, w, hops))
    add("gcn", False, 2, compare2(model, lambda x, e, ww, s: model(x, e), batch, batch.edge_index, None, 2))
    torch.manual_seed(seed)
    model = TE.SAGE(i, H, o, norm=True).to(DT).train()
    with torch.no_grad():
        model.bn.running_mean.normal_(); model.bn.running_var.uniform_(0.5, 2.0)
    fwd = lambda x, e, ww, s: model(x, e)
    with frozen_norm_stats(model):
        add("batchnorm", True, 2, compare2(model, fwd, batch, batch.edge_index, None, 2))
        add("batchnorm", True, 1, compare2(model, fwd, batch, batch.edge_index, None, 1))
    add("batchnorm", False, 2, compare2(model, fwd, batch, batch.edge_index, None, 2))
    try:
        import torch_geometric_temporal.nn.recurrent as rec
    except ImportError:
        return rows, "PyG Temporal missing: snapshot models skipped (run with $B)"
    b2, o2, x, state = TB._snapshot((batch, i, o))

    def run(model, st, ei, w, hops, **kw):
        fwd = lambda xx, e, ww, s: model(xx, e, ww, None if st is None else [h if s is None else h[s] for h in st], **kw)
        return compare2(model, fwd, b2, ei, w, hops, x=x)

    for fixed in (True, False):
        torch.manual_seed(seed)
        kw = evolvegcno_exact_kwargs() if fixed else {}
        model = TB.TwoCells([rec.EvolveGCNO(TB.FEAT, **kw), rec.EvolveGCNO(TB.FEAT, **kw)], o2, stateful=False).to(DT)
        ei, w = parent_gcn_weights(b2.edge_index, b2.num_nodes, dtype=DT) if fixed else (b2.edge_index, None)
        for hops in ((1, 2, 3) if fixed else (2,)):
            add("evolvegcn_o", fixed, hops, run(model, None, ei, w, hops))
        torch.manual_seed(seed)
        model = TB.TwoCells([rec.TGCN(TB.FEAT, TB.FEAT), rec.TGCN(TB.FEAT, TB.FEAT)], o2, stateful=True).to(DT)
        if fixed:
            exact_tgcn(model)
        ei, w = parent_gcn_weights(b2.edge_index, b2.num_nodes, dtype=DT) if fixed else (b2.edge_index, None)
        for hops in ((1, 2, 3) if fixed else (2,)):
            add("tgcn", fixed, hops, run(model, state, ei, w, hops))
    for norm in ("sym", None):
        for fixed in (True, False):
            torch.manual_seed(seed)
            model = TB.TwoCells([rec.GConvGRU(TB.FEAT, TB.FEAT, K=2, normalization=norm) for _ in range(2)], o2,
                                stateful=True).to(DT)
            lam = None if norm == "sym" else float(2 * torch.bincount(b2.edge_index[0], minlength=b2.num_nodes).max())
            kw = {} if lam is None else dict(lambda_max=torch.tensor(lam, dtype=DT))
            if fixed:
                exact_gconvgru(model)
                ei, w = parent_cheb_weights(model, b2.edge_index, b2.num_nodes, lambda_max=lam, dtype=DT)
                kw = {}
            else:
                ei, w = b2.edge_index, None
            for hops in ((3, 4, 5) if fixed else (4, 6)):
                add(f"gconvgru_{norm or 'comb'}", fixed, hops, run(model, state, ei, w, hops, **kw))
    return rows, ""


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, nargs="+", default=list(C.REPORTED_SEEDS))
    a = ap.parse_args()
    C.guard(a.seeds)
    out = C.RESULTS / "D2_backbones.json"
    res = C.read_json(out, dict(rows=[], notes=[]))
    done = {r["seed"] for r in res["rows"]}
    for seed in a.seeds:
        if seed in done:
            continue
        t = time.time()
        rows, note = one_seed(seed)
        res["rows"] += rows
        if note:
            res["notes"].append(note)
        C.write_json(out, res)
        worst = max((r["global_scale"] for r in rows if r["fixed"] is not False and r["note"] == ""
                     and not (r["backbone"].startswith("gconvgru") and r["hops"] < 4) and r["hops"] >= 2), default=0)
        print(f"  D2 backbones seed {seed}: {len(rows)} checks, worst exact-config global {worst:.1e} ({time.time() - t:.0f}s)",
              flush=True)
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
