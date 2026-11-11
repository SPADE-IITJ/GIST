"""E4: does a cost-bounding update compute the update that was intended? (intro Gap 3)

    python -m experiments.e4_fidelity --model gat --variant cora_label [--seeds 110 ...] [--repair-device cuda]

Setup per seed: the prelim stream with label (concept) drift; the graph, features and labels are replayed with the
warm model frozen up to the first drift event + 1. The repair update is fixed for every method: roots R = 0.75 N of
the event's trainable drifted nodes plus 0.25 N uniform anchors (GIST's selective set with the true region),
loss = mean cross-entropy over R.

Part 1, gradient at the warm weights (float64, CPU), each method vs the exact full-graph gradient:
  exact_lhop    GIST: the induced 2-hop subgraph of R (GCN keeps the parent's normalisation)
  one_hop       the 1-hop subgraph (truncated receptive field)
  ns_a_b        neighbour sampling with fanout [a, b] (PyG NeighborLoader); [2, 2] is Online CGL's setting;
                three draws, mean error and the error of their mean (unbiasedness)
  cluster_gcn   Cluster-GCN's approximation: the block-diagonal graph (edges between METIS clusters dropped),
                clusters of ~150 nodes as in its Reddit / Amazon2M setups (k = clamp(n / 150, 16, 20000))
  saint_rw      GraphSAINT random-walk sampler, walks of length 2 from each root; induced subgraph
  saint_node    GraphSAINT node sampler: R plus 4|R| nodes drawn with p ~ degree^2; induced subgraph
  saint_edge    GraphSAINT edge sampler: R plus the endpoints of 4|R| edges drawn with p ~ 1/d_u + 1/d_v
  gas           GNNAutoScale: in-batch nodes B = R, out-of-batch layer-1 activations from a history (fresh at the
                warm weights) with no gradient through them; gas_cluster: B = the ~150-node clusters holding R
  vrgcn         VR-GCN-style: B = R plus 2 sampled neighbours per root, others from the history
  lmc           LMC (repair only): GAS's forward plus a backward compensation for the out-of-batch
                messages from stored gradients. With fresh stores the compensated gradient is exact, so LMC's error is
                staleness: lmc@s refreshes the stores every s steps (compensation = exact - GAS gradient at the last
                refresh, added to the current GAS gradient)
GraphSAINT's aggregator normalisation is not applied (our backbones take no per-edge coefficients except GCN); its
loss normalisation is moot because the loss is on R only.

Part 2, repair (float32, --repair-device): k = 1, 5, 20 Adam steps (the run's retrain_lr) with each method's
gradient, same R; GAS's history refreshed every s in {1, 10} steps; IGLU-style lazy updates apply the gradient
computed s in {1, 5} steps earlier (stale). After each k: holdout accuracy on drifted and on clean nodes (exact
full-graph inference) and disagreement with the exactly repaired model.
"""
from __future__ import annotations

import argparse
import copy
import time

import numpy as np
import torch
import torch.nn.functional as F
from torch_geometric.utils import k_hop_subgraph, subgraph

from . import common as C

R = C.R
FANOUTS = ([2, 2], [5, 5], [15, 10], [25, 15])


def replay_to(ds, plan, step):
    """Graph, features and labels after `step` (inclusive), as every arm sees them."""
    from gist.stream.drift_process import apply_drift
    from gist.stream.graph_state import GraphState
    g = GraphState(ds.num_nodes, plan.base_edges)
    x, y = ds.data.x.clone(), ds.data.y.clone()
    for t in range(step + 1):
        g.add_edges(plan.edges_at(t))
        for ev in plan.drift_at(t):
            apply_drift(ev, x, y, g)
    return g.edge_index, x, y


def fwd(model, x, ei, ew=None):
    return (model(x, ei, ew) if ew is not None else model(x, ei))[0]


def split(model, name):
    """(layer 1 -> post-activation h1, layer 2) of a 2-layer backbone; GCN layers take edge weights."""
    if name == "gat":
        return (lambda x, ei, ew=None: torch.relu(model.gat1(x, ei))), (lambda h, ei, ew=None: model.gat2(h, ei))
    if name == "gcn":
        c1, c2 = model.convs
        return (lambda x, ei, ew=None: torch.relu(c1(x, ei, ew))), (lambda h, ei, ew=None: c2(h, ei, ew))
    return (lambda x, ei, ew=None: torch.relu(model.conv1(x, ei))), (lambda h, ei, ew=None: model.conv2(h, ei))


class Graph:
    """The current graph on one device, with the losses every method computes for roots R."""

    def __init__(self, model, name, x, y, ei, n):
        self.model, self.name, self.x, self.y, self.ei, self.n = model, name, x, y, ei, n
        self.pw = model.parent_edges(ei, n, dtype=x.dtype) if hasattr(model, "parent_edges") else None

    def _full_edges(self):
        return self.pw if self.pw is not None else (self.ei, None)

    def exact_full(self, roots):
        ei, w = self._full_edges()
        out = fwd(self.model, self.x, ei, w)
        return F.cross_entropy(out[roots], self.y[roots]), self.n, int(self.ei.shape[1])

    def lhop(self, roots, hops):
        from gist.models.exact import weighted_subgraph
        if self.pw is not None:
            subset, sub_ei, mapping, w = weighted_subgraph(roots, hops, self.pw[0], self.pw[1], self.n)
        else:
            subset, sub_ei, mapping, _ = k_hop_subgraph(roots, hops, self.ei, relabel_nodes=True, num_nodes=self.n)
            w = None
        out = fwd(self.model, self.x[subset], sub_ei, w)
        return F.cross_entropy(out[mapping], self.y[roots]), int(subset.numel()), int(sub_ei.shape[1])

    def induced(self, nodes, roots):
        """Loss on `roots` over the subgraph induced by nodes + roots (GCN renormalises on it, as samplers do)."""
        nodes = torch.unique(torch.cat([nodes, roots]))
        sub_ei, _ = subgraph(nodes, self.ei, relabel_nodes=True, num_nodes=self.n)
        pos = torch.full((self.n,), -1, dtype=torch.long, device=nodes.device)
        pos[nodes] = torch.arange(nodes.numel(), device=nodes.device)
        out = fwd(self.model, self.x[nodes], sub_ei)
        return F.cross_entropy(out[pos[roots]], self.y[roots]), int(nodes.numel()), int(sub_ei.shape[1])

    def sampled(self, roots, fanout, seed):
        from torch_geometric.data import Data
        from torch_geometric.loader import NeighborLoader
        torch.manual_seed(seed)
        data = Data(x=self.x.cpu(), y=self.y.cpu(), edge_index=self.ei.cpu(), num_nodes=self.n)
        b = next(iter(NeighborLoader(data, num_neighbors=fanout, input_nodes=roots.cpu(),
                                     batch_size=int(roots.numel()), shuffle=False)))
        dev = self.x.device
        out = fwd(self.model, b.x.to(dev), b.edge_index.to(dev))
        return (F.cross_entropy(out[: b.batch_size], b.y[: b.batch_size].to(dev)), int(b.num_nodes),
                int(b.edge_index.shape[1]))

    def block(self, roots, cid):
        """Cluster-GCN: the whole graph with between-cluster edges removed (GCN renormalises on it, as Cluster-GCN
        does). Cost = the nodes and edges of the clusters holding R, which is all its minibatch materialises."""
        keep = cid[self.ei[0]] == cid[self.ei[1]]
        ei_b = self.ei[:, keep]
        out = fwd(self.model, self.x, ei_b)
        held = torch.isin(cid, torch.unique(cid[roots]))
        return (F.cross_entropy(out[roots], self.y[roots]), int(held.sum()),
                int((held[ei_b[0]] & held[ei_b[1]]).sum()))

    def hist(self):
        l1, _ = split(self.model, self.name)
        ei, w = self._full_edges()
        with torch.no_grad():
            return l1(self.x, ei, w).detach()

    def history_mix(self, roots, batch_nodes, hist):
        """GAS / VR-GCN: layer-1 activations outside the batch come from `hist`, with no gradient through them."""
        l1, l2 = split(self.model, self.name)
        ei, w = self._full_edges()
        h = l1(self.x, ei, w)
        inb = torch.zeros(self.n, dtype=torch.bool, device=h.device)
        inb[batch_nodes] = True
        h = torch.where(inb[:, None], h, hist)
        out = l2(h, ei, w)
        return F.cross_entropy(out[roots], self.y[roots]), int(batch_nodes.numel()), int(self.ei.shape[1])

    def run(self, spec, roots, hist=None, seed=0):
        kind = spec[0]
        if kind == "full":
            return self.exact_full(roots)
        if kind == "lhop":
            return self.lhop(roots, spec[1])
        if kind == "induced":
            return self.induced(spec[1].to(roots.device), roots)
        if kind == "ns":
            return self.sampled(roots, spec[1], spec[2] + seed)
        if kind == "block":
            return self.block(roots, spec[1].to(roots.device))
        if kind == "hist":
            return self.history_mix(roots, spec[1].to(roots.device), hist)
        raise ValueError(kind)


def specs_for(ei, n, roots, cid, seed):
    """Every method's update for roots R, as data. Random draws are made once here (one minibatch per method)."""
    g = torch.Generator().manual_seed(seed + 0x5A1)
    src, dst = ei[0], ei[1]
    order = torch.argsort(src)
    src, dst = src[order], dst[order]
    deg = torch.bincount(src, minlength=n)
    rowptr = torch.zeros(n + 1, dtype=torch.long)
    rowptr[1:] = torch.cumsum(deg, 0)

    def rand_nbr(u):
        d = deg[u]
        off = (torch.rand(u.numel(), generator=g) * d.clamp(min=1)).long()
        v = dst[(rowptr[u] + off).clamp(max=dst.numel() - 1)]
        return torch.where(d > 0, v, u)

    clusters = torch.from_numpy(np.nonzero(np.isin(cid, np.unique(cid[roots.numpy()])))[0]).long()
    walk, cur = [roots], roots
    for _ in range(2):
        cur = rand_nbr(cur)
        walk.append(cur)
    degf = deg.double()
    node_s = torch.multinomial((degf.pow(2) + 1e-12).float(), min(4 * roots.numel(), n), replacement=False, generator=g)
    pe = 1 / degf[ei[0]].clamp(min=1) + 1 / degf[ei[1]].clamp(min=1)
    e_s = torch.multinomial(pe.float(), min(4 * roots.numel(), ei.shape[1]), replacement=False, generator=g)
    vr = torch.unique(torch.cat([roots, rand_nbr(roots), rand_nbr(roots)]))
    s = {"exact_lhop": ("lhop", 2), "one_hop": ("lhop", 1), "cluster_gcn": ("block", torch.from_numpy(cid).long()),
         "saint_rw": ("induced", torch.unique(torch.cat(walk))), "saint_node": ("induced", node_s),
         "saint_edge": ("induced", torch.unique(ei[:, e_s].reshape(-1))), "gas": ("hist", roots),
         "gas_cluster": ("hist", clusters),
         "vrgcn": ("hist", vr)}
    for f in FANOUTS:
        s[f"ns_{f[0]}_{f[1]}"] = ("ns", f, 0)
    return s


def grad_of(model, fn):
    model.zero_grad()
    loss, nodes, edges = fn()
    loss.backward()
    return torch.cat([p.grad.reshape(-1) for p in model.parameters()]).detach().clone(), nodes, edges


def compare(g, ref):
    d = g - ref
    return dict(rel_err=float(d.abs().max() / ref.abs().max()), rel_l2=float(d.norm() / ref.norm()),
                cos=float(torch.dot(g, ref) / (g.norm() * ref.norm() + 1e-300)))


REPAIR = ["exact_lhop", "one_hop", "cluster_gcn", "saint_rw", "saint_node", "saint_edge", "ns_2_2", "ns_5_5",
          "ns_15_10", "ns_25_15", "gas@1", "gas@10", "gas_cluster@1", "vrgcn@1", "iglu@1", "iglu@5", "lmc@5", "lmc@10"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True, choices=C.MODELS)
    ap.add_argument("--variant", required=True, choices=list(C.VARIANTS))
    ap.add_argument("--seeds", type=int, nargs="+", default=list(C.REPORTED_SEEDS))
    ap.add_argument("--repair-device", default="cpu")
    ap.add_argument("--ks", type=int, nargs="+", default=[1, 5, 20])
    ap.add_argument("--calibration", action="store_true")
    a = ap.parse_args()
    C.guard(a.seeds, C.CAL_SEEDS if a.calibration else None)
    cfg = C.config(a.model, a.variant, **{"model.warmup_epochs": C.calibration(a.model, a.variant)["warmup_epochs"]})
    out_p = C.RESULTS / "E4" / f"{a.model}_{a.variant}{'_cal' if a.calibration else ''}.json"
    res = C.read_json(out_p, dict(model=a.model, variant=a.variant, seeds={}))
    from gist.drift.partitioners import metis_partition
    for seed in a.seeds:
        if str(seed) in res["seeds"]:
            continue
        t0 = time.time()
        ds, plan = C.stream(cfg, a.variant, seed)
        warm = C.warm_state(cfg, ds, plan, seed,
                            key=a.variant.split("_")[0] if a.variant.endswith(("_label", "_struct")) else a.variant)
        ev = plan.drift_events[0]
        ei, x, y = replay_to(ds, plan, ev.step)
        n = ds.num_nodes
        train = R._train_pool(ds.train_mask, plan, ev.step)
        N = cfg.adaptation.max_retrain_nodes
        gen = torch.Generator().manual_seed(seed + 0xE4)
        drifted = ev.nodes.long()
        tgt = drifted[train[drifted]]
        tgt = tgt[torch.randperm(tgt.numel(), generator=gen)[: int(round(0.75 * N))]]
        pool = train.nonzero(as_tuple=True)[0]
        anc = pool[torch.randperm(pool.numel(), generator=gen)[: N - tgt.numel()]]
        roots = torch.unique(torch.cat([tgt, anc]))
        k = int(np.clip(n // 150, 16, 20000))
        cid = metis_partition(ei, n, num_parts=k).numpy()
        specs = specs_for(ei, n, roots, cid, seed)

        # part 1: float64 gradients at the warm weights, CPU
        model = R.build_backbone(a.model, ds.in_dim, ds.out_dim, cfg.model.hidden, cfg.model.heads)
        model.load_state_dict(copy.deepcopy(warm))
        model = model.double().train()
        G = Graph(model, a.model, x.double(), y, ei, n)
        ref, _, _ = grad_of(model, lambda: G.exact_full(roots))
        hist = G.hist()
        part1 = {}
        for name, spec in specs.items():
            draws = 3 if spec[0] == "ns" else 1
            gs = []
            for d in range(draws):
                gvec, nodes, edges = grad_of(model, lambda: G.run(spec, roots, hist, seed=seed * 31 + d))
                gs.append(gvec)
            errs = [compare(gv, ref) for gv in gs]
            part1[name] = dict(nodes=nodes, edges=edges, **{k_: float(np.mean([e[k_] for e in errs])) for k_ in errs[0]})
            if draws > 1:
                part1[name]["mean_of_draws"] = compare(torch.stack(gs).mean(0), ref)

        # part 2: repair steps, float32 on --repair-device
        dev = torch.device(a.repair_device)
        dmask = torch.zeros(n, dtype=torch.bool)
        dmask[drifted] = True
        lr = cfg.adaptation.retrain_lr if cfg.adaptation.retrain_lr is not None else cfg.model.lr
        part2, preds = {}, {}
        rd = roots.to(dev)
        for meth in REPAIR:
            base, _, s = meth.partition("@")
            s = int(s) if s else 0
            mdl = R.build_backbone(a.model, ds.in_dim, ds.out_dim, cfg.model.hidden, cfg.model.heads)
            mdl.load_state_dict(copy.deepcopy(warm))
            mdl = mdl.to(dev).train()
            opt = torch.optim.Adam(mdl.parameters(), lr=lr, weight_decay=cfg.model.weight_decay)
            Gd = Graph(mdl, a.model, x.float().to(dev), y.to(dev), ei.to(dev), n)
            spec = ("lhop", 2) if base == "iglu" else ("hist", roots) if base == "lmc" else specs[base]
            hist_d, stale, rec, comp = None, [], {}, None
            for step in range(1, max(a.ks) + 1):
                if spec[0] == "hist" and (hist_d is None or (step - 1) % max(s, 1) == 0):   # GAS / VR-GCN / LMC histories
                    hist_d = Gd.hist()
                if base == "lmc" and (step - 1) % s == 0:       # refresh LMC's stores: exact - GAS at these weights
                    opt.zero_grad()
                    Gd.exact_full(rd)[0].backward()
                    g_ex = [p_.grad.detach().clone() for p_ in mdl.parameters()]
                    opt.zero_grad()
                    Gd.history_mix(rd, rd, hist_d)[0].backward()
                    comp = [a_ - p_.grad.detach() for a_, p_ in zip(g_ex, mdl.parameters())]
                opt.zero_grad()
                loss, _, _ = Gd.run(spec, rd, hist_d, seed=seed * 31 + step)
                loss.backward()
                if base == "lmc":
                    for p_, c_ in zip(mdl.parameters(), comp):
                        p_.grad.add_(c_)
                if base == "iglu":       # lazy: apply the gradient computed s steps ago
                    stale.append([p.grad.detach().clone() for p in mdl.parameters()])
                    for p, gg in zip(mdl.parameters(), stale[max(0, len(stale) - 1 - s)]):
                        p.grad = gg.clone()
                opt.step()
                if step in a.ks:
                    with torch.no_grad():
                        ei_e, w_e = Gd._full_edges()
                        out = fwd(mdl, Gd.x, ei_e, w_e).argmax(1).cpu()
                    preds[(meth, step)] = out
                    ok = out == y
                    em = ds.eval_mask
                    rec[str(step)] = dict(acc_drifted=float(ok[em & dmask].float().mean()),
                                          acc_clean=float(ok[em & ~dmask].float().mean()),
                                          acc_all=float(ok[em].float().mean()))
            part2[meth] = rec
        with torch.no_grad():
            mdl = R.build_backbone(a.model, ds.in_dim, ds.out_dim, cfg.model.hidden, cfg.model.heads)
            mdl.load_state_dict(copy.deepcopy(warm))
            mdl = mdl.to(dev).eval()
            Gd = Graph(mdl, a.model, x.float().to(dev), y.to(dev), ei.to(dev), n)
            ei_e, w_e = Gd._full_edges()
            ok0 = fwd(mdl, Gd.x, ei_e, w_e).argmax(1).cpu() == y
            em = ds.eval_mask
            frozen = dict(acc_drifted=float(ok0[em & dmask].float().mean()), acc_clean=float(ok0[em & ~dmask].float().mean()),
                          acc_all=float(ok0[em].float().mean()))
        for meth in REPAIR:
            for k_ in a.ks:
                part2[meth][str(k_)]["disagree_exact"] = float(
                    (preds[(meth, k_)] != preds[("exact_lhop", k_)])[ds.eval_mask].float().mean())
        res["seeds"][str(seed)] = dict(event_step=int(ev.step), event_kind=ev.kind, drifted=int(drifted.numel()),
                                       roots=int(roots.numel()), clusters=k, frozen=frozen, part1=part1, part2=part2,
                                       wall_s=round(time.time() - t0, 1))
        C.write_json(out_p, res)
        e = part1
        print(f"  E4 {a.model}/{a.variant} seed {seed}: exact_lhop {e['exact_lhop']['rel_err']:.1e} one_hop {e['one_hop']['rel_err']:.1e} "
              f"ns15/10 {e['ns_15_10']['rel_err']:.1e} cluster {e['cluster_gcn']['rel_err']:.1e} gas {e['gas']['rel_err']:.1e} "
              f"saint_rw {e['saint_rw']['rel_err']:.1e} | {max(a.ks)}-step drifted acc exact {part2['exact_lhop'][str(max(a.ks))]['acc_drifted']:.3f} "
              f"frozen {frozen['acc_drifted']:.3f} ({time.time() - t0:.0f}s)", flush=True)
    print(f"wrote {out_p}")


if __name__ == "__main__":
    main()
