"""Update rules of the methods the introduction contrasts GIST with, as gist retrainers.

Each class reimplements a method's *update rule* inside the same runtime, so it sees the same stream, model, budget
and trigger as `full` and `selective`. The official code is in each method's public repository; where we
deviate, the docstring says so.

| arm            | stands for                         | what an update does                                                     |
|----------------|------------------------------------|-------------------------------------------------------------------------|
| global_periodic| D3-GNN (VLDB'24)                   | on a timer: a full-graph step on every trainable node                   |
| global_trigger | InstantGNN adaptive training (KDD'22)| when the drift signal fires: a full-graph step on every trainable node |
| roland         | ROLAND live update (KDD'22)        | on a timer: `inner` full-graph steps, then theta <- (1-a) theta_old + a theta_new |
| ergnn          | ER-GNN (AAAI'21), CGLB's CM sampler | full's N uniform roots plus a replay buffer chosen by coverage maximisation |
| ewc            | EWC (PNAS'17), as in CGLB          | full's update plus reg * F * (theta - theta*)^2, F = squared task gradient |
| twp            | TWP (AAAI'21), official code       | as ewc, with F = l_l * g_loss^2 + l_t * g_topo^2 (topo = first-layer attention norm; GAT only) |
| cat            | CaT (2023), official code CaT-CGL  | condense the update's data into synthetic nodes, then train on the condensed memory bank only |

`HP` holds their hyperparameters; drivers set it before `run_arm` (tuned on seeds 100-102).
"""
from __future__ import annotations

import copy
import time
from typing import Optional

import torch
import torch.nn.functional as F

from gist.adaptation import retrainers as RT
from gist.adaptation.subgraph import full_batch_stats
from gist.experiments import runner as R

HP = dict(
    ewc_reg=1e3,            # CGLB's EWC "reg"
    twp_lambda_l=1e3,       # TWP lambda_l (loss importance)
    twp_lambda_t=1e3,       # TWP lambda_t (topology importance; GAT only)
    ergnn_buffer_frac=0.25, # replay buffer size as a fraction of the budget N (= GIST's anchor share)
    ergnn_cm_quantile=0.1,  # CM distance threshold d: this quantile of sampled cross-class feature distances
    roland_inner=5,         # ROLAND trains each snapshot for several steps
    roland_alpha=0.5,       # ROLAND meta-update: theta <- (1 - alpha) * theta_prev + alpha * theta_new
    fisher_nodes=4096,      # trainable nodes the importance is estimated on
    cat_bank_frac=0.25,     # CaT memory bank size as a fraction of N (= ER-GNN's buffer)
    cat_per_update=0.05,    # synthetic nodes condensed from each update, as a fraction of N
    cat_inits=5,            # random GNN initialisations the distribution matching averages over
    cat_steps=30,           # condensation steps per update
    cat_lr=0.01,            # learning rate of the synthetic features
)
NEW_ARMS = ("global_periodic", "global_trigger", "roland", "ergnn", "ewc", "twp", "cat")
# Arms whose whole-graph update did not fit on the GPU during the current run_arm (streams.py records and clears it).
OOM = set()


def _device(model):
    return next(model.parameters()).device


def _full_graph(ctx, dev):
    g = ctx.sampler.data
    return g.x.to(dev), g.y.to(dev), g.edge_index.to(dev), int(g.num_nodes)


class GlobalRetrainer(RT.Retrainer):
    """Whole-graph update: forward and backward over the entire current graph, loss on every trainable node.

    D3-GNN pauses the stream, back-propagates over the whole graph and refreshes every embedding with a full-batch
    pass (its Sec. 4); InstantGNN retrains the whole model when its accumulated-change criterion fires (its Sec. 4.5;
    here the criterion is the shared drift trigger). ROLAND trains each new snapshot for `inner` steps and
    meta-averages the weights (its live-update algorithm)."""

    requires_community = False

    def __init__(self, name: str, period: Optional[int] = None, inner: int = 1, alpha: float = 1.0) -> None:
        self.name = name
        self.period = None if period is None else max(1, int(period))
        self.inner, self.alpha = max(1, int(inner)), float(alpha)

    def should_fire(self, step, drift_signal, controller):
        if self.name in OOM:          # a whole-graph step did not fit: the arm cannot update on this GPU
            return False
        if self.period is not None:
            return step > 0 and step % self.period == 0
        return controller.should_retrain(drift_signal, step)

    def apply(self, model, optimizer, ctx) -> Optional[RT.RetrainInfo]:
        dev = _device(model)
        x, y, ei, n = _full_graph(ctx, dev)
        mask = ctx.train_mask.to(dev)
        if not bool(mask.any()):
            return None
        before = {k: v.detach().clone() for k, v in model.state_dict().items()} if self.alpha < 1 else None
        model.train()
        try:
            for _ in range(self.inner):
                optimizer.zero_grad()
                out, _, _ = model(x, ei)
                loss = F.cross_entropy(out[mask], y[mask])
                loss.backward()
                optimizer.step()
        except torch.OutOfMemoryError:
            optimizer.zero_grad(set_to_none=True)
            if before is not None:
                model.load_state_dict(before)
            OOM.add(self.name)
            torch.cuda.empty_cache()
            return None
        if before is not None:
            with torch.no_grad():
                for k, v in model.state_dict().items():
                    if v.is_floating_point():
                        v.copy_((1 - self.alpha) * before[k] + self.alpha * v)
        return RT.RetrainInfo(step=ctx.step, loss=float(loss.item()), nodes_targeted=int(mask.sum()),
                              nodes_updated=int(mask.sum()), nodes_in_backward=n * self.inner,
                              edges_in_backward=int(ei.shape[1]) * self.inner, communities_selected=0,
                              communities_total=0)


def _uniform_roots(ctx, n):
    """full's draw: n trainable roots, seeded by (seed, step) exactly as FullRetrainer._global_batch."""
    pool = ctx.train_mask.nonzero(as_tuple=True)[0]
    g = torch.Generator().manual_seed(ctx.seed + ctx.step)
    return pool[torch.randperm(pool.numel(), generator=g)[:n]]


def _root_step(model, optimizer, batch, penalty=None):
    """One step on the mean loss of the batch roots (roots-only supervision), plus an optional penalty."""
    model.train()
    optimizer.zero_grad()
    out, _, _ = model(batch.x, batch.edge_index)
    loss = F.cross_entropy(out[: batch.batch_size], batch.y[: batch.batch_size])
    total = loss + (penalty() if penalty is not None else 0.0)
    total.backward()
    optimizer.step()
    return float(loss.item())


class _UniformBatchRetrainer(RT.Retrainer):
    """Shared by ER-GNN, EWC and TWP: full's N uniform roots, sampled with the step's sampler, fired by the trigger."""

    requires_community = False

    def __init__(self, max_nodes: int) -> None:
        self.max_nodes = int(max_nodes)

    def _batch(self, ctx, roots, dev):
        t = time.perf_counter()
        b = ctx.sampler.sample(roots, ctx.seed + ctx.step, ctx.step).to(dev)
        return b, time.perf_counter() - t

    def _info(self, ctx, batch, loss, sample_s, roots_drawn):
        s = full_batch_stats(batch)
        return RT.RetrainInfo(step=ctx.step, loss=loss, nodes_targeted=int(batch.batch_size),
                              nodes_updated=int(batch.batch_size), nodes_in_backward=s.nodes_in_backward,
                              edges_in_backward=s.edges_in_backward, communities_selected=0, communities_total=0,
                              region_sample_s=sample_s, roots_drawn=roots_drawn)


class ERGNNRetrainer(_UniformBatchRetrainer):
    """ER-GNN: each update trains on the new roots plus a replay buffer; loss = mean over both, i.e.
    (1 - beta) L_new + beta L_buffer with beta = |B| / (|B| + |new|) (CGLB's ergnn_model). The buffer is chosen per
    class by coverage maximisation (CGLB CM_sampler): nodes with the fewest other-class nodes within distance d in
    feature space. It is built from the trainable pool at the first update (the "previous task") and re-selected by
    CM over buffer + new roots after every update, keeping its size."""

    name = "ergnn"

    def __init__(self, max_nodes: int) -> None:
        super().__init__(max_nodes)
        self.buffer = None
        self.size = max(1, int(round(max_nodes * HP["ergnn_buffer_frac"])))

    def _cm(self, cand, x, y, g):
        classes = torch.unique(y[cand])
        per = max(1, self.size // max(1, classes.numel()))
        xs = x[cand].float()
        far = cand[torch.randperm(cand.numel(), generator=g)[:2000]]
        d_all = torch.cdist(x[far].float(), x[far].float())
        diff = y[far][:, None] != y[far][None, :]
        d = torch.quantile(d_all[diff], HP["ergnn_cm_quantile"]) if bool(diff.any()) else d_all.max()
        out = []
        for c in classes.tolist():
            own = cand[y[cand] == c]
            if own.numel() > 1000:
                own = own[torch.randperm(own.numel(), generator=g)[:1000]]
            other = far[y[far] != c]
            if other.numel() == 0:
                out.append(own[:per]); continue
            cover = (torch.cdist(x[own].float(), x[other].float()) < d).sum(1)
            out.append(own[cover.argsort()[:per]])
        sel = torch.cat(out)
        return sel[: self.size]

    def apply(self, model, optimizer, ctx):
        dev = _device(model)
        g = torch.Generator().manual_seed(ctx.seed * 7 + ctx.step)
        data = ctx.sampler.data
        if self.buffer is None:
            self.buffer = self._cm(ctx.train_mask.nonzero(as_tuple=True)[0], data.x, data.y, g)
        new = _uniform_roots(ctx, self.max_nodes)
        roots = torch.unique(torch.cat([new, self.buffer]))
        batch, s = self._batch(ctx, roots, dev)
        loss = _root_step(model, optimizer, batch)
        self.buffer = self._cm(torch.unique(torch.cat([self.buffer, new])), data.x, data.y, g)
        return self._info(ctx, batch, loss, s, int(roots.numel()))


class EWCRetrainer(_UniformBatchRetrainer):
    """EWC as in CGLB: importance F = (grad of the old task's loss)^2 at theta* (the warm weights; nothing has updated
    before the first fire), penalty reg * F * (theta - theta*)^2. TWP adds the squared gradient of the first-layer
    attention norm (its "topology" importance, official code TWP/*/twp_model.py), weighted lambda_t; for models without
    attention that term is undefined and TWP reduces to EWC with reg = lambda_l. TWP's optional gradient-norm term
    (beta) is off."""

    name = "ewc"

    def __init__(self, max_nodes: int, twp: bool = False) -> None:
        super().__init__(max_nodes)
        self.twp = twp
        self.name = "twp" if twp else "ewc"
        self.importance = self.star = None

    def _importance(self, model, ctx, dev):
        pool = ctx.train_mask.nonzero(as_tuple=True)[0]
        g = torch.Generator().manual_seed(ctx.seed + 0xF15)
        nodes = pool[torch.randperm(pool.numel(), generator=g)[: HP["fisher_nodes"]]]
        batch = ctx.sampler.sample(nodes, ctx.seed + 0xF15, ctx.step).to(dev)
        params = [p for p in model.parameters() if p.requires_grad]
        model.train()
        model.zero_grad()
        out, attn1, _ = model(batch.x, batch.edge_index)
        F.cross_entropy(out[: batch.batch_size], batch.y[: batch.batch_size]).backward(retain_graph=self.twp)
        g_loss = [p.grad.detach().clone().pow(2) if p.grad is not None else torch.zeros_like(p) for p in params]
        if self.twp:
            lam_l, lam_t = HP["twp_lambda_l"], HP["twp_lambda_t"]
            if attn1 is not None:
                model.zero_grad()
                torch.norm(attn1[1]).backward()
                g_topo = [p.grad.detach().clone().pow(2) if p.grad is not None else torch.zeros_like(p) for p in params]
            else:
                g_topo = [torch.zeros_like(p) for p in params]
            imp = [lam_l * a + lam_t * b for a, b in zip(g_loss, g_topo)]
        else:
            imp = [HP["ewc_reg"] * a for a in g_loss]
        model.zero_grad()
        self.importance = imp
        self.star = [p.detach().clone() for p in params]

    def apply(self, model, optimizer, ctx):
        dev = _device(model)
        if self.importance is None:
            self._importance(model, ctx, dev)
        params = [p for p in model.parameters() if p.requires_grad]
        penalty = lambda: sum((w * (p - s).pow(2)).sum() for w, p, s in zip(self.importance, params, self.star))
        roots = _uniform_roots(ctx, self.max_nodes)
        batch, s = self._batch(ctx, roots, dev)
        loss = _root_step(model, optimizer, batch, penalty)
        return self._info(ctx, batch, loss, s, int(roots.numel()))


class CaTRetrainer(_UniformBatchRetrainer):
    """CaT (Liu, Qiu, Huang 2023): Condense and Train. Each update's incoming data (full's N uniform roots, with their
    sampled 2-hop context) is condensed per class into a few synthetic nodes by distribution matching: the synthetic
    features are optimised so that, under several randomly initialised copies of the backbone, the mean embedding of the
    synthetic nodes of each class matches that of the real ones. Synthetic nodes carry no edges (CaT's condensed graphs
    use the identity adjacency). The base snapshot is condensed at the first update (the "previous task"). The update then
    trains on the condensed memory bank only (CaT's Training in Memory), capped at cat_bank_frac * N nodes, oldest out.
    Cost charged = the real nodes condensation reads + the bank."""

    name = "cat"

    def __init__(self, max_nodes: int) -> None:
        super().__init__(max_nodes)
        self.bank_x, self.bank_y = None, None
        self.cap = max(4, int(round(max_nodes * HP["cat_bank_frac"])))
        self.per_update = max(2, int(round(max_nodes * HP["cat_per_update"])))
        self._rand = []

    def _random_models(self, model, dev):
        if not self._rand:
            for k in range(HP["cat_inits"]):
                m = copy.deepcopy(model).to(dev)
                g = torch.Generator(device="cpu").manual_seed(0xCA7 + k)
                with torch.no_grad():
                    for p_ in m.parameters():
                        p_.copy_(torch.randn(p_.shape, generator=g) * 0.1)
                m.eval()
                for p_ in m.parameters():
                    p_.requires_grad_(False)
                self._rand.append(m)
        return self._rand

    def _condense(self, model, batch, budget, dev):
        """Synthetic (x, y) for the batch's roots, `budget` nodes in total, split across classes by frequency."""
        x_r, y_r = batch.x[: batch.batch_size], batch.y[: batch.batch_size]
        classes, counts = torch.unique(y_r, return_counts=True)
        per = torch.clamp((counts.float() / counts.sum() * budget).round().long(), min=1)
        xs, ys = [], []
        g = torch.Generator(device="cpu").manual_seed(int(batch.batch_size) * 7 + budget)
        for c, k in zip(classes.tolist(), per.tolist()):
            idx = (y_r == c).nonzero(as_tuple=True)[0]
            pick = idx[torch.randperm(idx.numel(), generator=g)[:k].to(idx.device)]
            xs.append(x_r[pick].detach().clone()); ys.append(torch.full((pick.numel(),), c, dtype=y_r.dtype, device=dev))
        x_s = torch.cat(xs).clone().requires_grad_(True)
        y_s = torch.cat(ys)
        empty = torch.empty((2, 0), dtype=torch.long, device=dev)
        rand = self._random_models(model, dev)
        with torch.no_grad():
            real = [m(batch.x, batch.edge_index)[0][: batch.batch_size] for m in rand]
        opt = torch.optim.Adam([x_s], lr=HP["cat_lr"])
        for _ in range(HP["cat_steps"]):
            opt.zero_grad()
            loss = 0.0
            for m, h_r in zip(rand, real):
                h_s = m(x_s, empty)[0]
                for c in classes.tolist():
                    loss = loss + (h_r[y_r == c].mean(0) - h_s[y_s == c].mean(0)).pow(2).sum()
            loss.backward()
            opt.step()
        return x_s.detach(), y_s

    def apply(self, model, optimizer, ctx):
        dev = _device(model)
        t = time.perf_counter()
        if self.bank_x is None:          # condense the base snapshot first (the previous task)
            pool = ctx.train_mask.nonzero(as_tuple=True)[0]
            g = torch.Generator().manual_seed(ctx.seed + 0xCA7)
            base = ctx.sampler.sample(pool[torch.randperm(pool.numel(), generator=g)[: self.max_nodes]],
                                      ctx.seed + 0xCA7, ctx.step).to(dev)
            self.bank_x, self.bank_y = self._condense(model, base, self.cap - self.per_update, dev)
        roots = _uniform_roots(ctx, self.max_nodes)
        batch = ctx.sampler.sample(roots, ctx.seed + ctx.step, ctx.step).to(dev)
        x_new, y_new = self._condense(model, batch, self.per_update, dev)
        self.bank_x = torch.cat([self.bank_x, x_new])[-self.cap:]
        self.bank_y = torch.cat([self.bank_y, y_new])[-self.cap:]
        condense_s = time.perf_counter() - t
        model.train()
        optimizer.zero_grad()
        empty = torch.empty((2, 0), dtype=torch.long, device=dev)
        out, _, _ = model(self.bank_x, empty)
        loss = F.cross_entropy(out, self.bank_y)
        loss.backward()
        optimizer.step()
        s_ = full_batch_stats(batch)
        return RT.RetrainInfo(step=ctx.step, loss=float(loss.item()), nodes_targeted=int(self.bank_x.shape[0]),
                              nodes_updated=int(self.bank_x.shape[0]),
                              nodes_in_backward=int(self.bank_x.shape[0]), edges_in_backward=0,
                              communities_selected=0, communities_total=0, region_sample_s=condense_s,
                              roots_drawn=int(s_.nodes_in_backward))


_ORIGINAL = getattr(R, "_introval_original_build_retrainer", R.build_retrainer)


class _OOMGuard:
    """Wraps any arm's retrainer: an update that does not fit on the GPU is recorded (OOM) and the arm stops updating,
    instead of crashing the job (ogbn-products, wide models). streams.py writes row["oom"]."""

    def __init__(self, inner, arm):
        self._inner, self._arm = inner, arm

    def __getattr__(self, k):
        return getattr(self._inner, k)

    def should_fire(self, step, drift_signal, controller):
        return False if self._arm in OOM else self._inner.should_fire(step, drift_signal, controller)

    def apply(self, model, optimizer, ctx):
        try:
            return self._inner.apply(model, optimizer, ctx)
        except torch.OutOfMemoryError:
            optimizer.zero_grad(set_to_none=True)
            OOM.add(self._arm)
            torch.cuda.empty_cache()
            return None


def build_retrainer(arm, community_backend, adaptation_cfg, num_hops=2):
    r = _build(arm, community_backend, adaptation_cfg, num_hops)
    return r if arm == "baseline" else _OOMGuard(r, arm)


def _build(arm, community_backend, adaptation_cfg, num_hops=2):
    if arm in ("global_periodic", "roland"):
        period = adaptation_cfg.periodic_period
        if period is None:
            raise ValueError(f"{arm} needs adaptation.periodic_period")
        if arm == "roland":
            return GlobalRetrainer(arm, period=period, inner=HP["roland_inner"], alpha=HP["roland_alpha"])
        return GlobalRetrainer(arm, period=period)
    if arm == "global_trigger":
        return GlobalRetrainer(arm)
    if arm == "ergnn":
        return ERGNNRetrainer(adaptation_cfg.full_retrain_nodes)
    if arm in ("ewc", "twp"):
        return EWCRetrainer(adaptation_cfg.full_retrain_nodes, twp=arm == "twp")
    if arm == "cat":
        return CaTRetrainer(adaptation_cfg.full_retrain_nodes)
    return _ORIGINAL(arm, community_backend, adaptation_cfg, num_hops)


R._introval_original_build_retrainer = _ORIGINAL
R.build_retrainer = build_retrainer


# =====================================================================================================================
# Review round. New arms only; no existing arm changes.
#   selective_logged    GIST exactly as in D1 (same draws in the same order, same update), plus |S| per repair
#   selective_topup     GIST with |S| = N: when the drifted regions and the rehearsal draw give fewer than N distinct
#                       roots, the shortfall is drawn uniformly from the remaining training nodes
#   full_isoroots       Full with as many uniform roots as GIST drew on that seed (selective_logged's mean |S|)
#   selective_exact     GIST's roots, parent graph P = the whole current graph G_t instead of a fan-out-limited batch
#   full_exact          Full's uniform roots, parent graph P = the whole current graph G_t
#   selective_periodic  GIST's roots and update, fired on Full's realised schedule (a timer) instead of the detector
# =====================================================================================================================
REVIEW_ARMS = ("selective_logged", "selective_topup", "full_isoroots", "selective_exact", "full_exact",
               "selective_periodic")


class _RegionRoots(RT.SelectiveRetrainer):
    """SelectiveRetrainer with the root draw split from the sample, so the roots can be counted, topped up to N, or
    trained on a different parent graph. With topup=False the draws and the update are those of `selective`."""

    topup = False

    def _roots(self, ctx, drifting):
        # SelectiveRetrainer._region_batch up to the sample, draw for draw: same generator, same order of draws.
        if ctx.sampler is None or ctx.train_mask is None:
            return None
        generator = torch.Generator().manual_seed(ctx.seed + ctx.step)
        n_anchor = int(round(self.max_nodes * self.anchor_fraction))
        n_target = max(1, self.max_nodes - n_anchor)
        if ctx.oracle_nodes is not None and not self.random_region:
            pool = ctx.oracle_nodes[ctx.train_mask[ctx.oracle_nodes]]
            if pool.numel() > n_target:
                pool = pool[torch.randperm(pool.numel(), generator=generator)[:n_target]]
            nodes = pool
        else:
            nodes = self.backend.community_nodes(drifting, restrict_mask=ctx.train_mask, limit=n_target,
                                                 generator=generator)
        if nodes is None:
            return None
        if n_anchor > 0:
            pool = ctx.train_mask.nonzero(as_tuple=True)[0]
            if pool.numel() > 0:
                pick = torch.randperm(pool.numel(), generator=generator)[:n_anchor]
                nodes = torch.unique(torch.cat([nodes, pool[pick]]))
        if self.topup and nodes.numel() < self.max_nodes:
            pool = ctx.train_mask.nonzero(as_tuple=True)[0]
            rest = pool[~torch.isin(pool, nodes)]
            extra = rest[torch.randperm(rest.numel(), generator=generator)[: self.max_nodes - nodes.numel()]]
            nodes = torch.unique(torch.cat([nodes, extra]))
        return nodes if nodes.numel() else None

    def _region_batch(self, ctx, drifting):
        nodes = self._roots(ctx, drifting)
        self._n_roots = 0 if nodes is None else int(nodes.numel())
        if nodes is None:
            return None
        start = time.perf_counter()
        region = ctx.sampler.sample(nodes, ctx.seed + ctx.step, ctx.step)
        self._last_region_sample_s = time.perf_counter() - start
        return region

    def apply(self, model, optimizer, ctx):
        self._n_roots = 0
        info = super().apply(model, optimizer, ctx)
        if info is not None:
            info.roots_drawn = self._n_roots
        return info


class _TopUp(_RegionRoots):
    topup = True


class _Periodic(_RegionRoots):
    def __init__(self, period, **kw):
        super().__init__(**kw)
        self.period = max(1, int(period))

    def should_fire(self, step, drift_signal, controller):
        return step > 0 and step % self.period == 0


def _exact_update(model, optimizer, ctx, roots, num_hops):
    """One update with parent graph P = the whole current graph: the live graph state the sampler draws from (current
    edges, features and labels), not a fan-out-limited batch. H = P[N_L(S; P)] is extracted on the host, GCN's edge
    coefficients are computed on P and sliced (subgraph.with_parent_edges), and only H moves to the GPU."""
    from torch_geometric.data import Data
    from gist.adaptation.subgraph import subgraph_gradient_step
    g = ctx.sampler.data
    n = int(g.num_nodes)
    parent = Data(x=g.x, y=g.y, edge_index=g.edge_index, num_nodes=n)
    parent.n_id = torch.arange(n)
    targets = torch.zeros(n, dtype=torch.bool)
    targets[roots.cpu()] = True
    out = subgraph_gradient_step(model, optimizer, parent, targets, num_hops, halo_train_mask=None, adapter=ctx.adapter)
    if out is None:
        return None
    loss, s = out
    return RT.RetrainInfo(step=ctx.step, loss=loss, nodes_targeted=s.nodes_targeted, nodes_updated=s.nodes_targeted,
                          nodes_in_backward=s.nodes_in_backward, edges_in_backward=s.edges_in_backward,
                          communities_selected=0, communities_total=0, region_sample_s=0.0,
                          roots_drawn=int(roots.numel()))


class _SelectiveExact(_RegionRoots):
    def apply(self, model, optimizer, ctx):
        if ctx.community is None:
            return None
        drifting = self.backend.drifting(ctx.community)
        roots = self._roots(ctx, drifting)
        if roots is None:
            return None
        info = _exact_update(model, optimizer, ctx, roots, self.num_hops)
        if info is not None:
            info.communities_selected = len(drifting)
            info.communities_total = int(getattr(ctx.community, "num_communities", 0))
        return info


class _FullExact(RT.FullRetrainer):
    def __init__(self, max_nodes, num_hops):
        super().__init__(supervise_halo=False, max_nodes=max_nodes)
        self.num_hops = num_hops

    def apply(self, model, optimizer, ctx):
        # FullRetrainer._global_batch's draw: the same uniform roots as `full`, without the fan-out sample.
        if self.max_nodes <= 0 or ctx.sampler is None or ctx.train_mask is None:
            return None
        pool = ctx.train_mask.nonzero(as_tuple=True)[0]
        if pool.numel() == 0:
            return None
        generator = torch.Generator().manual_seed(ctx.seed + ctx.step)
        roots = pool[torch.randperm(pool.numel(), generator=generator)[: self.max_nodes]]
        return _exact_update(model, optimizer, ctx, roots, self.num_hops)


def _region_kw(community_backend, a, num_hops):
    return dict(backend=community_backend, mode=a.selective_mode, background_weight=a.background_weight,
                num_hops=num_hops, max_nodes=a.max_retrain_nodes, anchor_fraction=a.anchor_fraction,
                halo_rehearsal=a.halo_rehearsal, random_region=False, audit_events=0)


_build_before_review = _build


def _build(arm, community_backend, adaptation_cfg, num_hops=2):
    a = adaptation_cfg
    if arm == "selective_logged":
        return _RegionRoots(**_region_kw(community_backend, a, num_hops))
    if arm == "selective_topup":
        return _TopUp(**_region_kw(community_backend, a, num_hops))
    if arm == "selective_periodic":
        if a.periodic_period is None:
            raise ValueError("selective_periodic needs adaptation.periodic_period")
        return _Periodic(a.periodic_period, **_region_kw(community_backend, a, num_hops))
    if arm == "selective_exact":
        return _SelectiveExact(**_region_kw(community_backend, a, num_hops))
    if arm == "full_exact":
        return _FullExact(a.full_retrain_nodes, num_hops)
    if arm == "full_isoroots":                 # streams.py sets full_retrain_nodes to GIST's mean |S| on the seed
        return RT.FullRetrainer(supervise_halo=a.supervise_halo, max_nodes=a.full_retrain_nodes)
    return _build_before_review(arm, community_backend, adaptation_cfg, num_hops)


# =====================================================================================================================
# Exactness ablation. New arms only; no existing arm changes.
#   selective_trunc   GIST's roots and fan-out parent as in `selective_logged`, but the update runs on the roots'
#                     (L-1)-hop subgraph: the depth condition of Eq. (3) is broken, everything else is unchanged
#   selective_nofix   GIST's roots with the whole-graph parent as in `selective_exact`, but the layers normalize on H
#                     itself instead of taking their edge coefficients from P (GCN: the third condition is broken;
#                     for backbones without `parent_edges` it equals `selective_exact`, so it is run for GCN only)
# =====================================================================================================================
EXACTNESS_ARMS = ("selective_trunc", "selective_nofix")


class _SelectiveNoFix(_RegionRoots):
    def apply(self, model, optimizer, ctx):
        from torch_geometric.data import Data
        from torch_geometric.utils import k_hop_subgraph
        from gist.adaptation.subgraph import subgraph_gradient_step
        if ctx.community is None:
            return None
        drifting = self.backend.drifting(ctx.community)
        roots = self._roots(ctx, drifting)
        if roots is None:
            return None
        g = ctx.sampler.data
        n = int(g.num_nodes)
        # H = P[N_L(S; P)] with P the whole current graph, cut out *before* the model sees any graph, so that
        # with_parent_edges computes GCN's coefficients on H's own degrees (the unfixed layer).
        subset, ei, mapping, _ = k_hop_subgraph(roots.cpu(), self.num_hops, g.edge_index, relabel_nodes=True,
                                                num_nodes=n)
        h = Data(x=g.x[subset], y=g.y[subset], edge_index=ei, num_nodes=int(subset.numel()))
        h.n_id = subset
        targets = torch.zeros(int(subset.numel()), dtype=torch.bool)
        targets[mapping] = True
        out = subgraph_gradient_step(model, optimizer, h, targets, self.num_hops, halo_train_mask=None,
                                     adapter=ctx.adapter)
        if out is None:
            return None
        loss, s = out
        return RT.RetrainInfo(step=ctx.step, loss=loss, nodes_targeted=s.nodes_targeted,
                              nodes_updated=s.nodes_targeted, nodes_in_backward=s.nodes_in_backward,
                              edges_in_backward=s.edges_in_backward, communities_selected=len(drifting),
                              communities_total=int(getattr(ctx.community, "num_communities", 0)),
                              region_sample_s=0.0, roots_drawn=int(roots.numel()))


_build_before_exactness = _build


def _build(arm, community_backend, adaptation_cfg, num_hops=2):
    if arm == "selective_trunc":
        if num_hops < 2:
            raise ValueError("selective_trunc needs num_hops >= 2")
        return _RegionRoots(**_region_kw(community_backend, adaptation_cfg, num_hops - 1))
    if arm == "selective_nofix":
        return _SelectiveNoFix(**_region_kw(community_backend, adaptation_cfg, num_hops))
    return _build_before_exactness(arm, community_backend, adaptation_cfg, num_hops)


# =====================================================================================================================
# Severity-adaptive repair scheduling. New arms only; no existing arm changes.
#   selective_adaptive  GIST's roots and exact update, scheduled by the drift severity s = D_tot - tau (in standard
#                       deviations, since D_tot is standardised):
#                       (1) budget: a repair draws a share f(s) = clip(0.5 + s / 4, 0.5, 1) of the budget N, so mild
#                           drift gets a small repair and s >= 2 the full budget;
#                       (2) shock override: if s >= SHOCK it may fire MIN_GAP steps after the last repair instead of
#                           waiting for the cooldown, and takes SHOCK_STEPS gradient steps on the same subgraph H;
#                       (3) rate limit: a token bucket (refill RATE per step, capacity CAP) pays for every repair with
#                           f(s) x steps tokens; with too few tokens the repair is deferred, and the regions that are
#                           still displaced when tokens return are the ones repaired.
#   selective_nocool    GIST's roots and update with no cooldown at all (fires on every flagged step): the reference
#                       for endless drift.
# The detector score reaches the arm through a read-only hook on CompositeDriftDetector.update (the reading is stored,
# never changed), so every other arm behaves exactly as before.
# =====================================================================================================================
ADAPTIVE_ARMS = ("selective_adaptive", "selective_nocool")
SHOCK, MIN_GAP, SHOCK_STEPS = 3.0, 10, 3
RATE, CAP = 15 / 1000, 3.0          # the default repair rate of fixed GIST (about 15 per 1,000 steps), bursts of 3
LAST_READING = {"total": float("-inf"), "tau": float("inf"), "ready": False}

from gist.drift import detector as _DET   # noqa: E402

if not getattr(_DET.CompositeDriftDetector.update, "_gist_hooked", False):
    _orig_update = _DET.CompositeDriftDetector.update

    def _hooked_update(self, *a, **k):
        reading = _orig_update(self, *a, **k)
        LAST_READING.update(total=float(reading.total), tau=float(self.cfg.total_threshold), ready=bool(self.ready))
        return reading

    _hooked_update._gist_hooked = True
    _DET.CompositeDriftDetector.update = _hooked_update


def _severity():
    r = LAST_READING
    return (r["total"] - r["tau"]) if r["ready"] else float("-inf")


class _Adaptive(_RegionRoots):
    def __init__(self, **kw):
        super().__init__(**kw)
        self.N = int(self.max_nodes)
        self.tokens = CAP
        self.plan = None

    def should_fire(self, step, drift_signal, controller):
        self.tokens = min(CAP, self.tokens + RATE)
        self.plan = None
        if not drift_signal:
            return False
        s = _severity()
        last = controller.last_retrain
        shock = s >= SHOCK
        cooled = last is None or step - last > controller.cooldown
        if not cooled and not (shock and step - last > MIN_GAP):
            return False
        frac = min(1.0, max(0.5, 0.5 + s / 4.0))
        steps = SHOCK_STEPS if shock else 1
        if self.tokens < frac * steps:
            return False                     # deferred until the bucket refills
        self.plan = (frac, steps, s)
        return True

    def apply(self, model, optimizer, ctx):
        if self.plan is None:
            return None
        frac, steps, s = self.plan
        self.max_nodes = max(1, int(round(self.N * frac)))
        info, done = None, 0
        try:
            for _ in range(steps):           # the same roots and H each time (draws seeded by seed + step)
                out = super().apply(model, optimizer, ctx)
                if out is None:
                    break
                info, done = out, done + 1
        finally:
            self.max_nodes = self.N
        if info is not None:
            self.tokens -= frac * done
            info.steps_taken = done
            info.severity = float(s)
        self.plan = None
        return info


class _NoCooldown(_RegionRoots):
    def should_fire(self, step, drift_signal, controller):
        return bool(drift_signal)


_build_before_adaptive = _build


def _build(arm, community_backend, adaptation_cfg, num_hops=2):
    if arm == "selective_adaptive":
        return _Adaptive(**_region_kw(community_backend, adaptation_cfg, num_hops))
    if arm == "selective_nocool":
        return _NoCooldown(**_region_kw(community_backend, adaptation_cfg, num_hops))
    return _build_before_adaptive(arm, community_backend, adaptation_cfg, num_hops)
