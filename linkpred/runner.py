"""One arm on one replayed link stream.

Each step t: score the slice's query edges with the current model on the graph visible before them, feed misses to
the per-community scan, insert the slice, and maybe update.

Arms (all start from the same warm model; all updates use the same rule, budget N and optimiser):
* ``baseline``: never updates. Its embeddings still follow the growing graph (incremental inference, fixed weights).
* ``full``: on a trigger, N edges drawn uniformly from the last W slices.
* ``selective``: on a trigger, up to 75% of N from edges touching the flagged communities, the rest uniform anchors.
* ``periodic``: full's rule on a timer with full's update count (period (T-1)//k, so exactly k updates).
* ``random_region``: selective's rule on random communities of the same total size.

An update samples the fanout computation graph of its edges' endpoints on the visible graph, with its own target
edges removed from message passing, and runs ``update_steps`` Adam steps of in-batch InfoNCE on it. Its cost is the
size of that graph (nodes, edges), analytic FLOPs, the allocator peak and wall time.
"""
from __future__ import annotations

import copy
import time
from dataclasses import asdict, dataclass, field
from typing import Optional

import numpy as np
import torch

from gist.models.registry import backbone_flops

from .lp import build_model, embed, info_nce, rank_metrics
from .sampler import TemporalSampler, pair_key
from .scan import CommunityScan
from .stream import BASE, NEVER, VAL, LinkPlan

ARMS = ("baseline", "full", "selective", "periodic", "random_region")


@dataclass
class LPConfig:
    model: str = "sage"
    hidden: int = 128
    out_dim: int = 64
    heads: int = 4
    fanout: tuple = (15, 10)
    temp: float = 0.1
    warm_iters: int = 1000
    warm_lr: float = 5e-3
    warm_batch: int = 1024
    budget: int = 1024             # N edges per update
    window: int = 100              # slices an update draws from
    update_steps: int = 5
    lr: float = 1e-3
    pool: int = 1000               # random negatives per step
    cooldown: int = 50
    warm_steps: int = 50           # no trigger before this step
    fast_hl: float = 8.0
    slow_hl: float = 300.0
    min_support: float = 20.0
    tau: float = 4.0
    region_z: float = 2.0
    region_cap: float = 0.15
    anchor: float = 0.25
    recent: int = 100              # an edge is "recent drift" if its community drifted within this many steps
    groups_new: int = 50           # temporal: an endpoint is "new" if it arrived within this many steps

    def to_dict(self):
        return asdict(self)


def _gen(device, *parts) -> torch.Generator:
    g = torch.Generator(device=device)
    g.manual_seed(int(np.random.default_rng(list(parts)).integers(2 ** 62)))
    return g


def warmup(cfg: LPConfig, plan: LinkPlan, sampler: TemporalSampler, x: torch.Tensor, seed: int, device,
           checkpoints=(), on_checkpoint=None):
    """Train on G0's base edges (key BASE) with G0 minus the validation edges as the message-passing graph."""
    torch.manual_seed(seed)
    model = build_model(cfg.model, x.shape[1], cfg.hidden, cfg.out_dim, cfg.heads).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=cfg.warm_lr)
    lo, hi = plan.edges.to(device)
    base = torch.nonzero(plan.key.to(device) == BASE).view(-1)
    n = plan.num_nodes
    checkpoints = set(checkpoints)
    for it in range(1, cfg.warm_iters + 1):
        g = _gen(device, seed, 0x3A, it)
        pick = base[torch.randint(len(base), (min(cfg.warm_batch, len(base)),), device=device, generator=g)]
        u, v = lo[pick], hi[pick]
        roots, inv = torch.unique(torch.cat([u, v]), return_inverse=True)
        sub = sampler.sample(roots, VAL, cfg.fanout, g, exclude=pair_key(u, v, n))
        model.train()
        opt.zero_grad()
        z = embed(model, x, sub)
        loss = info_nce(z[inv[: len(u)]], z[inv[len(u):]], cfg.temp)
        loss.backward()
        opt.step()
        if it in checkpoints and on_checkpoint is not None:
            on_checkpoint(it, model)
    return model


@torch.no_grad()
def score_edges(model, cfg, plan, sampler, x, u, v, pool, kv, g):
    roots, inv = torch.unique(torch.cat([u, v, pool]), return_inverse=True)
    sub = sampler.sample(roots, kv, cfg.fanout, g)
    model.eval()
    z = embed(model, x, sub)
    B = len(u)
    return rank_metrics(z[inv[:B]], z[inv[B:2 * B]], z[inv[2 * B:]])


def validation_mrr(model, cfg, plan, sampler, x, seed, device, max_edges=2000):
    """MRR of the held-out G0 edges (key VAL), on G0 without them."""
    lo, hi = plan.edges.to(device)
    val = torch.nonzero(plan.key.to(device) == VAL).view(-1)
    g = _gen(device, seed, 0x7B)
    if len(val) > max_edges:
        val = val[torch.randperm(len(val), device=device, generator=g)[:max_edges]]
    present = torch.nonzero(sampler.visible_degree(torch.arange(plan.num_nodes, device=device), VAL) > 0).view(-1)
    pool = present[torch.randint(len(present), (min(cfg.pool, len(present)),), device=device, generator=g)]
    rr, hit = score_edges(model, cfg, plan, sampler, x, lo[val], hi[val], pool, VAL, g)
    return float(rr.mean()), float(hit.mean())


def _region(scan: CommunityScan, sizes: np.ndarray, cfg: LPConfig, n: int) -> list:
    z = scan.scores()
    if not z.size or not np.isfinite(z).any():
        return []
    order = np.argsort(-z)
    out, tot = [], 0
    for c in order:
        if c >= len(sizes) or (out and z[c] <= cfg.region_z):
            break
        if out and tot + sizes[c] > cfg.region_cap * n:
            break
        out.append(int(c))
        tot += int(sizes[c])
    return out


def _window_d(plan: LinkPlan, t: int, width: int, dev) -> torch.Tensor:
    """``plan.window`` on the device, with the sorted keys cached once per plan."""
    if "sk_d" not in plan.meta:
        plan.meta["sk_d"] = plan.key[plan.stream_ids].to(dev)
        plan.meta["sid_d"] = plan.stream_ids.to(dev)
    sk = plan.meta["sk_d"]
    lo = int(torch.searchsorted(sk, torch.tensor(2 * (t - width + 1) - 1, device=dev)))
    hi = int(torch.searchsorted(sk, torch.tensor(2 * t + 1, device=dev)))
    return plan.meta["sid_d"][lo:hi]


def run_arm(arm: str, cfg: LPConfig, plan: LinkPlan, sampler: TemporalSampler, x: torch.Tensor, warm_state: dict,
            device, period: Optional[int] = None, k_events: Optional[int] = None) -> dict:
    assert arm in ARMS
    dev = torch.device(device)
    model = build_model(cfg.model, x.shape[1], cfg.hidden, cfg.out_dim, cfg.heads).to(dev)
    model.load_state_dict(warm_state)
    opt = torch.optim.Adam(model.parameters(), lr=cfg.lr)
    scan = CommunityScan(cfg.fast_hl, cfg.slow_hl, cfg.min_support)
    n, T = plan.num_nodes, plan.steps
    lo_d, hi_d = plan.edges.to(dev)
    lo_c, hi_c = plan.edges
    cid = plan.cid.numpy()
    sizes = np.bincount(cid[cid >= 0], minlength=plan.num_comms)
    ds = plan.drift_step
    ever = ds < NEVER
    fk = plan.first_key.numpy()
    present_order = plan.meta["present_order"].to(dev)
    in_dim, hid = x.shape[1], cfg.hidden
    gh = hid // cfg.heads if cfg.model == "gat" else hid

    groups = ("all", "drifted", "recent", "redirected", "clean", "new", "old")
    sums = {k: [0.0, 0.0, 0] for k in groups}           # rr, hits, count over t >= warm_steps
    step_mrr = np.full(T, np.nan)
    maxz = np.zeros(T)
    updates, flags = [], []
    last, last_flag = -10 ** 9, -10 ** 9
    for t in range(T):
        ids = plan.scored[t]
        if len(ids):
            g = _gen(dev, plan.seed, 0x51, t)
            cnt = plan.present_count(t)
            pool = present_order[torch.randint(cnt, (min(cfg.pool, cnt),), device=dev, generator=g)]
            u, v = lo_d[ids], hi_d[ids]
            rr, hit = score_edges(model, cfg, plan, sampler, x, u, v, pool, 2 * t, g)
            rr, hit = rr.cpu().numpy(), hit.cpu().numpy()
            uc, vc = lo_c[ids].numpy(), hi_c[ids].numpy()
            miss = (hit < 0.5).astype(float)
            scan.update(np.r_[miss, miss], np.r_[cid[uc], cid[vc]])
            step_mrr[t] = rr.mean()
            if t >= cfg.warm_steps:
                cu, cv = cid[uc], cid[vc]
                dsu = np.where(cu >= 0, ds[np.maximum(cu, 0)], NEVER)
                dsv = np.where(cv >= 0, ds[np.maximum(cv, 0)], NEVER)
                masks = {"all": np.ones(len(rr), bool)}
                if not plan.temporal:
                    masks["drifted"] = (dsu <= t) | (dsv <= t)
                    masks["recent"] = ((dsu <= t) & (dsu > t - cfg.recent)) | ((dsv <= t) & (dsv > t - cfg.recent))
                    masks["redirected"] = plan.redirected[ids].numpy()
                    eu = np.where(cu >= 0, ever[np.maximum(cu, 0)], False)
                    ev = np.where(cv >= 0, ever[np.maximum(cv, 0)], False)
                    masks["clean"] = ~eu & ~ev
                else:
                    newb = 2 * (t - cfg.groups_new)
                    nw = (fk[uc] >= newb) | (fk[vc] >= newb)
                    masks["new"], masks["old"] = nw, ~nw
                for k_, mk in masks.items():
                    sums[k_][0] += float(rr[mk].sum())
                    sums[k_][1] += float(hit[mk].sum())
                    sums[k_][2] += int(mk.sum())
        mz = scan.max_z()
        maxz[t] = mz
        flagged = t >= cfg.warm_steps and mz > cfg.tau
        if flagged and t - last_flag > cfg.cooldown:
            last_flag = t
            flags.append(dict(step=t, z=mz, region=_region(scan, sizes, cfg, n)[:20]))
        if arm == "baseline":
            fire = False
        elif arm == "periodic":
            fire = period is not None and t > 0 and t % period == 0 and t // period <= (k_events or 0)
        else:
            fire = flagged and t - last > cfg.cooldown
        if not fire:
            continue
        last = t
        upd = _update(arm, cfg, plan, sampler, x, model, opt, scan, sizes, t, lo_d, hi_d, cid, ds, dev, in_dim, gh)
        updates.append(upd)

    def mean(k_):
        s = sums[k_]
        return dict(mrr=s[0] / s[2] if s[2] else None, hits20=s[1] / s[2] if s[2] else None, n=s[2])

    return dict(arm=arm, seed=plan.seed, groups={k_: mean(k_) for k_ in groups if sums[k_][2]},
                step_mrr=[None if np.isnan(a) else round(float(a), 5) for a in step_mrr],
                max_z=[round(float(a), 3) for a in maxz], flags=flags, updates=updates, events=len(updates))


def _update(arm, cfg, plan, sampler, x, model, opt, scan, sizes, t, lo_d, hi_d, cid, ds, dev, in_dim, gh):
    """Choose the update's edges (on the device: the window can hold millions), then train on them."""
    n = plan.num_nodes
    g_pick = _gen(dev, plan.seed, 0x77, t, ARMS.index(arm))
    t_sel = time.perf_counter()
    win = _window_d(plan, t, cfg.window, dev)
    N = min(cfg.budget, len(win))
    region, reg_share = [], None
    if arm in ("selective", "random_region"):
        sel = _region(scan, sizes, cfg, n)
        if arm == "random_region":
            want = int(sizes[sel].sum()) if sel else 0
            live = torch.nonzero(torch.from_numpy(sizes > 0)).view(-1)
            perm = live[torch.randperm(len(live), generator=_gen("cpu", plan.seed, 0x78, t))].tolist()
            region, tot = [], 0
            for c in perm:
                if tot >= want:
                    break
                region.append(int(c))
                tot += int(sizes[c])
        else:
            region = sel
        cid_d = plan.meta.setdefault("cid_d", plan.cid.to(dev))
        R = torch.tensor(region, dtype=torch.long, device=dev)
        rmask = torch.isin(cid_d[lo_d[win]], R) | torch.isin(cid_d[hi_d[win]], R)
        cand = torch.nonzero(rmask).view(-1)
        n_reg = min(int(round((1 - cfg.anchor) * N)), len(cand))
        pick_r = cand[torch.randperm(len(cand), device=dev, generator=g_pick)[:n_reg]]
        rest = torch.ones(len(win), dtype=torch.bool, device=dev)
        rest[pick_r] = False
        rest = torch.nonzero(rest).view(-1)
        pick_a = rest[torch.randperm(len(rest), device=dev, generator=g_pick)[: N - n_reg]]
        pick = torch.cat([pick_r, pick_a])
        reg_share = n_reg / max(N, 1)
    else:
        pick = torch.randperm(len(win), device=dev, generator=g_pick)[:N]
    ids = win[pick]
    u, v = lo_d[ids], hi_d[ids]
    if dev.type == "cuda":
        torch.cuda.synchronize(dev)
    select_ms = 1000 * (time.perf_counter() - t_sel)
    roots, inv = torch.unique(torch.cat([u, v]), return_inverse=True)
    g = _gen(dev, plan.seed, 0x52, t)
    cuda = dev.type == "cuda"
    if cuda:
        torch.cuda.synchronize(dev)
    t0 = time.perf_counter()
    sub = sampler.sample(roots, 2 * t + 1, cfg.fanout, g, exclude=pair_key(u, v, n))
    if cuda:
        torch.cuda.synchronize(dev)
        base_mem = torch.cuda.memory_allocated(dev)
        torch.cuda.reset_peak_memory_stats(dev)
    t1 = time.perf_counter()
    model.train()
    for _ in range(cfg.update_steps):
        opt.zero_grad()
        z = embed(model, x, sub)
        loss = info_nce(z[inv[: len(u)]], z[inv[len(u):]], cfg.temp)
        loss.backward()
        opt.step()
    if cuda:
        torch.cuda.synchronize(dev)
    t2 = time.perf_counter()
    nodes, edges = int(len(sub.n_id)), int(sub.edge_index.shape[1])
    out = dict(step=t, select_ms=round(select_ms, 3), edges_trained=int(N), roots=int(len(roots)), nodes=nodes, mp_edges=edges,
               flops=int(backbone_flops(cfg.model, nodes, edges, in_dim, gh, cfg.out_dim, cfg.heads) * cfg.update_steps),
               sample_ms=round(1000 * (t1 - t0), 3), train_ms=round(1000 * (t2 - t1), 3),
               peak_bytes=int(torch.cuda.max_memory_allocated(dev) - base_mem) if cuda else None,
               loss=round(float(loss), 4), region=len(region), region_nodes=int(sizes[region].sum()) if region else 0,
               region_share=reg_share)
    if not plan.temporal and region:
        rec = (ds <= t) & (ds > t - cfg.recent)
        out["region_precision"] = float(sizes[region][rec[region]].sum() / max(sizes[region].sum(), 1))
        out["region_recall"] = float(sizes[region][rec[region]].sum() / max(sizes[rec].sum(), 1)) if rec.any() else None
        out["chance"] = float(sizes[rec].sum() / max(sizes.sum(), 1))
    return out
