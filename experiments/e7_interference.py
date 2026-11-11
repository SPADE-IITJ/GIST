"""E7: what a repair update does to inference served on the same GPU (intro: the update recurs "on a live system,
alongside the inference workload").

    python -m experiments.e7_interference --model gat --variant arxiv [--seeds 110 ...] [--seconds 20]

Two processes share one L40S (each its own CUDA context, as a serving process and a repair process would):
* serving: inference on pre-sampled batches of the stream (batch size and fanout of the config), back to back;
  per-request latency (CUDA-synchronised) with its start time;
* repair: under each condition, updates back to back for the whole window:
    idle       no repair
    full       N uniform roots, the config's fanout (prelim `full`)
    selective  0.75 N roots from METIS communities + 0.25 N anchors, same fanout (GIST)
    global     the whole graph (D3-GNN / InstantGNN style)
Reports p50 / p95 / p99 inference latency per condition and the repair step time under contention. The share of
serving time repair takes at the measured trigger rate is computed by agg.py from E2's rows.
Must run alone on its card; the lane guarantees no foreign process.
"""
from __future__ import annotations

import argparse
import copy
import time

import numpy as np
import torch
import torch.multiprocessing as mp
import torch.nn.functional as F

from . import common as C

R = C.R


def _batches(cfg, ds, plan, warm, seed, n_batches=60):
    from gist.stream.graph_state import GraphState
    g = GraphState(ds.num_nodes, plan.base_edges)
    for t in range(min(500, plan.steps)):
        g.add_edges(plan.edges_at(t))
    data = copy.copy(ds.data)
    data.edge_index = g.edge_index
    serve = [R.sample_batch(data, plan.root_nodes[t], cfg.stream.num_neighbors, plan.sample_seeds[t])
             for t in range(n_batches)]
    N = cfg.adaptation.max_retrain_nodes
    gen = torch.Generator().manual_seed(seed + 0xE7)
    pool = ds.train_mask.nonzero(as_tuple=True)[0]
    full = R.sample_batch(data, pool[torch.randperm(pool.numel(), generator=gen)[:N]], cfg.stream.num_neighbors, seed)
    from gist.drift.partitioners import metis_partition
    k = int(np.clip(ds.num_nodes // 1000, 16, 256))
    cid = metis_partition(g.edge_index, ds.num_nodes, num_parts=k)
    comm = torch.randperm(k, generator=gen)
    region = torch.isin(cid, comm[: max(1, k // 20)]) & ds.train_mask
    rn = region.nonzero(as_tuple=True)[0]
    tgt = rn[torch.randperm(rn.numel(), generator=gen)[: int(0.75 * N)]]
    anc = pool[torch.randperm(pool.numel(), generator=gen)[: N - tgt.numel()]]
    sel = R.sample_batch(data, torch.unique(torch.cat([tgt, anc])), cfg.stream.num_neighbors, seed)
    return serve, dict(full=full, selective=sel, global_=data), g.edge_index


def _serve(model_args, warm, batches, start, stop, out_q, dev_id, ready):
    torch.cuda.set_device(dev_id)
    dev = torch.device("cuda", dev_id)
    model = R.build_backbone(*model_args).to(dev).eval()
    model.load_state_dict(warm)
    bs = [(b.x.to(dev), b.edge_index.to(dev)) for b in batches]
    with torch.no_grad():
        for x, ei in bs[:5]:
            model(x, ei)
    torch.cuda.synchronize(dev)
    ready.put("serve")
    start.wait()
    lat, i = [], 0
    while not stop.is_set():
        x, ei = bs[i % len(bs)]
        t = time.perf_counter()
        with torch.no_grad():
            model(x, ei)
        torch.cuda.synchronize(dev)
        lat.append((t, 1000 * (time.perf_counter() - t)))
        i += 1
    out_q.put(("serve", lat))


def _repair(model_args, warm, batch, start, stop, out_q, dev_id, lr, ready):
    torch.cuda.set_device(dev_id)
    dev = torch.device("cuda", dev_id)
    model = R.build_backbone(*model_args).to(dev).train()
    model.load_state_dict(warm)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    x, ei = batch.x.to(dev), batch.edge_index.to(dev)
    y = batch.y.to(dev)
    bs = getattr(batch, "batch_size", None) or int(batch.num_nodes)
    try:                               # one warm-up step, so the window measures steady state
        opt.zero_grad(); out, _, _ = model(x, ei); F.cross_entropy(out[:bs], y[:bs]).backward(); opt.step()
        torch.cuda.synchronize(dev)
    except torch.OutOfMemoryError:
        ready.put("repair"); start.wait(); out_q.put(("repair", "oom")); return
    ready.put("repair")
    start.wait()
    steps = []
    try:
        while not stop.is_set():
            t = time.perf_counter()
            opt.zero_grad()
            out, _, _ = model(x, ei)
            F.cross_entropy(out[:bs], y[:bs]).backward()
            opt.step()
            torch.cuda.synchronize(dev)
            steps.append(1000 * (time.perf_counter() - t))
    except torch.OutOfMemoryError:     # the update does not fit next to the serving process
        out_q.put(("repair", "oom"))
        return
    out_q.put(("repair", steps))


def run_condition(cond, model_args, warm, serve, repair_batch, seconds, lr, dev_id=0):
    ctx = mp.get_context("spawn")
    start, stop, q, ready = ctx.Event(), ctx.Event(), ctx.Queue(), ctx.Queue()
    procs = [ctx.Process(target=_serve, args=(model_args, warm, serve, start, stop, q, dev_id, ready))]
    if repair_batch is not None:
        procs.append(ctx.Process(target=_repair, args=(model_args, warm, repair_batch, start, stop, q, dev_id, lr, ready)))
    for p in procs:
        p.start()
    for _ in procs:                    # the window opens only once every process is loaded and warmed up
        ready.get(timeout=600)
    start.set()
    time.sleep(seconds)
    stop.set()
    got = dict(q.get() for _ in procs)
    for p in procs:
        p.join()
    lat = np.array([l for _, l in got["serve"]])
    r = got.get("repair", [])
    oom = r == "oom"
    r = [] if oom else r
    return dict(condition=cond, repair_oom=oom, requests=int(lat.size), p50_ms=float(np.percentile(lat, 50)),
                p95_ms=float(np.percentile(lat, 95)), p99_ms=float(np.percentile(lat, 99)), mean_ms=float(lat.mean()),
                repair_steps=len(r), repair_step_p50_ms=float(np.median(r)) if r else None,
                nodes=None if repair_batch is None else int(repair_batch.num_nodes))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True, choices=C.MODELS)
    ap.add_argument("--variant", required=True, choices=list(C.VARIANTS))
    ap.add_argument("--seeds", type=int, nargs="+", default=list(C.REPORTED_SEEDS))
    ap.add_argument("--seconds", type=float, default=20.0)
    a = ap.parse_args()
    C.guard(a.seeds)
    cfg = C.config(a.model, a.variant, **{"model.warmup_epochs": C.calibration(a.model, a.variant)["warmup_epochs"]})
    out_p = C.RESULTS / "E7" / f"{a.model}_{a.variant}.json"
    res = C.read_json(out_p, dict(model=a.model, variant=a.variant, seconds=a.seconds, seeds={}))
    lr = cfg.adaptation.retrain_lr or cfg.model.lr
    for seed in a.seeds:
        if str(seed) in res["seeds"]:
            continue
        ds, plan = C.stream(cfg, a.variant, seed)
        warm = C.warm_state(cfg, ds, plan, seed, key=a.variant)
        serve, rep, ei = _batches(cfg, ds, plan, warm, seed)
        g = rep["global_"]
        gb = copy.copy(g)
        gb.batch_size = int(g.num_nodes)
        model_args = (a.model, ds.in_dim, ds.out_dim, cfg.model.hidden, cfg.model.heads)
        rows = []
        for cond, b in (("idle", None), ("full", rep["full"]), ("selective", rep["selective"]), ("global", gb)):
            rows.append(run_condition(cond, model_args, warm, serve, b, a.seconds, lr))
            r = rows[-1]
            print(f"  E7 {a.model}/{a.variant} seed {seed} {cond:<9} infer p50 {r['p50_ms']:.2f} p99 {r['p99_ms']:.2f} ms "
                  f"({r['requests']} req)  repair step p50 {r['repair_step_p50_ms']}", flush=True)
        res["seeds"][str(seed)] = rows
        C.write_json(out_p, res)
    print(f"wrote {out_p}")


if __name__ == "__main__":
    main()
