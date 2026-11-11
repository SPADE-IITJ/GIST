#!/usr/bin/env python3
"""linkpred driver.

    run.py prep      --graphs G ...                 # parse + cache the edge lists, print sizes
    run.py probe     --model M --graph G            # warmup length, by validation MRR on G0 (seeds 100-102)
    run.py calibrate --model M --graph G            # tau: 99.5th pct of the scan's max z on the null twin
    run.py run       --model M --graph G [--seeds]  # all arms, real stream
    run.py audit     --model M --graph G            # C4 exactness of 3 updates (float64, CPU)

Seeds: 100-102 calibration (default). 110-114 only with --reported. 0-9 refused.
"""
from __future__ import annotations

import argparse
import dataclasses
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))

from linkpred import data as D  # noqa: E402
from linkpred.audit import audit_update  # noqa: E402
from linkpred.features import structural_features  # noqa: E402
from linkpred.runner import ARMS, LPConfig, run_arm, validation_mrr, warmup  # noqa: E402
from linkpred.sampler import TemporalSampler, pair_key  # noqa: E402
from linkpred.stream import build_plan  # noqa: E402

CAL, RES, WARM = HERE / "calibration", HERE / "results", HERE / "data" / "cache" / "warm"
CAL_SEEDS = (100, 101, 102)
CHECKPOINTS = (250, 500, 1000, 2000, 4000)


def guard(seeds, reported=False):
    ok = set(CAL_SEEDS) | (set(range(110, 115)) if reported else set())
    bad = [s for s in seeds if s not in ok]
    if bad:
        raise SystemExit(f"seeds {bad} not allowed (calibration 100-102; 110-114 need --reported; 0-9 never)")


def config(model: str, graph: dict, **kw) -> LPConfig:
    n = graph["num_nodes"]
    cfg = LPConfig(model=model, budget=int(min(2048, max(64, n // 40))))
    cal = CAL / f"{model}_{graph['name']}.json"
    if cal.exists():
        c = json.loads(cal.read_text())
        if "warm_iters" in c:
            cfg.warm_iters = int(c["warm_iters"])
        if "tau" in c:
            cfg.tau = float(c["tau"])
    return dataclasses.replace(cfg, **kw)


def save_cal(model, name, **fields):
    CAL.mkdir(exist_ok=True)
    p = CAL / f"{model}_{name}.json"
    c = json.loads(p.read_text()) if p.exists() else {}
    c.update(fields)
    p.write_text(json.dumps(c, indent=1))


def graph(name):
    g = D.load(name)
    g["name"] = name
    return g


def plan_for(g, seed, null=False, steps=1000):
    t = g["time"]
    return build_plan(g["edges"], g["num_nodes"], seed=seed, steps=steps, time=t,
                      shuffle_time=null and t is not None, inject=not null)


def warm_state(cfg, plan, sampler, x, seed, dev, name):
    WARM.mkdir(parents=True, exist_ok=True)
    p = WARM / f"{cfg.model}_{name}_s{seed}_i{cfg.warm_iters}.pt"
    if p.exists():
        return torch.load(p, map_location=dev)
    model = warmup(cfg, plan, sampler, x, seed, dev)
    state = {k: v.detach().clone() for k, v in model.state_dict().items()}
    torch.save(state, p)
    return state


def device():
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def cmd_prep(a):
    for name in a.graphs:
        t0 = time.time()
        g = graph(name)
        m = g["edges"].shape[1]
        span = "" if g["time"] is None else f" time {g['time'].min():.0f}..{g['time'].max():.0f}"
        print(f"{name:14} n {g['num_nodes']:>9,} m {m:>11,}{span}  ({time.time() - t0:.0f}s)", flush=True)


def cmd_probe(a):
    guard(a.seeds)
    g, dev = graph(a.graph), device()
    curves = {}
    for seed in a.seeds:
        plan = plan_for(g, seed)
        sampler = TemporalSampler(plan, dev)
        x = structural_features(plan, dev)
        cfg = config(a.model, g, warm_iters=max(CHECKPOINTS))
        curve = {}

        def on_ck(it, model):
            curve[it] = validation_mrr(model, cfg, plan, sampler, x, seed, dev)[0]
            print(f"  {a.model}/{a.graph} seed {seed} iters {it}: val MRR {curve[it]:.4f}", flush=True)
        warmup(cfg, plan, sampler, x, seed, dev, checkpoints=CHECKPOINTS, on_checkpoint=on_ck)
        curves[seed] = curve
    mean = {it: float(np.mean([curves[s][it] for s in curves])) for it in CHECKPOINTS}
    best = max(mean.values())
    pick = min(it for it in CHECKPOINTS if mean[it] >= 0.98 * best)
    save_cal(a.model, a.graph, warm_iters=pick, probe={str(s): c for s, c in curves.items()}, probe_mean=mean)
    print(f"{a.model}/{a.graph}: warm_iters {pick} (mean val MRR {mean[pick]:.4f}, best {best:.4f})")


def cmd_calibrate(a):
    guard(a.seeds)
    g, dev = graph(a.graph), device()
    zs, flags, steps = [], 0, 0
    for seed in a.seeds:
        null = plan_for(g, seed, null=True)
        sampler = TemporalSampler(null, dev)
        x = structural_features(null, dev)
        cfg = config(a.model, g)
        state = warm_state(cfg, null, sampler, x, seed, dev, a.graph)
        r = run_arm("baseline", cfg, null, sampler, x, state, dev)
        z = np.asarray(r["max_z"])[cfg.warm_steps:]
        zs.append(z)
        print(f"  {a.model}/{a.graph} seed {seed}: null max z p50 {np.median(z):.2f} p99.5 {np.quantile(z, 0.995):.2f}", flush=True)
    allz = np.concatenate(zs)
    tau = float(np.quantile(allz, 0.995))
    save_cal(a.model, a.graph, tau=tau, null_flag_rate=float((allz > tau).mean()))
    print(f"{a.model}/{a.graph}: tau {tau:.3f}")


def cmd_run(a):
    guard(a.seeds, a.reported)
    g, dev = graph(a.graph), device()
    out_p = RES / f"{a.model}_{a.graph}{a.tag}.json"
    RES.mkdir(exist_ok=True)
    res = json.loads(out_p.read_text()) if out_p.exists() else dict(model=a.model, graph=a.graph, family=g["family"], seeds={})
    for seed in a.seeds:
        if str(seed) in res["seeds"] and set(res["seeds"][str(seed)]["arms"]) >= set(a.arms):
            continue
        t0 = time.time()
        plan = plan_for(g, seed)
        sampler = TemporalSampler(plan, dev)
        x = structural_features(plan, dev)
        cfg = config(a.model, g)
        state = warm_state(cfg, plan, sampler, x, seed, dev, a.graph)
        res["cfg"] = cfg.to_dict()
        res["stream"] = dict(num_nodes=plan.num_nodes, num_edges=int(plan.edges.shape[1]), steps=plan.steps,
                             scored_mean=float(np.mean([len(s) for s in plan.scored])), num_comms=plan.num_comms,
                             temporal=plan.temporal)
        arms = {}
        for arm in [x_ for x_ in ARMS if x_ in a.arms and x_ != "periodic"]:
            arms[arm] = run_arm(arm, cfg, plan, sampler, x, state, dev)
            _say(a, seed, arm, arms[arm])
        if "periodic" in a.arms:
            k = max(arms.get("full", {}).get("events", 0), 1)
            arms["periodic"] = run_arm("periodic", cfg, plan, sampler, x, state, dev, period=max(1, (plan.steps - 1) // k), k_events=k)
            _say(a, seed, "periodic", arms["periodic"])
        res["seeds"][str(seed)] = dict(arms=arms, drift_events=[dataclasses.asdict(e) for e in plan.events],
                                       wall_s=round(time.time() - t0, 1))
        out_p.write_text(json.dumps(res))
    print(f"wrote {out_p}")


def _say(a, seed, arm, r):
    gr = r["groups"]
    s = " ".join(f"{k} {v['mrr']:.4f}" for k, v in gr.items() if v["mrr"] is not None)
    cost = ""
    if r["updates"]:
        cost = f"  nodes/upd {np.mean([u['nodes'] for u in r['updates']]):.0f} train_ms {np.mean([u['train_ms'] for u in r['updates']]):.1f}"
    print(f"  {a.model}/{a.graph} seed {seed} {arm:13} MRR {s}  upd {r['events']}{cost}", flush=True)


def cmd_audit(a):
    g = graph(a.graph)
    dev = device()
    seed = a.seeds[0]                      # default CAL_SEEDS[0] = 100; the review audits seed 0 via experiments.final
    plan = plan_for(g, seed)
    sampler = TemporalSampler(plan, dev)
    x = structural_features(plan, dev)
    cfg = config(a.model, g)
    state = warm_state(cfg, plan, sampler, x, seed, dev, a.graph)
    from linkpred.lp import build_model
    model = build_model(cfg.model, x.shape[1], cfg.hidden, cfg.out_dim, cfg.heads)
    model.load_state_dict({k: v.cpu() for k, v in state.items()})
    rng = np.random.default_rng([seed, 0xAD])
    cid = plan.cid.numpy()
    out = []
    for t in (plan.steps // 4, plan.steps // 2, 3 * plan.steps // 4):
        win = plan.window(t, cfg.window).numpy()
        lo, hi = plan.edges[0][win].numpy(), plan.edges[1][win].numpy()
        c = cid[lo[rng.integers(len(win))]]
        cand = np.nonzero((cid[lo] == c) | (cid[hi] == c))[0]
        N = min(cfg.budget, a.max_edges, len(win))
        pick = rng.choice(cand, min(N, len(cand)), replace=False) if len(cand) else rng.choice(len(win), N, replace=False)
        ids = torch.from_numpy(win[pick]).long()
        u, v = plan.edges[0][ids], plan.edges[1][ids]
        ei = sampler.visible_edge_index(2 * t + 1, exclude=pair_key(u, v, plan.num_nodes).to(sampler.device)).cpu()
        r = audit_update(model, cfg.model, x.cpu(), ei, plan.num_nodes, u, v, cfg.temp)
        r["step"] = t
        out.append(r)
        print(f"  audit {a.model}/{a.graph} t {t}: 2-hop {r['rel_2hop']:.2e}  1-hop {r['rel_1hop']:.2e}  roots {r['roots']} sub {r['sub_nodes_2hop']}", flush=True)
    RES.mkdir(exist_ok=True)
    (RES / (f"audit_{a.model}_{a.graph}.json" if seed == 100 else f"audit_{a.model}_{a.graph}_s{seed}.json")).write_text(json.dumps(out, indent=1))


def cmd_cell(a):
    """probe -> calibrate -> run -> audit for one model x graph, skipping the parts already on disk (resumable)."""
    cal = CAL / f"{a.model}_{a.graph}.json"
    c = json.loads(cal.read_text()) if cal.exists() else {}
    if "warm_iters" not in c:
        cmd_probe(a)
        c = json.loads(cal.read_text())
    if "tau" not in c:
        cmd_calibrate(a)
    cmd_run(a)
    if not (RES / f"audit_{a.model}_{a.graph}.json").exists():
        cmd_audit(a)
    print(f"CELL DONE {a.model}/{a.graph}", flush=True)


def main():
    p = argparse.ArgumentParser()
    sp = p.add_subparsers(dest="cmd", required=True)
    q = sp.add_parser("prep")
    q.add_argument("--graphs", nargs="+", default=list(D.GRAPHS))
    for name in ("probe", "calibrate", "run", "audit", "cell"):
        q = sp.add_parser(name)
        q.add_argument("--model", required=True, choices=["sage", "gcn", "gat", "gin"])
        q.add_argument("--graph", required=True, choices=list(D.GRAPHS))
        q.add_argument("--seeds", type=int, nargs="+", default=list(CAL_SEEDS))
        if name in ("run", "cell"):
            q.add_argument("--tag", default="")
            q.add_argument("--arms", nargs="+", default=list(ARMS))
            q.add_argument("--reported", action="store_true")
        if name in ("audit", "cell"):
            q.add_argument("--max-edges", type=int, default=512)
    a = p.parse_args()
    dict(prep=cmd_prep, probe=cmd_probe, calibrate=cmd_calibrate, run=cmd_run, audit=cmd_audit, cell=cmd_cell)[a.cmd](a)


if __name__ == "__main__":
    main()
