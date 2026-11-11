#!/usr/bin/env python3
"""Preliminary claims campaign: one driver for the warmup probe, the drift-threshold calibration and the main runs.

    prelim.py probe     --model gat --dataset cora               # seed 100: warmup plateau per model x dataset
    prelim.py calibrate --model gat --dataset cora               # seeds 100-102: p95 of the null D_total
    prelim.py run       --model gat --dataset cora --seeds 110 111 112 113 114

Calibration (probe, calibrate) runs on seeds 100-102 only. The reported runs use 110-114. Seeds 0-9 are refused
everywhere: they are the paper's evaluation seeds. The design was fixed before any evaluation run.
"""
from __future__ import annotations

import argparse
import copy
import json
import os
import sys
import time
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from gist.config import load_config  # noqa: E402
from gist.data.loaders import load_dataset  # noqa: E402
from gist.evaluation.profiling import resolve_device  # noqa: E402
from gist.experiments.runner import _train_pool, run_arm, sample_batch, warmup_edges, warmup_model  # noqa: E402
from gist.models.registry import BACKBONES  # noqa: E402
from gist.seeding import derive, seed_everything  # noqa: E402
from gist.stream.schedule import build_plan  # noqa: E402
from gist.stream.temporal import build_temporal_plan  # noqa: E402

EVALUATION_SEEDS = frozenset(range(10))
CALIBRATION_SEEDS = (100, 101, 102)
PRELIM = ROOT / "prelim"
CALIBRATION = PRELIM / "configs" / "calibration"     # one JSON per model x dataset (jobs run in parallel)
DATA_ROOT = str(ROOT / "data")

# Per dataset: base config, data name, stream length, matched budget N, warmup epochs probed.
DATASETS = {
    "arxiv_year": dict(config="configs/arxiv.yaml", name="ogbn-arxiv", temporal=True, steps=1000, budget=8192, epochs=[50, 100, 200, 400]),
    "arxiv": dict(config="configs/arxiv.yaml", name="ogbn-arxiv", temporal=False, steps=1000, budget=8192, epochs=[50, 100, 200, 400]),
    "cora": dict(config="configs/default.yaml", name="Cora", temporal=False, steps=2000, budget=128, epochs=[5, 20, 50, 100, 200, 400]),
    "pubmed": dict(config="configs/default.yaml", name="PubMed", temporal=False, steps=2000, budget=128, epochs=[5, 20, 50, 100, 200, 400]),
    "photo": dict(config="configs/default.yaml", name="Photo", temporal=False, steps=2000, budget=128, epochs=[5, 20, 50, 100, 200, 400]),
    # Added for the C2/C3 pilot in place of Photo: a temporal graph with real drift. Budget N = 2048 roots
    # keeps N at ~8 % of the trainable pool, as on the other graphs (Cora 128 / 1,625; Elliptic 2,048 / ~27,900).
    "elliptic": dict(config="configs/elliptic.yaml", name="elliptic", temporal=True, steps=1000, budget=2048, epochs=[50, 100, 200, 400]),
}
# The same for every model and dataset, fixed before calibration.
COMMON = {
    "adaptation.retrain_lr": 0.001,
    "adaptation.supervise_halo": False,      # roots only
    "adaptation.halo_rehearsal": False,
    "adaptation.anchor_fraction": 0.25,      # rho, pre-registered after G4
    "model.warmup_symmetric": True,          # ver-15 fix
    "drift.community_refresh_lag": 20,       # ver-15 fix
    "runtime.device": "auto",
}
ARMS = ["baseline", "full", "selective", "periodic", "random_region", "iso_full"]


def guard(seeds, allowed=None):
    bad = sorted(set(seeds) & EVALUATION_SEEDS)
    if bad:
        raise SystemExit(f"seeds {bad} are the paper's evaluation seeds (0-9); this campaign never runs them")
    if allowed is not None and not set(seeds) <= set(allowed):
        raise SystemExit(f"calibration uses seeds {list(allowed)} only")


def config(model: str, dataset: str, **extra):
    spec = DATASETS[dataset]
    over = {**COMMON, "data.name": spec["name"], "data.root": DATA_ROOT, "model.backbone": model,
            "stream.steps": spec["steps"], "adaptation.max_retrain_nodes": spec["budget"],
            "adaptation.full_retrain_nodes": spec["budget"], **extra}
    return load_config(str(ROOT / spec["config"]), **over)


def cal_path(model: str, dataset: str) -> Path:
    return CALIBRATION / f"{model}_{dataset}.json"


def calibrated(model: str, dataset: str) -> dict:
    p = cal_path(model, dataset)
    cal = json.loads(p.read_text()) if p.exists() else {}
    if "total_threshold" not in cal:
        raise SystemExit(f"{model}/{dataset} is not calibrated yet (run probe and calibrate on seeds 100-102)")
    return cal


def save_calibration(model: str, dataset: str, **fields):
    p = cal_path(model, dataset)
    cal = json.loads(p.read_text()) if p.exists() else {}
    cal.update(fields)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(cal, indent=2, sort_keys=True))


def stream(cfg, dataset: str, seed: int):
    seed_everything(seed)
    ds = load_dataset(cfg.data.name, cfg.data.root, cfg.data.train_fraction, split_seed=derive(seed, 0x51))
    if DATASETS[dataset]["temporal"]:
        plan = build_temporal_plan(ds.data.edge_index, ds.data.node_year, ds.num_nodes, seed=seed,
                                   steps=cfg.stream.steps, batch_size=cfg.stream.batch_size)
    else:
        plan = build_plan(ds.data.edge_index, ds.num_nodes, seed=seed, steps=cfg.stream.steps,
                          batch_size=cfg.stream.batch_size, base_edge_fraction=cfg.data.base_edge_fraction,
                          drift_cfg=cfg.drift_process, x=ds.data.x, y=ds.data.y, num_classes=ds.out_dim)
    return ds, plan


# --- probe ------------------------------------------------------------------------------------------------------

def probe(args):
    guard([args.seed], CALIBRATION_SEEDS)
    spec = DATASETS[args.dataset]
    cfg = config(args.model, args.dataset, **{"model.warmup_epochs": max(spec["epochs"])})
    ds, plan = stream(cfg, args.dataset, args.seed)
    dev = resolve_device(cfg.runtime.device)
    data = copy.copy(ds.data)
    data.edge_index = warmup_edges(cfg, ds.num_nodes, plan)
    data.n_id = torch.arange(data.num_nodes)
    # Holdout nodes present before the stream starts (all of them for injected drift), a seeded random 20,000.
    present = _train_pool(torch.ones(ds.num_nodes, dtype=torch.bool), plan, -1)
    ev = (ds.eval_mask & present).nonzero(as_tuple=True)[0]
    if ev.numel() > 20000:
        ev = ev[torch.randperm(ev.numel(), generator=torch.Generator().manual_seed(12345))[:20000].sort().values]
    batch = sample_batch(data, ev, cfg.stream.num_neighbors, seed=12345).to(dev)
    scores = {}

    @torch.no_grad()
    def on_epoch(epoch, model):
        if epoch in spec["epochs"]:
            model.eval()
            out = model(batch.x, batch.edge_index)[0][: batch.batch_size]
            scores[epoch] = float((out.argmax(1) == batch.y[: batch.batch_size]).float().mean())
            model.train()
            print(f"  {args.model}/{args.dataset} epochs={epoch:>4} holdout {scores[epoch]:.4f}", flush=True)

    t = time.perf_counter()
    warmup_model(cfg, ds, plan, seed=derive(args.seed, 0x77), on_epoch=on_epoch)
    best = max(scores, key=lambda e: (scores[e], -e))
    save_calibration(args.model, args.dataset, warmup_epochs=best, probe={str(k): v for k, v in scores.items()},
                     probe_seed=args.seed, probe_eval_nodes=int(ev.numel()))
    print(f"  -> warmup_epochs {best} ({time.perf_counter() - t:.0f}s)", flush=True)


# --- calibrate --------------------------------------------------------------------------------------------------

def calibrate(args):
    guard(args.seeds, CALIBRATION_SEEDS)
    cal = json.loads(cal_path(args.model, args.dataset).read_text())
    cfg = config(args.model, args.dataset, **{"model.warmup_epochs": cal["warmup_epochs"]})
    totals = []
    for seed in args.seeds:
        ds, plan = stream(cfg, args.dataset, seed)
        model, _ = warmup_model(cfg, ds, plan, seed=derive(seed, 0x77))
        warm = {k: v.detach().clone() for k, v in model.state_dict().items()}
        r = run_arm("baseline", cfg, ds, plan, warm)   # never retrains: the null distribution of D_total
        totals.extend(s.drift["total"] for s in r.steps[cfg.drift.warmup:])
        print(f"  {args.model}/{args.dataset} seed {seed}: {len(r.steps)} steps", flush=True)
    totals = np.asarray(totals, dtype=float)
    thr = float(np.percentile(totals[np.isfinite(totals)], 95))
    save_calibration(args.model, args.dataset, total_threshold=thr, calibration_seeds=list(args.seeds),
                     null_flag_rate_at_threshold=float((totals > thr).mean()))
    print(f"  -> total_threshold {thr:.3f}", flush=True)


# --- run --------------------------------------------------------------------------------------------------------

def record(r, arm, seed, plan, period=None, target=None):
    ev = r.retrain_events
    lat = r.profile["latency"]
    mem = r.profile.get("retrain_memory", {})
    mean = lambda xs: float(np.mean(xs)) if xs else 0.0
    return dict(
        seed=seed, arm=arm, period=period, iso_target_nodes=target,
        accuracy=r.summary["accuracy"], final_accuracy=r.summary["final_accuracy"],
        recovery_mean_steps=r.summary["recovery_mean_steps"], recovery_rate=r.summary["recovery_rate"],
        drift_flags=r.summary["drift_flags"], flag_steps=[s.step for s in r.steps if s.drift["flag"]],
        events=len(ev), retrain_steps=[e.step for e in ev],
        nodes=mean([e.nodes_in_backward for e in ev]), edges=mean([e.edges_in_backward for e in ev]),
        roots_drawn=mean([e.roots_drawn for e in ev]), communities_selected=mean([e.communities_selected for e in ev]),
        gflops_per_event=r.profile["retrain_gflops"] / max(len(ev), 1), peak_activation_mb=r.profile["peak_activation_bytes"] / 2**20,
        retrain_p50_ms=lat.get("retrain", {}).get("p50_ms", 0.0), retrain_p95_ms=lat.get("retrain", {}).get("p95_ms", 0.0),
        region_sample_ms=1000 * mean([e.region_sample_s for e in ev]),
        retrain_peak_mb=mem.get("peak_bytes_mean", 0.0) / 2**20, retrain_delta_mb=mem.get("delta_bytes_mean", 0.0) / 2**20,
        audits=[e.audit for e in ev if e.audit is not None],
        step_accuracy=[float("nan") if s.accuracy is None else s.accuracy for s in r.steps],
        eval_support=float(np.mean([s.eval_support for s in r.steps])), wall_s=r.profile["wall_s"],
    )


def run(args):
    guard(args.seeds)
    if set(args.seeds) & set(CALIBRATION_SEEDS):
        raise SystemExit("reported runs must not reuse the calibration seeds 100-102")
    cal = calibrated(args.model, args.dataset)
    extra = {"stream.steps": args.steps} if args.steps else {}
    base = config(args.model, args.dataset, **{"model.warmup_epochs": cal["warmup_epochs"],
                                               "drift.total_threshold": cal["total_threshold"],
                                               "runtime.sync_stages": True, "runtime.measure_retrain_memory": True, **extra})
    out = Path(args.out) if args.out else PRELIM / "results" / "runs" / f"{args.model}_{args.dataset}.json"
    state = json.loads(out.read_text()) if out.exists() else {}
    rows = state.get("rows", [])
    done = {(r["seed"], r["arm"]) for r in rows}
    meta = state.get("meta", {})

    def save():
        out.parent.mkdir(parents=True, exist_ok=True)
        hp = dict(model=args.model, dataset=args.dataset, steps=base.stream.steps, batch_size=base.stream.batch_size,
                  budget=base.adaptation.max_retrain_nodes, hidden=base.model.hidden, heads=base.model.heads,
                  warmup_epochs=base.model.warmup_epochs, total_threshold=base.drift.total_threshold,
                  retrain_lr=base.adaptation.retrain_lr, anchor_fraction=base.adaptation.anchor_fraction,
                  cooldown=base.adaptation.cooldown, refresh_lag=base.drift.community_refresh_lag,
                  warmup_symmetric=base.model.warmup_symmetric, supervision="roots-only", sync_stages=True)
        out.write_text(json.dumps(dict(hyperparameters=hp, seeds=args.seeds, arms=ARMS, meta=meta, rows=rows)))

    for seed in args.seeds:
        if all((seed, a) in done for a in ARMS):
            continue
        ds, plan = stream(base, args.dataset, seed)
        info = dict(drift_steps=[int(s) for s in plan.drift_steps], num_nodes=ds.num_nodes)
        if DATASETS[args.dataset]["temporal"]:
            info.update(step_window=plan.step_windows(), step_year=plan.step_year.tolist())
        meta[str(seed)] = info
        model, _ = warmup_model(base, ds, plan, seed=derive(seed, 0x77))
        warm = {k: v.detach().clone() for k, v in model.state_dict().items()}
        get = lambda arm: next(r for r in rows if r["seed"] == seed and r["arm"] == arm)
        for arm in ARMS:
            if (seed, arm) in done:
                continue
            cfg = copy.deepcopy(base)
            period = target = None
            if arm == "selective" and seed == args.seeds[0]:
                cfg.adaptation.exactness_audit = 3          # C4: first 3 events of the first seed
            if arm == "periodic":                           # C2: as many updates as full made, on a timer
                period = max(1, base.stream.steps // max(get("full")["events"], 1))
                cfg.adaptation.periodic_period = period
            if arm == "iso_full":                           # C3: full's uniform roots at selective's backward size
                target = int(round(get("selective")["nodes"]))
                cfg.adaptation.iso_target_nodes = target
            t = time.perf_counter()
            r = run_arm(arm, cfg, ds, plan, warm)
            rows.append(record(r, arm, seed, plan, period, target))
            done.add((seed, arm))
            save()
            x = rows[-1]
            print(f"  {args.model}/{args.dataset} seed {seed} [{arm:<13}] acc {x['accuracy']:.4f} last-100 {x['final_accuracy']:.4f} "
                  f"events {x['events']:>2} nodes {x['nodes']:>9.0f} ({time.perf_counter() - t:.0f}s)", flush=True)
    print(f"wrote {out}")


# --- audit ------------------------------------------------------------------------------------------------------

class _AuditsDone(Exception):
    pass


def audit(args):
    """C4 re-audit: the first seed's selective arm, as in `run`, stopped after its 3rd audit.

    The audit deep-copies the model and never touches training, so this recomputes the run's audits with the
    global-scale metric without rerunning the cell. Writes prelim/results/audit/<model>_<dataset>.json.
    """
    guard(args.seeds)
    from gist.adaptation import retrainers as RT
    cal = calibrated(args.model, args.dataset)
    cfg = config(args.model, args.dataset, **{"model.warmup_epochs": cal["warmup_epochs"],
                                              "drift.total_threshold": cal["total_threshold"],
                                              "runtime.sync_stages": True, "runtime.measure_retrain_memory": True,
                                              "adaptation.exactness_audit": args.count})
    seed = args.seeds[0]
    ds, plan = stream(cfg, args.dataset, seed)
    model, _ = warmup_model(cfg, ds, plan, seed=derive(seed, 0x77))
    warm = {k: v.detach().clone() for k, v in model.state_dict().items()}
    audits, original = [], RT.audit_exactness

    def spy(*a, **k):
        audits.append(original(*a, **k))
        if len(audits) >= args.count:
            raise _AuditsDone
        return audits[-1]

    RT.audit_exactness = spy
    try:
        run_arm("selective", cfg, ds, plan, warm)
    except _AuditsDone:
        pass
    finally:
        RT.audit_exactness = original
    out = PRELIM / "results" / "audit" / f"{args.model}_{args.dataset}.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(dict(model=args.model, dataset=args.dataset, seed=seed, audits=audits), indent=1))
    for a in audits:
        print(f"  {args.model}/{args.dataset} global {a['rel_diff_global']:.2e} (per-tensor {a['rel_diff']:.2e}, "
              f"worst {a['rel_diff_worst']['tensor']} |d| {a['rel_diff_worst']['abs_diff']:.1e} of {a['rel_diff_worst']['grad_max']:.1e})"
              f"  1-hop control global {a['rel_diff_short_global']:.2e}", flush=True)
    print(f"wrote {out}")


def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    for name in ("probe", "calibrate", "run", "audit"):
        p = sub.add_parser(name)
        p.add_argument("--model", choices=BACKBONES, required=True)
        p.add_argument("--dataset", choices=list(DATASETS), required=True)
        if name == "probe":
            p.add_argument("--seed", type=int, default=100)
        elif name == "calibrate":
            p.add_argument("--seeds", type=int, nargs="+", default=list(CALIBRATION_SEEDS))
        elif name == "audit":
            p.add_argument("--seeds", type=int, nargs="+", default=[110], help="first seed is audited, as in run")
            p.add_argument("--count", type=int, default=3)
        else:
            p.add_argument("--seeds", type=int, nargs="+", default=[110, 111, 112, 113, 114])
            p.add_argument("--out", default=None, help="smoke runs only: write here instead of prelim/results/runs/")
            p.add_argument("--steps", type=int, default=None, help="smoke runs only: shorter stream")
    args = ap.parse_args()
    dict(probe=probe, calibrate=calibrate, run=run, audit=audit)[args.cmd](args)


if __name__ == "__main__":
    main()
