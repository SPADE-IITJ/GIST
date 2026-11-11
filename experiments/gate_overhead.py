#!/usr/bin/env python3
"""G1: is the layer's overhead affordable end to end?

Times every arm on the same plan and warm state as budget_sweep.py (matched budget, roots-only, retrain_lr 0.001),
twice: once with per-stage CUDA synchronisation (stage breakdown is trustworthy, wall-clock is inflated) and once
without (wall-clock is trustworthy, stage times are not). `baseline` and `periodic` also run with detection off,
which prices detection end to end. `periodic`'s period is matched to the events `full` realised on the same seed
and budget, as experiments/compare.py::run_seed does.

Each arm-run waits until the GPU has had no other compute process for 5 minutes, and records the card's other
processes before and after; a run that shared the card is repeated. Rows are saved after every run, and a rerun
skips rows already present, so the sweep can be paused and resumed.

    python scripts/gate_overhead.py --seeds 100 101 102 --out results/g1_overhead.json

Held-out seeds only: evaluation seeds 0-9 are refused.
"""
from __future__ import annotations

import argparse
import copy
import dataclasses
import json
import os
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import torch
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from gist.config import load_config  # noqa: E402
from gist.data.loaders import load_dataset  # noqa: E402
from gist.drift import signals  # noqa: E402
from gist.experiments.runner import run_arm, warmup_model  # noqa: E402
from gist.seeding import derive, seed_everything  # noqa: E402
from gist.stream.schedule import build_plan  # noqa: E402

EVALUATION_SEEDS = frozenset(range(10))
FREE_MINUTES = 5


# --- GPU contention -------------------------------------------------------------------------------------------

def gpu_uuid() -> str | None:
    if not torch.cuda.is_available():
        return None
    index = os.environ.get("CUDA_VISIBLE_DEVICES", "0").split(",")[0]
    out = subprocess.run(["nvidia-smi", "--query-gpu=index,uuid", "--format=csv,noheader"],
                         capture_output=True, text=True, check=True).stdout
    return dict(line.split(", ") for line in out.strip().splitlines())[index]


def others_on(uuid: str | None) -> list[dict]:
    """Compute processes on our card other than this one."""
    if uuid is None:
        return []
    out = subprocess.run(["nvidia-smi", "--query-compute-apps=gpu_uuid,pid,used_memory", "--format=csv,noheader"],
                         capture_output=True, text=True, check=True).stdout
    procs = [line.split(", ") for line in out.strip().splitlines() if line]
    return [dict(pid=int(p[1]), mem=p[2]) for p in procs if p[0] == uuid and int(p[1]) != os.getpid()]


def wait_free(uuid: str | None) -> None:
    """Return at once if the card is free; otherwise wait until it has been free for FREE_MINUTES in a row."""
    if uuid is None or not others_on(uuid):
        return
    print(f"  GPU busy ({others_on(uuid)}); waiting for {FREE_MINUTES} free minutes", flush=True)
    free = 0
    while free < FREE_MINUTES:
        time.sleep(60)
        free = 0 if others_on(uuid) else free + 1


# --- one arm-run ----------------------------------------------------------------------------------------------

def key(seed, arm, budget, anchor, sync, detection, kind="main"):
    return (seed, arm, budget, anchor, sync, detection, kind)


def row_key(r):
    return key(r["seed"], r["arm"], r["budget"], r["anchor"], r["sync_stages"], r["detection"], r["kind"])


def timed_run(base, ds, plan, warm, seed, arm, budget, anchor, sync, detection, period, kind, uuid):
    cfg = copy.deepcopy(base)
    cfg.runtime.sync_stages = sync
    cfg.drift.detection = detection
    if budget:
        cfg.adaptation.max_retrain_nodes = budget
        cfg.adaptation.full_retrain_nodes = budget   # matched budget
    if anchor is not None:
        cfg.adaptation.anchor_fraction = anchor
    if period is not None:
        cfg.adaptation.periodic_period = period
    for attempt in range(1, 4):
        wait_free(uuid)
        before = others_on(uuid)
        r = run_arm(arm, cfg, ds, plan, warm)
        after = others_on(uuid)
        if not before and not after:
            break
        print(f"  contended (before {before}, after {after}); repeating", flush=True)
    ev = r.retrain_events
    return dict(
        seed=seed, arm=arm, budget=budget, anchor=anchor, sync_stages=sync, detection=detection, period=period,
        kind=kind, attempts=attempt, gpu_others_before=before, gpu_others_after=after, contended=bool(before or after),
        stage_times_synced=bool(r.profile["stage_times_synced"]), device=r.profile["device"],
        wall_s=r.profile["wall_s"], steps=r.summary["steps"], accuracy=r.summary["accuracy"],
        events=len(ev), retrain_steps=[e.step for e in ev],
        nodes=float(np.mean([e.nodes_in_backward for e in ev])) if ev else 0.0,
        region_sample_ms=1000.0 * float(np.mean([e.region_sample_s for e in ev])) if ev else 0.0,
        latency=r.profile["latency"],
    )


# --- M5: where detection's time goes ------------------------------------------------------------------------

def profile_detect(base, ds, plan, warm, steps: int) -> dict:
    """Two 200-step baseline runs with stages synchronised: one timing each drift signal, one under torch.profiler."""
    cfg = copy.deepcopy(base)
    cfg.runtime.sync_stages = True
    cfg.stream.steps = steps
    # The first `steps` steps of the same plan; the runner replays range(plan.steps).
    plan = dataclasses.replace(
        plan, stream_edges=plan.stream_edges[:, :steps], root_nodes=plan.root_nodes[:steps],
        sample_seeds=plan.sample_seeds[:steps],
        events_by_step={s: e for s, e in plan.events_by_step.items() if s < steps},
        drift_events=[e for s in sorted(plan.events_by_step) if s < steps for e in plan.events_by_step[s]])
    cuda = torch.cuda.is_available()
    parts = {"embedding": signals.EmbeddingDrift, "prediction": signals.PredictionDrift, "attention": signals.AttentionDrift}
    originals = {n: c.update for n, c in parts.items()}
    times = {n: [] for n in parts}

    def timed(name, fn):
        def wrapper(self, *a, **k):
            if cuda:
                torch.cuda.synchronize()
            t = time.perf_counter()
            out = fn(self, *a, **k)
            if cuda:
                torch.cuda.synchronize()
            times[name].append(1000 * (time.perf_counter() - t))
            return out
        return wrapper

    def labelled(name, fn):
        def wrapper(self, *a, **k):
            with torch.profiler.record_function(f"detect.{name}"):
                return fn(self, *a, **k)
        return wrapper

    try:
        for n, c in parts.items():
            c.update = timed(n, originals[n])
        r = run_arm("baseline", cfg, ds, plan, warm)
        detect = r.profile["latency"]["detect"]
        for n, c in parts.items():
            c.update = labelled(n, originals[n])
        acts = [torch.profiler.ProfilerActivity.CPU] + ([torch.profiler.ProfilerActivity.CUDA] if cuda else [])
        with torch.profiler.profile(activities=acts) as prof:
            run_arm("baseline", cfg, ds, plan, warm)
    finally:
        for n, c in parts.items():
            c.update = originals[n]

    # Attribute every op under a detect.* label to that label, by self CPU time.
    under, counts = {}, {}
    def walk(e, label):
        under.setdefault(label, {}).setdefault(e.name, 0.0)
        under[label][e.name] += e.self_cpu_time_total / 1000.0
        counts.setdefault(label, {}).setdefault(e.name, 0)
        counts[label][e.name] += 1
        for c in e.cpu_children:
            walk(c, label)
    for e in prof.events():
        if e.name.startswith("detect.") and e.cpu_parent is None or (e.name.startswith("detect.") and not e.cpu_parent.name.startswith("detect.")):
            for c in e.cpu_children:
                walk(c, e.name)
    ops = {lab: sorted(({"op": op, "self_cpu_ms_total": round(ms, 3), "calls": counts[lab][op]}
                        for op, ms in d.items()), key=lambda x: -x["self_cpu_ms_total"])[:8] for lab, d in under.items()}
    return dict(
        steps=steps, detect_stage=detect,
        per_signal_ms={n: dict(mean=float(np.mean(v)), p50=float(np.median(v)), total=float(np.sum(v)), calls=len(v)) for n, v in times.items()},
        top_ops_under_detect=ops,
        note="Signal timings synchronise before and after each signal; detect_stage is the runner's synced stage time. "
             "Ops are attributed by self CPU time from torch.profiler; aten::_local_scalar_dense is a host read of a GPU "
             "scalar (float()/.item()), which blocks until the GPU catches up.",
    )


# --- M1-M4 --------------------------------------------------------------------------------------------------

def ms_stage(row, stage, field="p50_ms"):
    return row["latency"].get(stage, {}).get(field, 0.0)


def total_ms(row, stage):
    return 1000.0 * row["latency"].get(stage, {}).get("total_s", 0.0)


def community_ms(row):
    """Community time left on the stream thread: the on-demand stage plus, with a background refresh, its per-step
    hand-off and any wait for an unfinished partition (`community_bg`)."""
    return total_ms(row, "community") + total_ms(row, "community_bg")


def mean_sd(values):
    """Mean and SD over seeds. Infinite values (e.g. an E* that never breaks even) are counted, not averaged in."""
    values = [float(x) for x in values]
    v = np.asarray([x for x in values if np.isfinite(x)], dtype=float)
    n_inf = sum(1 for x in values if np.isinf(x))
    if not v.size:
        return dict(mean=float("inf") if n_inf else float("nan"), sd=float("nan"), n=0, n_inf=n_inf, values=values)
    return dict(mean=float(v.mean()), sd=float(v.std(ddof=1)) if v.size > 1 else 0.0, n=int(v.size), n_inf=n_inf, values=values)


def label(arm, budget, anchor, detection):
    name = arm if anchor is None else f"selective rho={anchor:g}"
    return f"{name}" + (f" N={budget}" if budget else "") + ("" if detection else " (detection off)")


def metrics(rows, budgets, anchors) -> dict:
    main = [r for r in rows if r["kind"] == "main" and not r["contended"]]
    sel = lambda **k: [r for r in main if all(r[a] == b for a, b in k.items())]
    seeds = sorted({r["seed"] for r in main})
    out = {}
    # M1: detect vs infer per step, sync-on rows with detection on.
    on = [r for r in main if r["sync_stages"] and r["detection"]]
    out["M1"] = {label(r0["arm"], r0["budget"], r0["anchor"], True): dict(
        detect_p50_ms=mean_sd([ms_stage(r, "detect") for r in g]), infer_p50_ms=mean_sd([ms_stage(r, "infer") for r in g]),
        detect_total_s=mean_sd([total_ms(r, "detect") / 1000 for r in g]), infer_total_s=mean_sd([total_ms(r, "infer") / 1000 for r in g]),
        detect_over_infer_p50=mean_sd([ms_stage(r, "detect") / ms_stage(r, "infer") for r in g]),
        detect_over_infer_total=mean_sd([total_ms(r, "detect") / total_ms(r, "infer") for r in g]))
        for r0 in on for g in [sel(arm=r0["arm"], budget=r0["budget"], anchor=r0["anchor"], sync_stages=True, detection=True)]
        if r0["seed"] == seeds[0]}
    # M2: wall-clock per 1000 steps, every row.
    out["M2"] = {}
    for r0 in [r for r in main if r["seed"] == seeds[0]]:
        g = sel(arm=r0["arm"], budget=r0["budget"], anchor=r0["anchor"], sync_stages=r0["sync_stages"], detection=r0["detection"])
        out["M2"][label(r0["arm"], r0["budget"], r0["anchor"], r0["detection"]) + (" [sync on]" if r0["sync_stages"] else " [sync off]")] = \
            mean_sd([1000.0 * r["wall_s"] / r["steps"] for r in g])
    # M3: per-event costs, sync-on rows.
    out["M3"] = {}
    for N in budgets:
        for arm, a in [("full", None), ("periodic", None)] + [("selective", a) for a in anchors]:
            g = sel(arm=arm, budget=N, anchor=a, sync_stages=True, detection=True)
            out["M3"][label(arm, N, a, True)] = dict(
                retrain_p50_ms=mean_sd([ms_stage(r, "retrain") for r in g]),
                community_ms_per_event=mean_sd([community_ms(r) / max(r["events"], 1) for r in g]),
                region_sample_ms=mean_sd([r["region_sample_ms"] for r in g]),
                events=mean_sd([r["events"] for r in g]))
    # M4: break-even event count against periodic without detection.
    out["M4"] = {}
    for N in budgets:
        for a in anchors:
            per_seed = []
            for s in seeds:
                f, x = sel(seed=s, arm="full", budget=N, sync_stages=True, detection=True), sel(seed=s, arm="selective", budget=N, anchor=a, sync_stages=True, detection=True)
                if not f or not x:
                    continue
                f, x = f[0], x[0]
                detect = total_ms(x, "detect") / x["steps"]
                saving = ms_stage(f, "retrain") - ms_stage(x, "retrain") - community_ms(x) / max(x["events"], 1)
                per_seed.append(dict(seed=s, detect_ms_per_step=detect, full_retrain_p50_ms=ms_stage(f, "retrain"),
                                     selective_retrain_p50_ms=ms_stage(x, "retrain"),
                                     community_ms_per_event=community_ms(x) / max(x["events"], 1),
                                     saving_ms_per_event=saving, E_star=(1000.0 * detect / saving) if saving > 0 else float("inf"),
                                     realised_events=x["events"]))
            out["M4"][f"rho={a:g} N={N}"] = dict(per_seed=per_seed, E_star=mean_sd([p["E_star"] for p in per_seed]),
                                                 ceiling=1000 // base_cooldown(rows), realised_events=mean_sd([p["realised_events"] for p in per_seed]))
    return out


def base_cooldown(rows):
    return rows[0].get("cooldown", 50) if rows else 50


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/arxiv.yaml")
    ap.add_argument("--steps", type=int, default=1000)
    ap.add_argument("--seeds", type=int, nargs="+", default=[100, 101, 102])
    ap.add_argument("--budgets", type=int, nargs="+", default=[2048, 8192])
    ap.add_argument("--anchors", type=float, nargs="+", default=[0.25, 0.0])
    ap.add_argument("--retrain-lr", type=float, default=0.001)
    ap.add_argument("--profile-steps", type=int, default=200, help="M5: steps in the detect-stage profile (0 to skip)")
    ap.add_argument("--repeat", action="store_true", default=True,
                    help="time one configuration twice (first seed, full, smallest N, sync off) for repeatability")
    ap.add_argument("--refresh-lag", type=int, default=0,
                    help="drift.community_refresh_lag: 0 refreshes on the event path (as G1 measured); k > 0 refreshes "
                         "in the background and installs the partition k steps after it starts")
    ap.add_argument("--set", nargs="*", default=[], metavar="KEY=VALUE",
                    help="extra config overrides, e.g. model.backbone=sage drift.total_threshold=2.1 (values parsed as YAML)")
    ap.add_argument("--out", default="results/g1_overhead.json")
    args = ap.parse_args()

    clash = sorted(set(args.seeds) & EVALUATION_SEEDS)
    if clash:
        ap.error(f"seeds {clash} are the evaluation seeds (0-9); this is a measurement on held-out seeds")

    base = load_config(args.config, **{
        "stream.steps": args.steps, "adaptation.retrain_lr": args.retrain_lr,
        "adaptation.supervise_halo": False, "adaptation.halo_rehearsal": False,   # roots only, as budget_sweep.py
        "drift.community_refresh_lag": args.refresh_lag,
        **{k: yaml.safe_load(v) for k, v in (kv.split("=", 1) for kv in args.set)},
    })
    uuid = gpu_uuid()
    out = Path(args.out)
    state = json.loads(out.read_text()) if out.exists() else {}
    rows = state.get("rows", [])
    done = {row_key(r) for r in rows}

    def save(extra=None):
        payload = dict(config=args.config, steps=args.steps, seeds=args.seeds, budgets=args.budgets, anchors=args.anchors,
                       retrain_lr=args.retrain_lr, supervision="roots-only", refresh_lag=args.refresh_lag, overrides=args.set, gpu_uuid=uuid,
                       gpu=torch.cuda.get_device_name(0) if uuid else "cpu", cooldown=base.adaptation.cooldown,
                       rows=rows, **(state.get("extra", {}) if extra is None else {}), **(extra or {}))
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(payload, indent=2))

    def do(ctx, seed, arm, budget, anchor, sync, detection, period=None, kind="main"):
        k = key(seed, arm, budget, anchor, sync, detection, kind)
        if k in done:
            return next(r for r in rows if row_key(r) == k)
        t = time.perf_counter()
        r = timed_run(*ctx, seed, arm, budget, anchor, sync, detection, period, kind, uuid)
        r["cooldown"] = base.adaptation.cooldown
        rows.append(r); done.add(k); save()
        print(f"  [{label(arm, budget, anchor, detection):<34} sync={'on ' if sync else 'off'}] wall {r['wall_s']:7.1f}s  "
              f"acc {r['accuracy']:.4f}  events {r['events']:>2}  detect p50 {ms_stage(r, 'detect'):6.2f} ms  "
              f"retrain p50 {ms_stage(r, 'retrain'):7.1f} ms  ({time.perf_counter() - t:.0f}s)", flush=True)
        return r

    ctx_first = None
    for seed in args.seeds:
        print(f"=== seed {seed} ===", flush=True)
        seed_everything(seed)
        ds = load_dataset(base.data.name, base.data.root, base.data.train_fraction, split_seed=derive(seed, 0x51))
        plan = build_plan(ds.data.edge_index, ds.num_nodes, seed=seed, steps=args.steps, batch_size=base.stream.batch_size,
                          base_edge_fraction=base.data.base_edge_fraction, drift_cfg=base.drift_process,
                          x=ds.data.x, y=ds.data.y, num_classes=ds.out_dim)
        model, _ = warmup_model(base, ds, plan, seed=derive(seed, 0x77))
        warm = {k: v.detach().clone() for k, v in model.state_dict().items()}
        ctx = (base, ds, plan, warm)
        ctx_first = ctx_first or (ctx, seed)
        for sync in (True, False):
            do(ctx, seed, "baseline", 0, None, sync, True)
            if not sync:
                do(ctx, seed, "baseline", 0, None, sync, False)
            for N in args.budgets:
                full = do(ctx, seed, "full", N, None, True, True) if sync else do(ctx, seed, "full", N, None, sync, True)
                for a in args.anchors:
                    do(ctx, seed, "selective", N, a, sync, True)
                # Matched to full's realised events on this seed and budget (the sync-on run, so every periodic row
                # of this seed and budget uses one period).
                events = next(r for r in rows if row_key(r) == key(seed, "full", N, None, True, True))["events"]
                period = max(1, args.steps // max(events, 1))
                do(ctx, seed, "periodic", N, None, sync, True, period)
                if not sync:
                    do(ctx, seed, "periodic", N, None, sync, False, period)
    if args.repeat and ctx_first:
        (ctx, seed) = ctx_first
        print("=== repeatability ===", flush=True)
        do(ctx, seed, "full", min(args.budgets), None, False, True, kind="repeat")
    extra = dict(state.get("extra", {}))
    if args.profile_steps and "M5" not in extra and ctx_first:
        print("=== M5: detect-stage profile ===", flush=True)
        wait_free(uuid)
        (ctx, seed) = ctx_first
        extra["M5"] = dict(seed=seed, gpu_others_before=others_on(uuid), **profile_detect(*ctx, args.profile_steps))
        extra["M5"]["gpu_others_after"] = others_on(uuid)
    extra["metrics"] = metrics(rows, args.budgets, args.anchors)
    state["extra"] = extra
    save(extra)
    print(f"wrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
