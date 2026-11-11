"""Multi-seed, multi-arm comparison with mean +/- std reporting."""

from __future__ import annotations

import copy
import json
from dataclasses import asdict
from pathlib import Path
from typing import Dict, List

import numpy as np

from ..config import Config
from ..data.loaders import load_dataset
from ..evaluation.metrics import aggregate_over_seeds, mean_std
from ..experiments.runner import ArmResult, run_arm, warmup_model
from ..seeding import derive, seed_everything
from ..stream.schedule import build_plan

REPORT_KEYS = [
    "accuracy",
    "f1",
    "precision",
    "recall",
    "final_accuracy",
    "drift_flags",
    "community_flags",
    "retrain_events",
    "nodes_targeted_total",
    "nodes_updated_total",
    "nodes_in_backward_total",
    "edges_in_backward_total",
    "sampler_rebuilds",
    "mean_communities_selected",
    "mean_communities_total",
    "mean_drift_total",
    "recovery_mean_steps",
    "recovery_rate",
    "drift_events_injected",
]


def run_seed(cfg: Config, seed: int, verbose: bool = True) -> Dict[str, ArmResult]:
    """Run every arm against one shared stream plan."""
    seed_everything(seed)

    dataset = load_dataset(
        cfg.data.name,
        cfg.data.root,
        cfg.data.train_fraction,
        split_seed=derive(seed, 0x51),
    )
    plan = build_plan(
        dataset.data.edge_index,
        dataset.num_nodes,
        seed=seed,
        steps=cfg.stream.steps,
        batch_size=cfg.stream.batch_size,
        base_edge_fraction=cfg.data.base_edge_fraction,
        drift_cfg=cfg.drift_process,
        x=dataset.data.x,
        y=dataset.data.y,
        num_classes=dataset.out_dim,
    )

    # One warmup, shared by every arm -> identical initial parameters.
    model, warm_losses = warmup_model(cfg, dataset, plan, seed=derive(seed, 0x77))
    warm_state = {k: v.detach().clone() for k, v in model.state_dict().items()}

    if verbose:
        print(f"  plan: {plan.describe()}")
        print(f"  warmup losses: {[round(x, 4) for x in warm_losses]}")

    # `periodic` runs last so its schedule can be matched to the realised
    # retraining budget of the drift-triggered arms on this very seed.
    ordered = sorted(cfg.experiment.arms, key=lambda a: a == "periodic")

    results: Dict[str, ArmResult] = {}
    for arm in ordered:
        arm_cfg = cfg
        if arm == "periodic" and cfg.adaptation.periodic_period is None:
            reference = results.get("full") or results.get("selective")
            events = len(reference.retrain_events) if reference else 0
            arm_cfg = copy.deepcopy(cfg)
            arm_cfg.adaptation.periodic_period = max(1, plan.steps // max(events, 1))
            if verbose:
                print(
                    f"  periodic budget-matched to {events} events "
                    f"-> period {arm_cfg.adaptation.periodic_period}"
                )

        result = run_arm(arm, arm_cfg, dataset, plan, warm_state)
        results[arm] = result
        if verbose:
            s = result.summary
            print(
                f"  [{arm:>9}] acc={s['accuracy']:.4f} f1={s['f1']:.4f} "
                f"retrains={s['retrain_events']:>3} drift_flags={s['drift_flags']:>4}"
            )
    return results


def run_comparison(cfg: Config, verbose: bool = True) -> Dict:
    per_seed: Dict[str, List[Dict]] = {arm: [] for arm in cfg.experiment.arms}
    timings: Dict[str, List[Dict]] = {arm: [] for arm in cfg.experiment.arms}
    profiles: Dict[str, List[Dict]] = {arm: [] for arm in cfg.experiment.arms}

    for seed in cfg.experiment.seeds:
        if verbose:
            print(f"\n=== seed {seed} ===")
        results = run_seed(cfg, seed, verbose=verbose)
        for arm, result in results.items():
            per_seed[arm].append(result.summary)
            timings[arm].append(result.timing)
            profiles[arm].append(result.profile)

    aggregated = {
        arm: aggregate_over_seeds(rows, REPORT_KEYS) for arm, rows in per_seed.items()
    }

    # Paired per-seed deltas against the baseline: with a shared plan the arms
    # are paired by construction, so the paired difference is the right
    # statistic and is far tighter than comparing two independent means.
    deltas: Dict[str, Dict[str, Dict[str, float]]] = {}
    if "baseline" in per_seed:
        base_rows = per_seed["baseline"]
        for arm, rows in per_seed.items():
            if arm == "baseline":
                continue
            deltas[arm] = {
                "accuracy_delta_pp": mean_std(
                    [100.0 * (r["accuracy"] - b["accuracy"]) for r, b in zip(rows, base_rows)]
                ),
                "accuracy_rel_pct": mean_std(
                    [
                        100.0 * (r["accuracy"] - b["accuracy"]) / b["accuracy"]
                        for r, b in zip(rows, base_rows)
                        if b["accuracy"] > 0
                    ]
                ),
                "f1_delta_pp": mean_std(
                    [100.0 * (r["f1"] - b["f1"]) for r, b in zip(rows, base_rows)]
                ),
            }

    return {
        "config": cfg.to_dict(),
        "per_seed": per_seed,
        "profile": {arm: _mean_profile(rows) for arm, rows in profiles.items()},
        "aggregated": aggregated,
        "deltas_vs_baseline": deltas,
        "timing": {
            arm: _mean_timing(rows) for arm, rows in timings.items()
        },
    }


def _mean_profile(rows: List[Dict]) -> Dict:
    """Average the HPC profile across seeds.

    Latency is summarised with tails as well as means: a streaming system is
    judged on its p99, and means hide exactly the retraining spikes this work
    is about.
    """
    rows = [r for r in rows if r]
    if not rows:
        return {}

    stages = sorted({s for r in rows for s in r.get("latency", {})})
    latency = {}
    for stage in stages:
        vals = [r["latency"][stage] for r in rows if stage in r.get("latency", {})]
        latency[stage] = {
            k: float(np.mean([v[k] for v in vals]))
            for k in ("mean_ms", "p50_ms", "p95_ms", "p99_ms", "max_ms", "total_s")
        }

    def avg(key, default=0.0):
        return float(np.mean([r.get(key, default) for r in rows]))

    return {
        "device": rows[0].get("device", "cpu"),
        "latency": latency,
        "events_per_s": avg("events_per_s"),
        "wall_s": avg("wall_s"),
        "retrain_gflops": avg("retrain_gflops"),
        "peak_activation_bytes": avg("peak_activation_bytes"),
        "memory_measured": bool(rows[0].get("memory", {}).get("measured", False)),
        "peak_bytes_mean": float(
            np.mean([r.get("memory", {}).get("peak_bytes_mean", 0.0) for r in rows])
        ),
    }


def _mean_timing(rows: List[Dict[str, Dict[str, float]]]) -> Dict[str, Dict[str, float]]:
    stages = sorted({stage for row in rows for stage in row})
    out: Dict[str, Dict[str, float]] = {}
    for stage in stages:
        means = [row[stage]["mean_ms"] for row in rows if stage in row]
        totals = [row[stage]["total_s"] for row in rows if stage in row]
        out[stage] = {
            "mean_ms": float(np.mean(means)) if means else 0.0,
            "total_s": float(np.mean(totals)) if totals else 0.0,
        }
    return out


def save_report(report: Dict, cfg: Config) -> Path:
    out_dir = Path(cfg.experiment.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / f"{cfg.experiment.tag}.json"
    path.write_text(json.dumps(report, indent=2, default=str))
    return path


def format_table(report: Dict) -> str:
    """Render the headline comparison as a fixed-width table."""
    agg = report["aggregated"]
    arms = list(agg.keys())
    lines: List[str] = []

    header = f"{'Metric':<26}" + "".join(f"{a:>22}" for a in arms)
    lines.append(header)
    lines.append("-" * len(header))

    for key in ["accuracy", "f1", "precision", "recall", "final_accuracy", "mean_drift_total"]:
        row = f"{key:<26}"
        for arm in arms:
            m = agg[arm][key]
            row += f"{m['mean']:>15.4f} ± {m['std']:.4f}"
        lines.append(row)

    for key in [
        "drift_flags",
        "community_flags",
        "retrain_events",
        "nodes_targeted_total",
        "nodes_updated_total",
        "mean_communities_selected",
        "mean_communities_total",
    ]:
        row = f"{key:<26}"
        for arm in arms:
            m = agg[arm][key]
            row += f"{m['mean']:>15.1f} ± {m['std']:.1f}"
        lines.append(row)

    deltas = report.get("deltas_vs_baseline", {})
    if deltas:
        lines.append("")
        lines.append("Paired delta vs baseline (per-seed, same stream plan)")
        lines.append("-" * 58)
        for arm, d in deltas.items():
            acc = d["accuracy_delta_pp"]
            rel = d["accuracy_rel_pct"]
            lines.append(
                f"  {arm:<12} accuracy {acc['mean']:+.3f} ± {acc['std']:.3f} pp"
                f"   ({rel['mean']:+.2f} ± {rel['std']:.2f} %)"
            )

    timing = report.get("timing", {})
    if timing:
        lines.append("")
        lines.append("Mean per-call stage time (ms) [preliminary - Phase 3 does this properly]")
        lines.append("-" * 72)
        stages = sorted({s for arm in timing.values() for s in arm})
        lines.append(f"{'stage':<16}" + "".join(f"{a:>14}" for a in timing))
        for stage in stages:
            row = f"{stage:<16}"
            for arm in timing:
                row += f"{timing[arm].get(stage, {}).get('mean_ms', 0.0):>14.2f}"
            lines.append(row)

    return "\n".join(lines)
