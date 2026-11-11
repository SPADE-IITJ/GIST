"""Calibration on seeds 100-102 only, written to experiments/calibration/ (prelim's files are read, never written).

    python -m experiments.calibrate probe --model gcn --variant elliptic      # warmup epochs, where prelim has none
    python -m experiments.calibrate threshold --model gat --variant cora_label  # p95 of the null D_total

`probe` is prelim's probe with its output redirected. `threshold` is prelim's calibrate on a stream variant: the
never-retraining baseline's D_total over seeds 100-102, 95th percentile.
"""
from __future__ import annotations

import argparse
import types

import numpy as np

from . import common as C

P, R = C.P, C.R


def probe(model, variant, seed=100):
    C.guard([seed], C.CAL_SEEDS)
    if variant != C.base_of(variant):
        raise SystemExit("probe the base dataset; variants share its warmup")
    orig = P.cal_path
    P.cal_path = lambda m, d: C.CAL / f"{m}_{d}.json"
    try:
        P.probe(types.SimpleNamespace(model=model, dataset=variant, seed=seed))
    finally:
        P.cal_path = orig


def threshold(model, variant, seeds=C.CAL_SEEDS, steps=None):
    C.guard(seeds, C.CAL_SEEDS)
    cal = C.calibration(model, variant)
    if "warmup_epochs" not in cal:
        raise SystemExit(f"{model}/{variant}: no warmup epochs yet (probe {C.base_of(variant)} first)")
    over = {"model.warmup_epochs": cal["warmup_epochs"]}
    if steps:
        over["stream.steps"] = steps
    cfg = C.config(model, variant, **over)
    totals = []
    for seed in seeds:
        ds, plan = C.stream(cfg, variant, seed)
        warm = C.warm_state(cfg, ds, plan, seed, key=variant.split("_")[0] if variant.endswith(("_label", "_struct")) else variant)
        r = R.run_arm("baseline", cfg, ds, plan, warm)
        totals.extend(s.drift["total"] for s in r.steps[cfg.drift.warmup:])
        print(f"  {model}/{variant} seed {seed}: {len(r.steps)} steps", flush=True)
    totals = np.asarray(totals, dtype=float)
    thr = float(np.percentile(totals[np.isfinite(totals)], 95))
    if steps:
        print(f"  (smoke: threshold {thr:.3f} not saved)")
        return thr
    C.save_calibration(model, variant, warmup_epochs=cal["warmup_epochs"], total_threshold=thr,
                       calibration_seeds=list(seeds), null_flag_rate_at_threshold=float((totals > thr).mean()))
    print(f"  -> {model}/{variant} total_threshold {thr:.3f}", flush=True)
    return thr


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("what", choices=["probe", "threshold"])
    ap.add_argument("--model", required=True, choices=C.MODELS)
    ap.add_argument("--variant", required=True, choices=list(C.VARIANTS))
    ap.add_argument("--steps", type=int, default=None, help="smoke only")
    a = ap.parse_args()
    if a.what == "probe":
        probe(a.model, a.variant)
    else:
        threshold(a.model, a.variant, steps=a.steps)


if __name__ == "__main__":
    main()
