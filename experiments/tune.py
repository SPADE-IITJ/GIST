"""Choose the baselines' hyperparameters on the calibration seeds (100-102) only.

    python -m experiments.tune select [--exp TUNE]

Reads results/streams/<exp>/<model>_arxiv_year<tag>.json, one tag per grid value (make_jobs.py builds the cells),
and, for each method, keeps the value with the highest mean accuracy over models x seeds (ties: the first in grid
order). Writes calibration/arms_hp.json (or results/smoke/arms_hp.json under GIST_SMOKE), which every reported
cell loads.
"""
from __future__ import annotations

import argparse
import json

import numpy as np

from . import common as C

# EWC / TWP: the first grid (1e2-1e4, CGLB's scale) left the penalty inert at our gradient scale (all three values
# scored the same as `full`, 2026-10-22), so it was widened to 1e5-1e9. Old tags stay on disk.
GRID = {
    "ewc": [{"ewc_reg": v} for v in (1e5, 1e6, 1e7, 1e8, 1e9)],
    "twp": [{"twp_lambda_l": v, "twp_lambda_t": v} for v in (1e5, 1e6, 1e7, 1e8, 1e9)],
    "roland": [{"roland_alpha": v} for v in (0.5, 0.9)],
}
VARIANT, MODELS = "arxiv_year", ("gat", "sage")


def tag(method, hp):
    return "_" + method + "_" + "_".join(f"{v:g}" for v in hp.values())


def select(exp="TUNE"):
    chosen, scores = {}, {}
    for method, grid in GRID.items():
        best = None
        for hp in grid:
            accs = []
            for m in MODELS:
                d = C.read_json(C.RESULTS / "streams" / exp / f"{m}_{VARIANT}{tag(method, hp)}.json", {})
                accs += [r["accuracy"] for r in d.get("rows", []) if r["arm"] == method]
            s = float(np.mean(accs)) if accs else float("nan")
            scores[tag(method, hp)] = dict(mean_accuracy=s, runs=len(accs))
            if accs and (best is None or s > best[0]):
                best = (s, hp)
        if best is None:
            raise SystemExit(f"no tuning results for {method} under {exp}")
        chosen.update(best[1])
    out = dict(hp=chosen, scores=scores, variant=VARIANT, models=list(MODELS), seeds=list(C.CAL_SEEDS),
               grid={k: v for k, v in GRID.items()})
    C.write_json(C.ARMS_HP, out)
    print(json.dumps(chosen), "->", C.ARMS_HP)


def drop(exp, arms, variants):
    """Remove `arms` rows from reported cells so the same command reruns only them (backup first)."""
    import shutil
    n = 0
    for v in variants:
        for m in C.MODELS:
            f = C.RESULTS / "streams" / exp / f"{m}_{v}.json"
            d = C.read_json(f)
            if not d:
                continue
            bak = C.RESULTS / "streams" / f"{exp}_pre_retune" / f.name
            bak.parent.mkdir(parents=True, exist_ok=True)
            if not bak.exists():
                shutil.copy2(f, bak)
            keep = [r for r in d["rows"] if r["arm"] not in arms]
            n += len(d["rows"]) - len(keep)
            d["rows"] = keep
            C.write_json(f, d)
    print(f"dropped {n} rows of {arms} from {exp} {variants} (backups in {exp}_pre_retune/)")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("what", choices=["select", "drop"])
    ap.add_argument("--exp", default="TUNE")
    ap.add_argument("--arms", nargs="*", default=[])
    ap.add_argument("--variants", nargs="*", default=[])
    a = ap.parse_args()
    select(a.exp) if a.what == "select" else drop(a.exp, a.arms, a.variants)


if __name__ == "__main__":
    main()
