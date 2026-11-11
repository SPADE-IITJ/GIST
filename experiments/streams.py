"""Streaming cells for E2 (repair vs frozen exact embeddings), E3b (parameter-updating baselines) and E5 (pitfalls).

    python -m experiments.streams --exp E2 --model gat --variant cora_label --arms baseline full selective periodic \
        global_periodic global_trigger [--seeds 110 ...] [--over model.lr=0.01 ...] [--tag _lr01] [--steps 200]

One cell = one model x stream variant x (experiment, tag); rows are appended per (seed, arm) and the file is
rewritten after each, so a stopped job resumes where it was. The pipeline is the prelim campaign's (same config,
plan, warmup, trigger and record), with experiments's arms installed. Timer arms (periodic, global_periodic, roland) fire
as often as `full` did on that seed, as in prelim. Besides the prelim record, each seed stores the per-step count of
embeddings an exact incremental-inference system (InkStream / Ripple / StreamTGN) would recompute (E2).
"""
from __future__ import annotations

import argparse
import ast
import copy
import time

import numpy as np
import torch

from . import common as C

C.install_arms()
from . import arms as A  # noqa: E402

P, R = C.P, C.R
TIMER_ARMS = ("periodic", "global_periodic", "roland", "selective_periodic")


def parse_over(items):
    out = {}
    for it in items or []:
        k, v = it.split("=", 1)
        try:
            out[k] = ast.literal_eval(v)
        except (ValueError, SyntaxError):
            out[k] = v
    return out


def recompute_volume(ds, plan, num_hops=2):
    """Per step: nodes whose L-layer embedding changes, i.e. those within L hops (on the graph after the step) of an
    endpoint of an arriving or drift-added edge, or of a node whose features changed. Label events change no input,
    so they add nothing: an exact incremental system recomputes nothing and its model's predictions do not move."""
    from torch_geometric.utils import k_hop_subgraph
    from gist.stream.graph_state import GraphState
    g = GraphState(ds.num_nodes, plan.base_edges)
    counts, label_only = [], []
    for step in range(plan.steps):
        seeds = [plan.edges_at(step).reshape(-1)]
        g.add_edges(plan.edges_at(step))
        has_label = False
        for ev in plan.drift_at(step):
            if ev.kind == "structural":
                g.add_edges(ev.edges)
                seeds.append(ev.edges.reshape(-1))
            elif ev.kind == "feature":
                seeds.append(ev.nodes.long())
            elif ev.kind == "label":
                has_label = True
        s = torch.unique(torch.cat(seeds).long())
        sub, _, _, _ = k_hop_subgraph(s, num_hops, g.edge_index, num_nodes=ds.num_nodes)
        counts.append(int(sub.numel()))
        label_only.append(has_label)
    return counts, label_only


class StepSplit:
    """Per-step holdout accuracy on drifted vs clean nodes, recorded alongside run_arm.

    Wraps runner.infer and runner.classification_metrics for one run_arm call. "Drifted" = hit by any injected drift
    event up to this step; on temporal (natural-drift) streams, = arrived within the last `recent` steps. Records, per
    step, [correct_drifted, n_drifted, correct_clean, n_clean] over the step's holdout roots."""

    def __init__(self, ds, plan, recent=200):
        self.eval_mask = ds.eval_mask
        self.n = ds.num_nodes
        self.arrival = getattr(plan, "arrival_step", None)
        self.recent = recent
        # by step only: two events can share a step, and comparing their node tensors fails (ties keep plan order)
        self.events = sorted(((int(e.step), e.nodes.long()) for e in plan.drift_events
                              if getattr(e, "nodes", None) is not None), key=lambda t: t[0])
        self.rows, self._step, self._hit = [], -1, torch.zeros(self.n, dtype=torch.bool)
        self.k = int(ds.out_dim)
        self.conf = torch.zeros(self.k * self.k, dtype=torch.long)   # holdout [true][pred], all steps (macro-F1)
        self._k = 0

    def __enter__(self):
        self._infer, self._metrics = R.infer, R.classification_metrics
        split = self

        def infer(model, batch):
            out = split._infer(model, batch)
            split._step += 1
            split._roots = batch.n_id[: batch.batch_size].cpu()
            split._preds = out[0].argmax(1)[: batch.batch_size].cpu()
            split._y = batch.y[: batch.batch_size].cpu()
            return out

        R.infer = infer
        return self

    def record(self):
        """Called after each step's inference: classify this step's holdout roots as drifted / clean."""
        t = self._step
        while self._k < len(self.events) and self.events[self._k][0] <= t:
            self._hit[self.events[self._k][1]] = True
            self._k += 1
        roots = self._roots
        ev = self.eval_mask[roots]
        if self.arrival is not None:
            drifted = (self.arrival[roots] > t - self.recent) & (self.arrival[roots] <= t)
        else:
            drifted = self._hit[roots]
        ok = self._preds == self._y
        d, c = ev & drifted, ev & ~drifted
        lab = ev & (self._y >= 0) & (self._y < self.k)
        self.conf += torch.bincount(self._y[lab] * self.k + self._preds[lab], minlength=self.k * self.k)
        self.rows.append([int(ok[d].sum()), int(d.sum()), int(ok[c].sum()), int(c.sum())])

    def __exit__(self, *exc):
        R.infer = self._infer


def run_arm_split(arm, cfg, ds, plan, warm):
    """run_arm, plus the per-step drifted / clean split (recorded through the evaluate stage's metrics call)."""
    split = StepSplit(ds, plan)
    orig_metrics = R.classification_metrics

    def metrics(preds, labels):
        split.record()
        return orig_metrics(preds, labels)

    with split:
        R.classification_metrics = metrics
        try:
            r = R.run_arm(arm, cfg, ds, plan, warm)
        finally:
            R.classification_metrics = orig_metrics
    r.confusion = split.conf.view(split.k, split.k).tolist()
    return r, split.rows


def run_cell(exp, model, variant, arms, seeds, over=None, tag="", steps=None, hp=None, calibration_ok=False, audit=0):
    C.guard(seeds, C.CAL_SEEDS if calibration_ok else None)
    if not calibration_ok and set(seeds) & set(C.CAL_SEEDS):
        raise SystemExit("reported cells must not reuse calibration seeds 100-102")
    chosen = C.read_json(C.ARMS_HP, {}) if not calibration_ok else {}
    A.HP.update(chosen.get("hp", {}))     # tuned on seeds 100-102; reported cells use them
    if hp:
        A.HP.update(hp)
    over = dict(over or {})
    if steps:
        over["stream.steps"] = steps
    base = C.run_config(model, variant, **over)
    out = C.RESULTS / "streams" / exp / f"{model}_{variant}{tag}.json"
    state = C.read_json(out, {})
    rows, meta = state.get("rows", []), state.get("meta", {})
    done = {(r["seed"], r["arm"]) for r in rows}
    hpd = dict(model=model, variant=variant, exp=exp, tag=tag, over={k: str(v) for k, v in over.items()},
               steps=base.stream.steps, budget=base.adaptation.max_retrain_nodes,
               warmup_epochs=base.model.warmup_epochs, total_threshold=base.drift.total_threshold,
               retrain_lr=base.adaptation.retrain_lr, model_lr=base.model.lr, arm_hp=dict(A.HP))

    def save():
        C.write_json(out, dict(hyperparameters=hpd, seeds=list(seeds), arms=list(arms), meta=meta, rows=rows))

    for seed in seeds:
        if all((seed, a) in done for a in arms):
            continue
        ds, plan = C.stream(base, variant, seed)
        if str(seed) not in meta:
            info = dict(drift_steps=[int(s) for s in plan.drift_steps], num_nodes=ds.num_nodes,
                        kinds=sorted({e.kind for e in plan.drift_events}))
            if exp == "E2":
                vol, lab = recompute_volume(ds, plan)
                info.update(recompute=vol, label_steps=[i for i, x in enumerate(lab) if x])
            meta[str(seed)] = info
        warm = C.warm_state(base, ds, plan, seed, key=variant.split("_")[0] if variant.endswith(("_label", "_struct")) else variant)
        for arm in arms:
            if (seed, arm) in done:
                continue
            cfg = copy.deepcopy(base)
            period = None
            if arm == "iso_full":                             # prelim's C3 control: full's roots at selective's cost
                sel = next((r for r in rows if r["seed"] == seed and r["arm"] == "selective"), None)
                if sel is None:
                    raise SystemExit(f"iso_full needs the selective arm first on seed {seed}")
                cfg.adaptation.iso_target_nodes = int(round(sel["nodes"]))
            if arm == "full_isoroots":                        # review: Full with as many uniform roots as GIST drew
                sel = next((r for r in rows if r["seed"] == seed and r["arm"] == "selective_logged"), None)
                if sel is None:
                    raise SystemExit(f"full_isoroots needs selective_logged first on seed {seed}")
                cfg.adaptation.full_retrain_nodes = max(1, int(round(sel["roots_drawn"])))
            if arm == "selective" and audit and seed == seeds[0]:
                cfg.adaptation.exactness_audit = audit          # C4: first `audit` selective updates of the first seed
            if arm in TIMER_ARMS:
                full = next((r for r in rows if r["seed"] == seed and r["arm"] == "full"), None)
                if full is None:
                    raise SystemExit(f"{arm} needs the full arm first on seed {seed} (put 'full' earlier in --arms)")
                period = max(1, base.stream.steps // max(full["events"], 1))
                cfg.adaptation.periodic_period = period
            t = time.perf_counter()
            A.OOM.discard(arm)
            r, split_rows = run_arm_split(arm, cfg, ds, plan, warm)
            row = P.record(r, arm, seed, plan, period)
            row["split"] = split_rows                          # per step: [ok_drifted, n_drifted, ok_clean, n_clean]
            row["drift_total"] = [None if not np.isfinite(st.drift["total"]) else round(float(st.drift["total"]), 4)
                                  for st in r.steps]           # detector score per step (ROC / delay, appendix)
            row["oom"] = arm in A.OOM            # its whole-graph step did not fit on the GPU: it never updated
            row["roots_per_event"] = [int(e.roots_drawn) for e in r.retrain_events]   # |S| of each repair (review)
            row["steps_per_event"] = [int(getattr(e, "steps_taken", 1)) for e in r.retrain_events]   # drift-range experiments
            row["severity_per_event"] = [round(float(getattr(e, "severity", float("nan"))), 3) for e in r.retrain_events]
            row["confusion"] = getattr(r, "confusion", None)     # holdout [true][pred] counts over the stream
            A.OOM.discard(arm)
            torch.cuda.empty_cache() if torch.cuda.is_available() else None
            rows.append(row)
            done.add((seed, arm))
            save()
            print(f"  {exp} {model}/{variant}{tag} seed {seed} [{arm:<15}] acc {row['accuracy']:.4f} "
                  f"events {row['events']:>2} nodes {row['nodes']:>9.0f} ({time.perf_counter() - t:.0f}s)", flush=True)
    save()
    print(f"wrote {out}")
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--exp", required=True)
    ap.add_argument("--model", required=True, choices=C.MODELS)
    ap.add_argument("--variant", required=True, choices=list(C.VARIANTS))
    ap.add_argument("--arms", nargs="+", required=True)
    ap.add_argument("--seeds", type=int, nargs="+", default=list(C.REPORTED_SEEDS))
    ap.add_argument("--over", nargs="*", default=[])
    ap.add_argument("--hp", nargs="*", default=[], help="arm hyperparameters, e.g. ewc_reg=100")
    ap.add_argument("--tag", default="")
    ap.add_argument("--steps", type=int, default=None, help="smoke only: shorter stream")
    ap.add_argument("--calibration", action="store_true", help="allow seeds 100-102 (tuning cells)")
    ap.add_argument("--audit", type=int, default=0, help="C4: float64 audits of the first seed's first N selective updates")
    a = ap.parse_args()
    run_cell(a.exp, a.model, a.variant, a.arms, a.seeds, parse_over(a.over), a.tag, a.steps,
             parse_over(a.hp), a.calibration, a.audit)


if __name__ == "__main__":
    main()
