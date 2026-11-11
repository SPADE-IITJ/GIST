#!/usr/bin/env python3
"""Tables from linkpred results: ``agg.py [TAG] [--md out.md]``.

Per graph (pooled over the models and seeds present):
* accuracy: MRR of each arm; G = full - baseline, C6 = selective - full, C2 = full - periodic,
  C3 = selective - random_region (paired by seed, in MRR points x100), drifted / clean groups;
* cost per update, selective / full: nodes, FLOPs, allocator peak, train ms;
* detection (injected drift, from the baseline arm, which never updates so its scan sees drift undisturbed):
  hit rate (event flagged within 100 steps), median delay, false flags per 1000 steps, top-region precision at the
  first flag vs chance.
"""
from __future__ import annotations

import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
RES = HERE / "results"
sys.path.insert(0, str(HERE.parent))
from linkpred.data import GRAPHS  # noqa: E402

argv = sys.argv[1:]
if "--md" in argv:
    i = argv.index("--md")
    argv = argv[:i] + argv[i + 2:]
args = [a for a in argv if not a.startswith("--")]
tag = args[0] if args else ""
md_out = sys.argv[sys.argv.index("--md") + 1] if "--md" in sys.argv else None
FAMS = ["social", "web", "p2p", "road", "citation"]


def load():
    out = defaultdict(list)
    for f in sorted(RES.glob(f"*{tag}.json")):
        if f.name.startswith("audit_"):
            continue
        r = json.loads(f.read_text())
        if tag == "" and r["graph"] + ".json" != f.name.split("_", 1)[1]:
            continue
        out[r["graph"]].append(r)
    return out


def mrr(a, g="all"):
    v = a["groups"].get(g)
    return None if v is None else v["mrr"]


def paired(rs, x, y, g="all"):
    d = []
    for r in rs:
        for s in r["seeds"].values():
            A = s["arms"]
            if x in A and y in A and mrr(A[x], g) is not None and mrr(A[y], g) is not None:
                d.append(100 * (mrr(A[x], g) - mrr(A[y], g)))
    return d


def cost(rs, key):
    num, den = [], []
    for r in rs:
        for s in r["seeds"].values():
            A = s["arms"]
            if "selective" in A and "full" in A and A["selective"]["updates"] and A["full"]["updates"]:
                vs = [u[key] for u in A["selective"]["updates"] if u.get(key) is not None]
                vf = [u[key] for u in A["full"]["updates"] if u.get(key) is not None]
                if vs and vf:
                    num.append(np.mean(vs))
                    den.append(np.mean(vf))
    return float(np.sum(num) / np.sum(den)) if den else None


def detection(rs):
    hits, delays, false, steps, prec, chance = 0, [], 0, 0, [], []
    n_ev = 0
    for r in rs:
        for s in r["seeds"].values():
            ev = s["drift_events"]
            b = s["arms"].get("baseline")
            if not ev or b is None:
                continue
            T = len(b["max_z"])
            steps += T - r["cfg"]["warm_steps"]
            fl = b["flags"]
            used = set()
            for e in ev:
                n_ev += 1
                after = [f for f in fl if e["step"] <= f["step"] < e["step"] + 100]
                if after:
                    hits += 1
                    delays.append(after[0]["step"] - e["step"])
                    used.add(after[0]["step"])
                    top = after[0]["region"][:1]
                    prec.append(float(any(c in (e["source"], e["target"]) for c in top)))
            near = lambda f: any(e["step"] <= f["step"] < e["step"] + 100 for e in ev)
            false += sum(1 for f in fl if not near(f))
            k = r["stream"]["num_comms"]
            chance.append(2 / k)
    if not n_ev:
        return None
    return dict(hit=hits / n_ev, delay=float(np.median(delays)) if delays else None,
                false_per_1000=1000 * false / max(steps, 1), top1_precision=float(np.mean(prec)) if prec else None,
                chance=float(np.mean(chance)), events=n_ev)


def fmt(x, p=2, sign=True):
    if x is None:
        return "—"
    return f"{x:+.{p}f}" if sign else f"{x:.{p}f}"


def mean_ci(d):
    if not d:
        return "—"
    d = np.asarray(d)
    se = d.std(ddof=1) / np.sqrt(len(d)) if len(d) > 1 else float("nan")
    return f"{d.mean():+.2f} ± {1.96 * se:.2f}" if len(d) > 1 else f"{d.mean():+.2f}"


rows = load()
lines = []
p = lambda s="": lines.append(s)
p(f"# linkpred results {tag or '(untagged)'}")
p()
p("MRR x100 differences, paired by seed, mean ± 95% CI over models x seeds. Cost = selective / full per update.")
p()
p("| Family | Graph | models | baseline MRR | full − base (G) | sel − full (C6) | full − periodic (C2) | sel − random (C3) | drifted: full − base | clean: sel − base | nodes | FLOPs | peak mem | train ms |")
p("|---|---|---|---|---|---|---|---|---|---|---|---|---|---|")
for fam in FAMS:
    for gname, spec in GRAPHS.items():
        if spec["family"] != fam or gname not in rows:
            continue
        rs = rows[gname]
        base = [mrr(s["arms"]["baseline"]) for r in rs for s in r["seeds"].values() if "baseline" in s["arms"]]
        temporal = rs[0]["stream"]["temporal"]
        drifted = paired(rs, "full", "baseline", "new" if temporal else "drifted")
        clean = paired(rs, "selective", "baseline", "old" if temporal else "clean")
        p(f"| {fam} | {gname} | {','.join(sorted(r['model'] for r in rs))} | {np.mean(base):.4f} | {mean_ci(paired(rs, 'full', 'baseline'))} | "
          f"{mean_ci(paired(rs, 'selective', 'full'))} | {mean_ci(paired(rs, 'full', 'periodic'))} | "
          f"{mean_ci(paired(rs, 'selective', 'random_region'))} | {mean_ci(drifted)} | {mean_ci(clean)} | "
          f"{fmt(cost(rs, 'nodes'), 3, False)} | {fmt(cost(rs, 'flops'), 3, False)} | {fmt(cost(rs, 'peak_bytes'), 3, False)} | "
          f"{fmt(cost(rs, 'train_ms'), 3, False)} |")
p()
p("(temporal graphs: 'drifted' = edges with a recently arrived endpoint, 'clean' = both endpoints old)")
p()
p("## Detection (injected drift; baseline arm's scan)")
p()
p("| Family | Graph | events | hit rate | median delay | false flags / 1000 steps | top-1 region precision | chance |")
p("|---|---|---|---|---|---|---|---|")
for fam in FAMS:
    for gname, spec in GRAPHS.items():
        if spec["family"] != fam or gname not in rows:
            continue
        d = detection(rows[gname])
        if d:
            p(f"| {fam} | {gname} | {d['events']} | {d['hit']:.2f} | {fmt(d['delay'], 0, False)} | {d['false_per_1000']:.1f} | "
              f"{fmt(d['top1_precision'], 2, False)} | {d['chance']:.3f} |")
p()
audits = sorted(RES.glob("audit_*.json"))
if audits:
    p("## Exactness audits (float64; max |Δgrad| / max |grad| over 3 updates)")
    p()
    p("| model / graph | 2-hop (exact) | 1 hop short |")
    p("|---|---|---|")
    for f in audits:
        a = json.loads(f.read_text())
        p(f"| {f.stem[6:]} | {max(x['rel_2hop'] for x in a):.1e} | {min(x['rel_1hop'] for x in a):.1e} |")
text = "\n".join(lines)
print(text)
if md_out:
    Path(md_out).write_text(text + "\n")
