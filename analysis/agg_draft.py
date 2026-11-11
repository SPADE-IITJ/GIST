"""Interim Phase D numbers for the evaluation draft (D1, D2, D4, D7). Read-only over results/*/raw; prints markdown.

    python analysis/agg_draft.py > results/draft_numbers.md

Paired over seeds on one replayed stream per seed. pp = percentage points. CI = 95% t over the pooled pairs; TOST at
2 pp uses the 90% CI. Partial cells (fewer than 10 seeds) are marked with their seed count.
"""
from __future__ import annotations

import glob
import json
from collections import defaultdict
from pathlib import Path

import numpy as np
from scipy import stats

ROOT = Path(__file__).resolve().parents[1]
F = ROOT / "results"
MODELS = ("gat", "sage", "gcn", "gin")
STREAMS = ("arxiv_year", "elliptic", "arxiv", "cora", "pubmed", "products")


def load(p):
    return json.loads(Path(p).read_text())


def mci(d, level=0.95):
    d = np.asarray([x for x in d if x is not None and np.isfinite(x)], float)
    if d.size < 2:
        return (float(d.mean()) if d.size else float("nan")), float("nan"), d.size
    h = stats.t.ppf(0.5 + level / 2, d.size - 1) * d.std(ddof=1) / np.sqrt(d.size)
    return float(d.mean()), float(h), d.size


def fmt(d, level=0.95):
    m, h, n = mci(d, level)
    return "—" if not np.isfinite(m) else (f"{m:+.2f} ± {h:.2f}" if np.isfinite(h) else f"{m:+.2f}")


def pval(d):
    d = np.asarray(d, float)
    return float(stats.ttest_1samp(d, 0).pvalue) if d.size >= 3 and not np.allclose(d, d[0]) else float("nan")


def by(rows, arm):
    return {r["seed"]: r for r in rows if r["arm"] == arm and not r.get("oom")}


def diff(rows, a, b, key="accuracy"):
    A, B = by(rows, a), by(rows, b)
    return [100 * (A[s][key] - B[s][key]) for s in sorted(set(A) & set(B))]


def split_acc(r):
    s = np.asarray([x for x in r.get("split") or [] if x], float)
    if s.size == 0:
        return float("nan"), float("nan")
    return (s[:, 0].sum() / max(s[:, 1].sum(), 1), s[:, 2].sum() / max(s[:, 3].sum(), 1))


def ratio(rows, key, a="selective", b="full"):
    A, B = by(rows, a), by(rows, b)
    v = [A[s][key] / B[s][key] for s in sorted(set(A) & set(B)) if B[s].get(key)]
    return float(np.mean(v)) if v else float("nan")


def d1():
    cells = {}
    for f in glob.glob(str(F / "D1_main/raw/streams/D1/*.json")):
        d = load(f)
        cells[(d["hyperparameters"]["model"], d["hyperparameters"]["variant"])] = d
    print("## D1 main matrix: per cell\n")
    print("Δ in pp vs frozen (mean ± 95% CI over seeds). TOST: selective − full inside ±2 pp at 90%. Cost: selective / full"
          " per update (nodes, GFLOPs, retrain peak MB, retrain p50 ms — ms from shared GPUs, indicative only).\n")
    print("| stream | model | seeds | frozen acc | full − frozen | p | selective − frozen | selective − full | TOST | "
          "periodic − frozen | random_region − frozen | iso_full − frozen | nodes | FLOPs | mem | ms | audit max | 1-hop min |")
    print("|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|")
    pooled = defaultdict(lambda: defaultdict(list))
    gains, tost_pass, tost_n, c5 = [], 0, 0, []
    for v in STREAMS:
        for m in MODELS:
            d = cells.get((m, v))
            if d is None:
                print(f"| {v} | {m} | 0 | — (not run yet) |" + " |" * 14)
                continue
            rows = d["rows"]
            fr = by(rows, "baseline")
            g = diff(rows, "full", "baseline")
            sf = diff(rows, "selective", "full")
            lo = mci(sf, 0.90)
            ok = np.isfinite(lo[1]) and abs(lo[0]) + lo[1] < 2
            tost_n += 1; tost_pass += ok
            p = pval(g)
            if np.isfinite(p) and p < 0.05 and np.mean(g) > 0:
                gains.append((v, m, np.mean(g)))
            aud = [a for r in rows if r["arm"] == "selective" for a in r.get("audits") or []]
            amax = max((a["rel_diff"] for a in aud), default=float("nan"))
            amin = min((a["rel_diff_short"] for a in aud), default=float("nan"))
            rn, rf = ratio(rows, "nodes"), ratio(rows, "gflops_per_event")
            rm, rt = ratio(rows, "retrain_peak_mb"), ratio(rows, "retrain_p50_ms")
            c5.append((v, m, rn, rf, rm, rt))
            for k, arm in (("full", "full"), ("selective", "selective"), ("periodic", "periodic"),
                           ("random_region", "random_region"), ("iso_full", "iso_full"), ("global_trigger", "global_trigger"),
                           ("global_periodic", "global_periodic"), ("roland", "roland"), ("ergnn", "ergnn"), ("ewc", "ewc"),
                           ("twp", "twp"), ("cat", "cat")):
                pooled[v][k] += diff(rows, arm, "baseline")
            pooled[v]["sel-full"] += sf
            pooled[v]["c2 full-periodic"] += diff(rows, "full", "periodic")
            pooled[v]["c3 selective-random"] += diff(rows, "selective", "random_region")
            for arm in ("baseline", "full", "selective"):
                for r in by(rows, arm).values():
                    a, c = split_acc(r)
                    pooled[v][f"split {arm} drifted"].append(100 * a); pooled[v][f"split {arm} clean"].append(100 * c)
            ns = len(fr)
            print(f"| {v} | {m} | {ns} | {100 * np.mean([r['accuracy'] for r in fr.values()]):.2f} | {fmt(g)} | "
                  f"{p:.3f} | {fmt(diff(rows, 'selective', 'baseline'))} | {fmt(sf)} | {'yes' if ok else 'no'} | "
                  f"{fmt(diff(rows, 'periodic', 'baseline'))} | {fmt(diff(rows, 'random_region', 'baseline'))} | "
                  f"{fmt(diff(rows, 'iso_full', 'baseline'))} | {rn:.3f} | {rf:.3f} | {rm:.3f} | {rt:.3f} | "
                  f"{amax:.1e} | {amin:.1e} |")
    print(f"\nTOST (C6): {tost_pass} of {tost_n} completed cells equivalent at 2 pp.")
    print(f"Significant gains of full over frozen (G, p < 0.05): {len(gains)} cells: " +
          ", ".join(f"{v}/{m} {g:+.2f}" for v, m, g in gains))
    arx = [x for x in c5 if x[0] in ("arxiv", "arxiv_year")]
    if arx:
        print(f"C5 on the Arxiv streams: nodes {min(x[2] for x in arx):.3f}–{max(x[2] for x in arx):.3f}, FLOPs "
              f"{min(x[3] for x in arx):.3f}–{max(x[3] for x in arx):.3f}, memory {min(x[4] for x in arx):.3f}–"
              f"{max(x[4] for x in arx):.3f} of full over {len(arx)} cells.")
    print(f"Cost over all completed cells: nodes {min(x[2] for x in c5):.3f}–{max(x[2] for x in c5):.3f}, "
          f"FLOPs {min(x[3] for x in c5):.3f}–{max(x[3] for x in c5):.3f}, memory {min(x[4] for x in c5):.3f}–"
          f"{max(x[4] for x in c5):.3f}.\n")
    print("## D1 pooled per stream (over models × seeds)\n")
    keys = ["full", "selective", "periodic", "random_region", "iso_full", "global_trigger", "global_periodic", "roland",
            "ergnn", "ewc", "twp", "cat", "sel-full", "c2 full-periodic", "c3 selective-random"]
    print("| stream | " + " | ".join(keys) + " |")
    print("|---|" + "---|" * len(keys))
    for v in STREAMS:
        if v in pooled:
            print(f"| {v} | " + " | ".join(fmt(pooled[v][k]) if pooled[v][k] else "—" for k in keys) + " |")
    print("\n## D1 drifted / clean holdout accuracy (%, mean over models × seeds; drifted = drifted in the last 200 steps)\n")
    print("| stream | frozen drifted | full drifted | selective drifted | frozen clean | full clean | selective clean |")
    print("|---|---|---|---|---|---|---|")
    for v in STREAMS:
        if v in pooled:
            g = lambda k: f"{np.nanmean(pooled[v][k]):.1f}" if pooled[v][k] else "—"
            print(f"| {v} | {g('split baseline drifted')} | {g('split full drifted')} | {g('split selective drifted')} | "
                  f"{g('split baseline clean')} | {g('split full clean')} | {g('split selective clean')} |")
    print()


def d2():
    rows = load(F / "D2_exact/raw/D2_backbones.json")["rows"]
    print("## D2 exactness audits (seed 0 cells; max relative per-tensor gradient difference vs the full graph)\n")
    print("| backbone | correction | hops | n | max per-tensor | max global-scale |")
    print("|---|---|---|---|---|---|")
    g = defaultdict(list)
    for r in rows:
        g[(r["backbone"], r["fixed"], r["hops"])].append(r)
    for k in sorted(g, key=lambda k: (k[0], str(k[1]), k[2])):
        v = g[k]
        print(f"| {k[0]} | {k[1]} | {k[2]} | {len(v)} | {max(x['per_tensor'] for x in v):.2e} | "
              f"{max(x['global_scale'] for x in v):.2e} |")
    print()


def d4():
    print("## D4 budget, rehearsal share and locality (selective vs full; pp, mean ± 95% CI over seeds)\n")
    print("| model | stream | N | variant | seeds | selective − full | nodes sel/full | FLOPs sel/full |")
    print("|---|---|---|---|---|---|---|---|")
    files = sorted(glob.glob(str(F / "D4_budget/raw/streams/D4/*.json")))
    base = {}
    for f in files:
        d = load(f)
        hp = d["hyperparameters"]
        if hp["tag"].count("_") == 1:                        # _N<N>: the default ρ = 0.25 run, with full
            base[(hp["model"], hp["variant"], hp["budget"])] = d["rows"]
    for f in files:
        d = load(f)
        hp = d["hyperparameters"]
        tag = hp["tag"]
        variant = "ρ=0.25" if tag.count("_") == 1 else tag.split("_", 2)[2].replace("rho", "ρ=")
        rows = d["rows"]
        if "full" not in {r["arm"] for r in rows}:
            rows = rows + [r for r in base.get((hp["model"], hp["variant"], hp["budget"]), []) if r["arm"] == "full"]
        print(f"| {hp['model']} | {hp['variant']} | {hp['budget']} | {variant} | {len(by(rows, 'selective'))} | "
              f"{fmt(diff(rows, 'selective', 'full'))} | {ratio(rows, 'nodes'):.3f} | {ratio(rows, 'gflops_per_event'):.3f} |")
    print()


def d7():
    print("## D7 limits\n")
    for f in sorted(glob.glob(str(F / "D7_limits/raw/D7/memory_*.json"))):
        d = load(f)
        print(f"**memory, {d['variant']} / {d['model']}** (median over seeds; OOM = out of memory on the 48 GB L40S)\n")
        print("| hidden | N | method | nodes | peak MB | ms | OOM seeds |")
        print("|---|---|---|---|---|---|---|")
        g = defaultdict(list)
        for r in d["rows"]:
            g[(r["hidden"], r["N"], r["method"])].append(r)
        for k in sorted(g):
            v = g[k]
            ok = [x for x in v if not x["oom"]]
            md = lambda key: f"{np.median([x[key] for x in ok]):,.0f}" if ok else "OOM"
            print(f"| {k[0]} | {k[1]} | {k[2]} | {md('nodes')} | {md('peak_mb')} | {md('ms')} | {len(v) - len(ok)}/{len(v)} |")
        print()
    print("**3-layer models** (exactness at 3 hops; 2-hop truncation as the control)\n")
    print("| graph | model | seeds | max rel. diff, 3-hop | min rel. diff, 2-hop (control) | nodes 3-hop / graph | "
          "GPU full ms | GPU selective ms |")
    print("|---|---|---|---|---|---|---|---|")
    for f in sorted(glob.glob(str(F / "D7_limits/raw/D7/deep_*.json"))):
        d = load(f)
        s = list(d["seeds"].values())
        gm = lambda k: np.median([x[k]["ms"] for x in s if x.get(k) and not x[k].get("oom")]) if s else float("nan")
        print(f"| {d['variant']} | {d['model']} | {len(s)} | {max(x['rel_3hop'] for x in s):.1e} | "
              f"{min(x['rel_2hop'] for x in s):.1e} | {np.mean([x['nodes_3hop'] / x['nodes'] for x in s]):.3f} | "
              f"{gm('gpu_full'):.1f} | {gm('gpu_selective'):.1f} |")
    print()


if __name__ == "__main__":
    print("# Phase D interim numbers (evaluation draft; partial until the queue finishes)\n")
    d1(); d2(); d4(); d7()
