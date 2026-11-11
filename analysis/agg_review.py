"""Compile the review-round results into one markdown report.

    OMP_NUM_THREADS=8 python analysis/agg_review.py > results/D8_review/REVIEW_RESULTS.md

Reads results/D8_review/raw/streams/D8/*.json, D1 (for the consistency check and the trigger-rate analysis),
D5_intro/raw/E1mix/*.json and D6_graphs/raw/audit_*_s0.json. Paired by seed; mean +- 95% CI (t) over seeds; TOST at
2 pp, alpha 0.05, as in the pre-registered C6. Cells or arms that have not run yet are listed as missing.
"""
from __future__ import annotations

import glob
import json
from pathlib import Path

import numpy as np
from scipy import stats

ROOT = Path(__file__).resolve().parents[1]
D8 = ROOT / "results/D8_review/raw/streams/D8"
D1 = ROOT / "results/D1_main/raw/streams/D1"
MODELS = ("gat", "sage", "gcn", "gin")
NAME = dict(gat="GAT", sage="SAGE", gcn="GCN", gin="GIN")
OUT = []


def p(s=""):
    OUT.append(s)


def load(path):
    path = Path(path)
    return json.loads(path.read_text()) if path.exists() else None


def by_arm(d):
    out = {}
    for r in (d or {}).get("rows", []):
        out.setdefault(r["arm"], {})[r["seed"]] = r
    return out


def ci(v):
    v = np.asarray(v, float)
    return float(stats.t.ppf(.975, len(v) - 1) * v.std(ddof=1) / np.sqrt(len(v))) if len(v) > 1 else float("nan")


def paired(a, b, key="accuracy", scale=100):
    s = sorted(set(a) & set(b))
    s = [x for x in s if not a[x].get("oom") and not b[x].get("oom")]
    return [scale * (a[x][key] - b[x][key]) for x in s]


def fmt(v, d=2):
    return "—" if not v else f"{np.mean(v):+.{d}f} ± {ci(v):.{d}f}"


def tost(v, m=2.0):
    v = np.asarray(v, float)
    if len(v) < 3:
        return "—"
    se = v.std(ddof=1) / np.sqrt(len(v))
    if se == 0:
        return "yes"
    t1, t2 = (v.mean() + m) / se, (v.mean() - m) / se
    pv = max(1 - stats.t.cdf(t1, len(v) - 1), stats.t.cdf(t2, len(v) - 1))
    return f"{'yes' if pv < .05 else 'no'} (p={pv:.3f})"


def mean_of(rows, key):
    v = [r[key] for r in rows.values() if not r.get("oom") and r.get(key) is not None]
    return float(np.mean(v)) if v else float("nan")


# --- R1 -----------------------------------------------------------------------------------------------------------
def r1():
    p("## R1. Matched roots, root rule × parent graph, timer trigger (Arxiv streams, seeds 0–9)\n")
    p("Nodes = mean nodes materialised per repair; ratio to `full` in brackets. |S| = distinct roots per repair. "
      "`*_exact` = parent is the whole current graph. Accuracy: pp against Frozen, mean ± 95% CI over seeds.\n")
    for v in ("arxiv_year", "arxiv"):
        p(f"### {v}\n")
        p("| model | GIST \\|S\\| / N (median, min–max per repair) | full | GIST (D1 rule) | GIST, \\|S\\|=N | Full, GIST's \\|S\\| | Full, whole-graph parent | GIST, whole-graph parent | OOM |")
        p("|---|---|---|---|---|---|---|---|---|")
        acc = []
        for m in MODELS:
            a = by_arm(load(D8 / f"{m}_{v}.json"))
            a.update({k: x for k, x in by_arm(load(D8 / f"{m}_{v}_exact.json")).items() if k not in a})
            if "full" not in a:
                p(f"| {NAME[m]} | missing | | | | | | | |")
                continue
            N = 8192
            per = [x / N for r in a.get("selective_logged", {}).values() for x in r.get("roots_per_event", [])]
            sdesc = f"{np.median(per):.2f} ({min(per):.2f}–{max(per):.2f}), {np.mean(np.isclose(per, 1)) * 100:.0f}% at N" if per else "—"
            f = mean_of(a["full"], "nodes")
            cell = lambda arm: "—" if arm not in a else f"{mean_of(a[arm], 'nodes') / 1e3:.1f}K ({mean_of(a[arm], 'nodes') / f:.2f})"
            oom = ", ".join(f"{arm} {sum(bool(r.get('oom')) for r in a[arm].values())}/{len(a[arm])}"
                            for arm in ("full_exact", "selective_exact") if arm in a and any(r.get("oom") for r in a[arm].values()))
            p(f"| {NAME[m]} | {sdesc} | {f / 1e3:.1f}K | {cell('selective_logged')} | {cell('selective_topup')} | "
              f"{cell('full_isoroots')} | {cell('full_exact')} | {cell('selective_exact')} | {oom or 'none'} |")
            acc.append((m, a))
        p("")
        p("| model | arm | − Frozen (pp) | − Full (pp) | TOST 2 pp vs Full | repairs | peak MB |")
        p("|---|---|---|---|---|---|---|")
        for m, a in acc:
            for arm in ("full", "selective_logged", "selective_topup", "full_isoroots", "selective_periodic", "full_exact", "selective_exact"):
                if arm not in a:
                    continue
                dfz = paired(a[arm], a["baseline"]) if "baseline" in a else []
                dff = paired(a[arm], a["full"]) if arm != "full" else []
                p(f"| {NAME[m]} | {arm} | {fmt(dfz)} | {fmt(dff) if arm != 'full' else ''} | {tost(dff) if arm != 'full' else ''} | "
                  f"{mean_of(a[arm], 'events'):.1f} | {mean_of(a[arm], 'retrain_peak_mb'):.0f} |")
        p("")
        p("Trigger: `selective_periodic` (timer at Full's realised rate) − `selective_logged` (detector), pp, paired: " +
          "; ".join(f"{NAME[m]} {fmt(paired(a['selective_periodic'], a['selective_logged']))}" for m, a in acc
                    if "selective_periodic" in a and "selective_logged" in a))
        p("")
        # consistency: selective_logged reproduces D1's selective (same draws; GPU kernels are not bit-deterministic)
        cons = []
        for m, a in acc:
            d1 = by_arm(load(D1 / f"{m}_{v}.json")).get("selective", {})
            if "selective_logged" in a and d1:
                dn = [abs(a["selective_logged"][s]["nodes"] / d1[s]["nodes"] - 1) for s in a["selective_logged"] if s in d1]
                da = [100 * abs(a["selective_logged"][s]["accuracy"] - d1[s]["accuracy"]) for s in a["selective_logged"] if s in d1]
                cons.append(f"{NAME[m]}: nodes within {100 * max(dn):.2f}%, accuracy within {max(da):.2f} pp")
        p("Consistency with D1 `selective`: " + ("; ".join(cons) if cons else "—"))
        p("")


# --- R2 -----------------------------------------------------------------------------------------------------------
def r2():
    p("## R2. Oracle trigger (Arxiv, injected; fires on the known event steps, same cooldown)\n")
    p("| model | trigger | repairs | Full − Frozen | GIST − Frozen | GIST − Full |")
    p("|---|---|---|---|---|---|")
    for m in MODELS:
        for tag, lab in (("", "detector"), ("_oracle", "oracle")):
            a = by_arm(load(D8 / f"{m}_arxiv{tag}.json"))
            if not {"baseline", "full", "selective_logged"} <= set(a):
                p(f"| {NAME[m]} | {lab} | missing | | | |")
                continue
            p(f"| {NAME[m]} | {lab} | {mean_of(a['full'], 'events'):.1f} / {mean_of(a['selective_logged'], 'events'):.1f} | "
              f"{fmt(paired(a['full'], a['baseline']))} | {fmt(paired(a['selective_logged'], a['baseline']))} | "
              f"{fmt(paired(a['selective_logged'], a['full']))} |")
    p("")


# --- R3 -----------------------------------------------------------------------------------------------------------
def f1_bal(conf):
    c = np.asarray(conf, float)
    tp = np.diag(c)
    rec = tp / np.maximum(c.sum(1), 1)
    prec = tp / np.maximum(c.sum(0), 1)
    f1 = np.where(prec + rec > 0, 2 * prec * rec / np.maximum(prec + rec, 1e-12), 0)
    present = c.sum(1) > 0
    return float(f1[present].mean()), float(rec[present].mean()), rec, c.sum(1)


def r3():
    p("## R3. Elliptic: macro-F1 and balanced accuracy (holdout, whole stream, seeds 0–9)\n")
    p("| model | arm | accuracy | macro-F1 | balanced acc. | minority recall | macro-F1 − Frozen (pp) | macro-F1 GIST − Full (pp) |")
    p("|---|---|---|---|---|---|---|---|")
    for m in MODELS:
        a = by_arm(load(D8 / f"{m}_elliptic.json"))
        if not {"baseline", "full", "selective_logged"} <= set(a):
            p(f"| {NAME[m]} | missing | | | | | | |")
            continue
        met = {arm: {s: f1_bal(r["confusion"]) for s, r in rows.items()} for arm, rows in a.items()}
        for arm in ("baseline", "full", "selective_logged"):
            mf = [met[arm][s][0] for s in sorted(met[arm])]
            ba = [met[arm][s][1] for s in sorted(met[arm])]
            minority = int(np.argmin(next(iter(met[arm].values()))[3]))
            mr = [met[arm][s][2][minority] for s in sorted(met[arm])]
            d0 = [100 * (met[arm][s][0] - met["baseline"][s][0]) for s in sorted(met[arm]) if s in met["baseline"]]
            dg = [100 * (met["selective_logged"][s][0] - met["full"][s][0]) for s in sorted(met["full"]) if s in met["selective_logged"]] if arm == "selective_logged" else []
            p(f"| {NAME[m]} | {arm} | {100 * mean_of(a[arm], 'accuracy'):.2f} | {100 * np.mean(mf):.2f} | {100 * np.mean(ba):.2f} | "
              f"{100 * np.mean(mr):.1f} | {fmt(d0) if arm != 'baseline' else ''} | {fmt(dg)} |")
    p("")


# --- R4 -----------------------------------------------------------------------------------------------------------
def r4():
    fs = sorted(glob.glob(str(ROOT / "results/D6_graphs/raw/audit_*_s0.json")))
    p("## R4. Link-prediction exactness on evaluation seed 0 (float64, 3 updates per model × graph)\n")
    if not fs:
        p("missing\n")
        return
    two = [r["rel_2hop"] for f in fs for r in json.loads(Path(f).read_text())]
    one = [r["rel_1hop"] for f in fs for r in json.loads(Path(f).read_text())]
    p(f"{len(two)} audited updates in {len(fs)} model × graph cells: 2-hop subgraph max relative difference "
      f"{max(two):.1e}; 1-hop control min {min(one):.1e}, median {np.median(one):.1e}.\n")


# --- R5 -----------------------------------------------------------------------------------------------------------
def r5():
    p("## R5. Static 2 × 2: root rule × parent graph (share of the graph materialised; seeds 0–9; ratio to Full = uniform roots, fan-out parent)\n")
    p("| graph | N | uniform, fan-out (Full) | uniform, whole graph | region, fan-out | region, whole graph | GIST rule, fan-out | GIST rule, whole graph |")
    p("|---|---|---|---|---|---|---|---|")
    for f in sorted(glob.glob(str(ROOT / "results/D5_intro/raw/E1mix/*.json"))):
        d = json.loads(Path(f).read_text())
        n = d["num_nodes"]
        for N in sorted({r["N"] for r in d["rows"]}):
            g = lambda kind, key: float(np.mean([r[key] for r in d["rows"] if r["N"] == N and r["roots"] == kind])) / n
            full = g("uniform", "fanout_25_15_nodes")
            cells = [f"{100 * g(k, key):.1f}% ({g(k, key) / full:.2f})" for k in ("uniform", "region", "gist")
                     for key in ("fanout_25_15_nodes", "exact_nodes")]
            p(f"| {d['graph']} | {N} | " + " | ".join(cells) + " |")
    p("")


# --- sensitivity --------------------------------------------------------------------------------------------------
def sens():
    p("## Sensitivity: cooldown κ and detector threshold τ (Arxiv-year, seeds 0–9)\n")
    p("| model | setting | repairs (Full / GIST) | Full − Frozen | GIST − Frozen | GIST − Full |")
    p("|---|---|---|---|---|---|")
    for m in MODELS:
        for tag, lab in (("", "κ=50, τ at 5% (default)"), ("_k25", "κ=25"), ("_k100", "κ=100"), ("_tau1", "τ at 1%"), ("_tau10", "τ at 10%")):
            a = by_arm(load(D8 / f"{m}_arxiv_year{tag}.json"))
            if not {"baseline", "full", "selective_logged"} <= set(a):
                p(f"| {NAME[m]} | {lab} | missing | | | |")
                continue
            p(f"| {NAME[m]} | {lab} | {mean_of(a['full'], 'events'):.1f} / {mean_of(a['selective_logged'], 'events'):.1f} | "
              f"{fmt(paired(a['full'], a['baseline']))} | {fmt(paired(a['selective_logged'], a['baseline']))} | "
              f"{fmt(paired(a['selective_logged'], a['full']))} |")
    p("")


# --- trigger behaviour, from D1 (no new run) ----------------------------------------------------------------------
def trigger_rate():
    p("## Trigger behaviour in D1 (no new run): how often the detector fires, and how many repairs the cooldown allows\n")
    p("| stream | flagged steps (%) | 50-step windows with a flag (%) | Full repairs / 1000 steps | cooldown ceiling |")
    p("|---|---|---|---|---|")
    for v in ("arxiv_year", "arxiv", "elliptic", "cora", "pubmed", "products"):
        fl, win, rep, ceil = [], [], [], []
        for f in glob.glob(str(D1 / f"*_{v}.json")):
            d = json.loads(Path(f).read_text())
            for r in d["rows"]:
                if r["arm"] != "full":
                    continue
                T = len(r["step_accuracy"])
                x = np.zeros(T, bool)
                x[[s for s in r["flag_steps"] if s < T]] = True
                cs = np.concatenate([[0], np.cumsum(x)])
                fl.append(x.mean())
                win.append(np.mean([cs[min(T, s + 50)] - cs[s] > 0 for s in range(T - 50)]))
                rep.append(1000 * r["events"] / T)
                ceil.append(1000 / 51)
        if fl:
            p(f"| {v} | {100 * np.mean(fl):.1f} | {100 * np.mean(win):.0f} | {np.mean(rep):.1f} | {np.mean(ceil):.1f} |")
    p("")


def main():
    p("# Review-round results\n")
    for f in (r1, r2, r3, r4, r5, sens, trigger_rate):
        try:
            f()
        except Exception as e:                 # a partial run must not hide the other sections
            p(f"**{f.__name__} failed: {e!r}**\n")
    print("\n".join(OUT))


if __name__ == "__main__":
    main()
