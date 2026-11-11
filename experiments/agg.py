"""Aggregate every experiments result into results/intro_numbers.md (results/smoke/ under GIST_SMOKE).

    python -m experiments.agg

One section per intro.tex claim (CLAIMS.md). Accuracy differences are in percentage points, paired by seed on one
replayed stream, pooled over models x seeds: mean ± 95% CI (t), and the count of cells (model) with a significant
difference (paired t over 5 seeds, p < 0.05). Missing results are skipped, so this runs at any point.
"""
from __future__ import annotations

import glob
import json
from collections import defaultdict
from pathlib import Path

import numpy as np
from scipy import stats

from . import common as C

RES = C.RESULTS
L = []
p = L.append


def ci(d):
    d = np.asarray([x for x in d if x is not None and np.isfinite(x)], float)
    if d.size == 0:
        return "—"
    if d.size == 1:
        return f"{d[0]:+.2f}"
    h = stats.t.ppf(0.975, d.size - 1) * d.std(ddof=1) / np.sqrt(d.size)
    return f"{d.mean():+.2f} ± {h:.2f}"


def sig(d):
    d = np.asarray(d, float)
    if d.size < 3 or np.allclose(d, d[0]):
        return None
    return float(stats.ttest_1samp(d, 0).pvalue)


def load_cells(exp):
    cells = {}
    for f in sorted(glob.glob(str(RES / "streams" / exp / "*.json"))):
        d = json.loads(Path(f).read_text())
        hp = d["hyperparameters"]
        cells[(hp["model"], hp["variant"], hp.get("tag", ""))] = d
    return cells


def by(rows, arm):
    return {r["seed"]: r for r in rows if r["arm"] == arm}


def post_event(row, drift_steps, w=100):
    acc = np.asarray(row["step_accuracy"], float)
    vals = [np.nanmean(acc[s:s + w]) for s in drift_steps if s < acc.size]
    return float(np.nanmean(vals)) if vals else float("nan")


def paired(cell, a, b, key="accuracy"):
    """Paired differences over seeds; a row whose update did not fit on the GPU (oom) never updated: excluded."""
    A, B = by(cell["rows"], a), by(cell["rows"], b)
    out = []
    for s in sorted(set(A) & set(B)):
        if A[s].get("oom") or B[s].get("oom"):
            continue
        if key == "post":
            ds = cell["meta"][str(s)]["drift_steps"]
            out.append(100 * (post_event(A[s], ds) - post_event(B[s], ds)))
        else:
            out.append(100 * (A[s][key] - B[s][key]))
    return out


NATURAL = ("arxiv_year", "elliptic")          # the streams where retraining has a positive impact (E2)
LABEL = ("cora_label", "pubmed_label", "arxiv_label")

# rows of the comparison table: (method, paper / system, E2 arm, E4 method, E7 condition, updates, an update covers)
COMPARE = [
    ("Exact incremental inference", "InkStream, Ripple, StreamTGN", "baseline", "frozen", "idle", "embeddings only", "affected embeddings; weights fixed"),
    ("Adaptive global retraining", "InstantGNN", "global_trigger", None, "global", "parameters", "whole graph, when the drift signal fires"),
    ("Periodic global retraining", "D3-GNN", "global_periodic", None, "global", "parameters", "whole graph, on a timer"),
    ("Live update + meta-averaging", "ROLAND", "roland", None, None, "parameters", "whole graph, 5 steps per snapshot"),
    ("Experience replay", "ER-GNN", "ergnn", None, None, "parameters", "N uniform roots + replay buffer (N/4)"),
    ("Weight consolidation", "EWC", "ewc", None, None, "parameters", "N uniform roots + penalty"),
    ("Topology-aware consolidation", "TWP", "twp", None, None, "parameters", "N uniform roots + penalty"),
    ("Full retraining (sampled)", "standard practice", "full", "ns_25_15", "full", "parameters", "N uniform roots, fanout [25, 15]"),
    ("Periodic retraining (sampled)", "standard practice", "periodic", None, None, "parameters", "N uniform roots, on a timer"),
    ("Neighbour sampling", "GraphSAGE", None, "ns_15_10", None, "parameters", "R, fanout [15, 10]"),
    ("Online neighbour sampling", "Online CGL", None, "ns_2_2", None, "parameters", "R, fanout [2, 2]"),
    ("Cluster batching", "Cluster-GCN", None, "cluster_gcn", None, "parameters", "R's clusters, inter-cluster edges dropped"),
    ("Subgraph sampling", "GraphSAINT (rw)", None, "saint_rw", None, "parameters", "random-walk subgraph around R"),
    ("Historical embeddings", "GNNAutoScale", None, "gas", None, "parameters", "R; out-of-batch activations from history"),
    ("Variance-reduced sampling", "VR-GCN", None, "vrgcn", None, "parameters", "R + 2 sampled neighbours; history"),
    ("Lazy updates", "IGLU", None, "iglu@1", None, "parameters", "R's 2-hop, gradient 1 step stale"),
    ("**Our Model (GIST)**", "this paper", "selective", "exact_lhop", "selective", "parameters", "exact induced 2-hop of the repaired region's roots"),
]


def summary():
    e2 = load_cells("E2")
    e4 = defaultdict(lambda: defaultdict(list))
    for f in sorted(glob.glob(str(RES / "E4" / "*.json"))):
        if f.endswith("_cal.json"):          # smoke runs on calibration seed 100: not results
            continue
        d = json.loads(Path(f).read_text())
        if d["variant"] not in LABEL:
            continue
        for s_, r in d["seeds"].items():
            ex = r["part1"]["exact_lhop"]["nodes"]
            for m_, v in r["part1"].items():
                e4[m_]["err"].append(v["rel_err"]); e4[m_]["cos"].append(v["cos"]); e4[m_]["nodes"].append(v["nodes"] / max(ex, 1))
            for m_, v in r["part2"].items():
                k = max(v, key=int)
                e4[m_]["drift"].append(v[k]["acc_drifted"]); e4[m_]["clean"].append(v[k]["acc_clean"]); e4[m_]["dis"].append(v[k]["disagree_exact"])
            e4["frozen"]["drift"].append(r["frozen"]["acc_drifted"]); e4["frozen"]["clean"].append(r["frozen"]["acc_clean"])
    e7 = defaultdict(list)
    for f in sorted(glob.glob(str(RES / "E7" / "*.json"))):
        d = json.loads(Path(f).read_text())
        for s_, rows in d["seeds"].items():
            idle = next(r["p99_ms"] for r in rows if r["condition"] == "idle")
            for r in rows:
                e7[r["condition"]].append("OOM" if r.get("repair_oom") else r["p99_ms"] / idle)

    def acc(arm):
        d = [x for (m, v, t), c in e2.items() if v in NATURAL and t == "" for x in paired(c, arm, "baseline")]
        return ci(d) if d else "—"

    def cost(arm, key):
        rat = []
        for (m, v, t), c in e2.items():
            if t != "":
                continue
            A = [r[key] for r in c["rows"] if r["arm"] == arm and r["events"] and not r.get("oom")]
            F_ = [r[key] for r in c["rows"] if r["arm"] == "full" and r["events"]]
            if A and F_:
                rat.append(np.mean(A) / np.mean(F_))
        return f"{np.mean(rat):.2f}×" if rat else "—"

    def oom(arm):
        cells = sorted({f"{m}/{v}" for (m, v, t), c in e2.items() if t == "" for r in c["rows"] if r["arm"] == arm and r.get("oom")})
        return "no: " + ", ".join(cells) if cells else "yes"

    p("## Summary: every compared method against GIST\n")
    p("One row per framework or paper. Our re-implementations run inside the same runtime, stream, backbones, budget and "
      "trigger (experiments/arms.py, e4_fidelity.py); seeds 110–114, 4 backbones. Columns and the datasets each averages over:\n")
    p("- **Same gradient as full-graph**: median max-relative gradient error vs the exact full-graph gradient, for the same repair "
      "update (E4, float64; label-drift Cora / PubMed / Arxiv).")
    p("- **Nodes, FLOPs (× full)**: per update, relative to sampled full retraining at the same budget N, averaged over every E2 "
      "stream the method ran on (E2). **Nodes (× exact 2-hop)**: what its update materialises relative to GIST's exact "
      "subgraph for the same roots (E4).")
    p("- **Fits on a 48 GB L40S**: whether its update ran without running out of memory in E2 (cells where it did not are listed).")
    p("- **Δ accuracy vs frozen**: pp, paired over seeds, mean ± 95% CI over models × seeds, over the datasets where updating has a "
      "positive impact: natural drift, Arxiv-by-year and Elliptic (E2). On injected and label drift every method is negative "
      "(see E2); those streams are excluded here for every row alike.")
    p("- **Drifted / clean accuracy after repair, disagreement with exact**: holdout accuracy after 20 repair steps on the drifted "
      "and the clean nodes, and the share of predictions that differ from the exactly repaired model (E4; label drift, where a "
      "repair has a positive impact on the drifted nodes).")
    p("- **Serving p99 × idle**: inference p99 latency while the update runs on the same GPU, over idle (E7; GAT and SAGE on Arxiv).\n")
    p("| Method | Paper / system | Updates | An update covers | Same gradient as full-graph? (rel. error) | Nodes (× full) | FLOPs (× full) "
      "| Nodes (× exact 2-hop) | Fits on a 48 GB L40S | Δ accuracy vs frozen, natural drift (pp) | Drifted-node acc after repair | "
      "Clean-node acc after repair | Disagree with exact repair | Serving p99 × idle |")
    p("|---|---|---|---|---|---|---|---|---|---|---|---|---|---|")
    med = lambda x: float(np.median(x)) if x else float("nan")
    mean = lambda x: float(np.mean(x)) if x else float("nan")
    for name, paper, arm, m4, c7, upd, scope in COMPARE:
        g = dict(e4.get(m4, {})) if m4 else {}
        rep_key = {"gas": "gas@1", "vrgcn": "vrgcn@1"}.get(m4)        # their repair runs carry the history refresh period
        if rep_key:
            for k_ in ("drift", "clean", "dis"):
                g[k_] = e4.get(rep_key, {}).get(k_, [])
        if m4 == "exact_lhop":
            grad = f"**yes, exact** ({med(g.get('err', [])):.0e})"
        elif m4 == "frozen":
            grad = "— (no update)"
        elif m4 and g.get("err"):
            grad = f"no ({med(g['err']):.1%})"
        elif m4 and m4.startswith("iglu"):
            grad = "no (stale by 1 step)"
        elif arm in ("global_trigger", "global_periodic", "roland"):
            grad = "yes (whole graph)"
        else:
            grad = "no (sampled batch)" if arm in ("full", "periodic", "ergnn", "ewc", "twp") else "—"
        nodes_f = "0" if arm == "baseline" else (cost(arm, "nodes") if arm else "—")
        flops_f = "0" if arm == "baseline" else (cost(arm, "gflops_per_event") if arm else "—")
        nodes_x = "1.00×" if m4 in ("exact_lhop", "iglu@1") else (f"{med(g['nodes']):.2f}×" if m4 and g.get("nodes") else "—")
        fits = "yes" if arm == "baseline" else (oom(arm) if arm else "yes")
        da = "0 (reference)" if arm == "baseline" else (acc(arm) if arm else "—")
        dr = f"{mean(g.get('drift', [])):.3f}" if g.get("drift") else "—"
        cl = f"{mean(g.get('clean', [])):.3f}" if g.get("clean") else "—"
        dis = "0 (reference)" if m4 == "exact_lhop" else (f"{mean(g['dis']):.1%}" if g.get("dis") else "—")
        if c7 and e7.get(c7):
            vals = e7[c7]
            num = [v for v in vals if v != "OOM"]
            lat = "1.00" if c7 == "idle" else ((f"{np.mean(num):.2f}×" if num else "") + (" (SAGE only; GAT OOM)" if "OOM" in vals else ""))
        else:
            lat = "—"
        p(f"| {name} | {paper} | {upd} | {scope} | {grad} | {nodes_f} | {flops_f} | {nodes_x} | {fits} | {da} | {dr} | {cl} | {dis} | {lat} |")
    p("")
    p("Reading it:")
    p("- **GIST is the only parameter update that is exact and cheaper than full retraining** (0.62× its nodes and "
      "FLOPs), and it fits wherever full retraining does.")
    p("- **Whole-graph updates** (InstantGNN, D3-GNN, ROLAND) are also exact, but materialise 6–10× full's nodes and run out "
      "of memory for GAT on ogbn-arxiv. ROLAND has the largest natural-drift gain (+0.65 pp, three backbones; GAT OOM) at "
      "about 10× full's cost.")
    p("- **Sampling, clustering, history and lazy methods** are cheaper, but compute a different update (3–34% gradient "
      "error). Their repaired models disagree with the exact repair on 4–10% of predictions, at about the same accuracy.")
    p("- **On natural drift**, sampled full retraining, EWC and TWP gain +0.35 to +0.44 pp. GIST's selective update "
      "stays near the frozen model (+0.06): its saving does not carry the gain in this setting. The trigger and region "
      "choice (C2 / C3) are not claimed.")
    p("- **Fresh embeddings with fixed weights** (InkStream / Ripple / StreamTGN) leave drifted nodes wrong: 3.5% accuracy, "
      "against 42–44% after any repair.\n")


def e1():
    files = sorted(glob.glob(str(RES / "E1" / "*.json")))
    if not files:
        return
    p("## E1. How much of the graph an update materialises (intro: cost of a parameter update)\n")
    p("Share of the graph in the 2-hop expansion of N roots (mean over seeds): exact, and with fanout [15, 10]; uniform roots vs a "
      "METIS region of the same N. `ratio` = uniform / region (exact).\n")
    p("| graph | family | nodes | mean deg | N | uniform exact | uniform fanout 15/10 | region exact | ratio | uniform 3-hop |")
    p("|---|---|---|---|---|---|---|---|---|---|")
    for f in files:
        d = json.loads(Path(f).read_text())
        n = d["num_nodes"]
        for N in sorted({r["N"] for r in d["rows"]}):
            u = [r for r in d["rows"] if r["N"] == N and r["roots"] == "uniform"]
            g = [r for r in d["rows"] if r["N"] == N and r["roots"] == "region"]
            if not u or not g:
                continue
            ue = np.mean([r["exact"]["2"] for r in u]) / n
            uf = np.mean([r["fanout_15_10"]["nodes"] for r in u]) / n
            ge = np.mean([r["exact"]["2"] for r in g]) / n
            u3 = np.mean([r["exact"].get("3", np.nan) for r in u]) / n
            p(f"| {d['graph']} | {d['family']} | {n:,} | {d['mean_degree']:.1f} | {N:,} | {ue:.1%} | {uf:.1%} | {ge:.1%} | "
              f"{ue / max(ge, 1e-12):.2f} | {u3:.1%} |")
    p("")


def e2():
    cells = load_cells("E2")
    if not cells:
        return
    p("## E2. Fresh embeddings with fixed weights vs repair (intro Gap 1)\n")
    p("`frozen` = baseline (fixed weights; the embeddings an exact incremental system computes). Δ vs frozen, pp, over the whole "
      "stream and in the 100 steps after each drift event (post). `recompute` = nodes per step an exact incremental system "
      "recomputes; on label-drift steps the input does not change, so it recomputes nothing new.\n")
    arms = ["full", "selective", "periodic", "global_periodic", "global_trigger", "roland", "ergnn", "ewc", "twp"]
    p("| variant | models | frozen acc | " + " | ".join(f"{a} − frozen" for a in arms) + " | post: full − frozen | recompute / step |")
    p("|---|---|---|" + "---|" * len(arms) + "---|---|")
    for v in C.VARIANTS:
        cs = [c for (m, vv, t), c in cells.items() if vv == v and t == ""]
        if not cs:
            continue
        frozen = [r["accuracy"] for c in cs for r in c["rows"] if r["arm"] == "baseline"]
        cols = []
        for a in arms:
            d = [x for c in cs for x in paired(c, a, "baseline")]
            oom = sorted({c["hyperparameters"]["model"] for c in cs for r in c["rows"] if r["arm"] == a and r.get("oom")})
            cols.append((ci(d) if d else "—") + (f" (OOM: {','.join(oom)})" if oom else ""))
        post = [x for c in cs for x in paired(c, "full", "baseline", "post")]
        rec = [np.mean(c["meta"][s]["recompute"]) for c in cs for s in c["meta"] if "recompute" in c["meta"][s]]
        p(f"| {v} | {len(cs)} | {np.mean(frozen):.4f} | " + " | ".join(cols) + f" | {ci(post)} | {np.mean(rec):,.0f} |")
    p("")
    p("Cost per update (mean nodes in backward; GFLOPs per event), pooled:\n")
    p("| variant | " + " | ".join(arms) + " |")
    p("|---|" + "---|" * len(arms))
    for v in C.VARIANTS:
        cs = [c for (m, vv, t), c in cells.items() if vv == v and t == ""]
        if not cs:
            continue
        cols = []
        for a in arms:
            rr = [r for c in cs for r in c["rows"] if r["arm"] == a and r["events"]]
            cols.append(f"{np.mean([r['nodes'] for r in rr]):,.0f} / {np.mean([r['gflops_per_event'] for r in rr]):.2f}" if rr else "—")
        p(f"| {v} | " + " | ".join(cols) + " |")
    p("")


def e3a():
    files = sorted(glob.glob(str(RES / "E3a" / "*.json")))
    if not files:
        return
    p("## E3a. Cost of one update vs region size and graph size (intro Gap 2)\n")
    p("Nodes materialised per update (mean over seeds). Selective at region r % of the nodes; the others do not depend on r.\n")
    for f in files:
        d = json.loads(Path(f).read_text())
        rows = d["rows"]
        fr = sorted({r["fraction"] for r in rows})
        regs = sorted({r["region_pct"] for r in rows if r["method"] == "selective"})
        p(f"**{d['graph']} / {d['model']}**\n")
        p("| graph fraction | nodes | full | ergnn | global | roland | " + " | ".join(f"selective r={r:g}%" for r in regs) + " | full ms | selective ms (min r) | global ms |")
        p("|---|---|---|---|---|---|" + "---|" * len(regs) + "---|---|---|")
        for x in fr:
            sub = [r for r in rows if r["fraction"] == x]
            def m(meth, key="nodes", reg=None):
                rr = [r for r in sub if r["method"] == meth and (reg is None or r["region_pct"] == reg)]
                if not rr:
                    return "—"
                oom = all(r.get("oom") for r in rr)
                if key == "ms":
                    return "OOM" if oom else f"{np.mean([r['ms'] for r in rr if r['ms'] is not None]):.1f}"
                return f"{np.mean([r[key] for r in rr]):,.0f}" + (" (OOM)" if oom else "")
            p(f"| {x:g} | {sub[0]['num_nodes']:,} | {m('full')} | {m('ergnn')} | {m('global')} | {m('roland')} | "
              + " | ".join(m('selective', reg=r) for r in regs)
              + f" | {m('full', 'ms')} | {m('selective', 'ms', regs[0])} | {m('global', 'ms')} |")
        p("")


def e4():
    files = sorted(f for f in glob.glob(str(RES / "E4" / "*.json")) if not f.endswith("_cal.json"))  # reported seeds only
    if not files:
        return
    p("## E4. Gradient fidelity of cost-bounding updates (intro Gap 3)\n")
    p("Part 1 (float64, at the warm weights): relative error max|g − g*| / max|g*| and cosine to the exact full-graph gradient "
      "(median over seeds and models); nodes as a share of the exact 2-hop subgraph's. Part 2: holdout accuracy on drifted / "
      "clean nodes after 20 repair steps, and disagreement with the exactly repaired model.\n")
    agg = defaultdict(lambda: defaultdict(list))
    for f in files:
        d = json.loads(Path(f).read_text())
        for s, r in d["seeds"].items():
            ex = r["part1"]["exact_lhop"]["nodes"]
            for meth, v in r["part1"].items():
                agg[(d["variant"], meth)]["err"].append(v["rel_err"])
                agg[(d["variant"], meth)]["cos"].append(v["cos"])
                agg[(d["variant"], meth)]["nodes"].append(v["nodes"] / max(ex, 1))
            for meth, v in r["part2"].items():
                k = max(v, key=int)
                agg[(d["variant"], meth)]["drift"].append(v[k]["acc_drifted"])
                agg[(d["variant"], meth)]["clean"].append(v[k]["acc_clean"])
                agg[(d["variant"], meth)]["dis"].append(v[k]["disagree_exact"])
            agg[(d["variant"], "frozen")]["drift"].append(r["frozen"]["acc_drifted"])
            agg[(d["variant"], "frozen")]["clean"].append(r["frozen"]["acc_clean"])
    p("| variant | method | rel. error | cosine | nodes / exact 2-hop | drifted acc (k max) | clean acc | disagree w/ exact |")
    p("|---|---|---|---|---|---|---|---|")
    med = lambda x: f"{np.median(x):.2e}" if x else "—"
    for (v, meth), d in sorted(agg.items()):
        p(f"| {v} | {meth} | {med(d['err'])} | {np.median(d['cos']):.4f} | {np.median(d['nodes']):.2f} | "
          f"{np.mean(d['drift']):.3f} | {np.mean(d['clean']):.3f} | {np.mean(d['dis']) if d['dis'] else float('nan'):.3f} |"
          if d.get("cos") else f"| {v} | {meth} | — | — | — | {np.mean(d['drift']) if d['drift'] else float('nan'):.3f} | "
          f"{np.mean(d['clean']) if d['clean'] else float('nan'):.3f} | {np.mean(d['dis']) if d.get('dis') else float('nan'):.3f} |")
    p("")


def e5():
    p("## E5. Evaluation pitfalls (intro Gap 4: `\\res{a}`–`\\res{k}`)\n")
    w = load_cells("E5w")
    if w:
        p("**Under-trained start** (`\\res{a,b,c}`): full − frozen (pp) by warmup epochs.\n")
        p("| variant | epochs | models | full − frozen | significant cells |")
        p("|---|---|---|---|---|")
        for v in ("cora", "arxiv_year"):
            for e in sorted({int(t[2:]) for (m, vv, t) in w if vv == v and t.startswith("_w")}):
                cs = [c for (m, vv, t), c in w.items() if vv == v and t == f"_w{e}"]
                d = [x for c in cs for x in paired(c, "full", "baseline")]
                ns = sum(1 for c in cs if (sig(paired(c, "full", "baseline")) or 1) < 0.05)
                p(f"| {v} | {e} | {len(cs)} | {ci(d)} | {ns} |")
        p("")
    lr = load_cells("E5lr")
    if lr:
        p("**Shared learning rate**: frozen accuracy and full − frozen when warmup and adaptation share lr.\n")
        p("| variant | shared lr | frozen acc | full − frozen |")
        p("|---|---|---|---|")
        for (m, v, t) in sorted(lr):
            pass
        for v in ("cora", "arxiv_year"):
            for t in sorted({t for (m, vv, t) in lr if vv == v}):
                cs = [c for (m, vv, tt), c in lr.items() if vv == v and tt == t]
                fr = [r["accuracy"] for c in cs for r in c["rows"] if r["arm"] == "baseline"]
                p(f"| {v} | {t[7:]} | {np.mean(fr):.4f} | {ci([x for c in cs for x in paired(c, 'full', 'baseline')])} |")
        p("")
    for name in ("arxiv", "cora"):
        d = C.read_json(RES / "E5" / f"coverage_{name}.json")
        if d:
            c = [r["cover_128"] for r in d["rows"]]
            p(f"**Fixed budget** (`\\res{{d,e}}`), {name}: a 128-root update covers {np.percentile(c, 10):.1%}–{np.percentile(c, 90):.1%} "
              f"(p10–p90; median {np.median(c):.1%}) of an event's trainable drifted nodes ({len(c)} events; matched budget N = {d['budget']}).\n")
        h = C.read_json(RES / "E5" / f"halo_{name}.json")
        if h:
            r = h["rows"]
            p(f"**Halo supervision** (`\\res{{f}}`), {name}: of the nodes halo supervision trains on, "
              f"{np.mean([x['truncated_share'] for x in r]):.1%} have a truncated receptive field and {np.mean([x['blind_share'] for x in r]):.1%} "
              f"none of their neighbours in the batch (roots: {np.mean([x['truncated_share_roots'] for x in r]):.1%}); "
              f"batch {np.mean([x['batch_nodes'] for x in r]):,.0f} nodes for {r[0]['roots']} roots, fanout {r[0]['fanout']}.\n")
    hl = load_cells("E5halo")
    if hl:
        p("**Halo vs roots-only under structural-only drift**: full(halo) − full(roots), and each − frozen.\n")
        p("| variant | full(roots) − frozen | full(halo) − frozen | halo − roots |")
        p("|---|---|---|---|")
        for v in ("cora_struct", "arxiv_struct"):
            dr, dh, dd = [], [], []
            for m in C.MODELS:
                a, b = hl.get((m, v, "")), hl.get((m, v, "_halo"))
                if not a:
                    continue
                dr += paired(a, "full", "baseline")
                if b:
                    F0, H, B0 = by(a["rows"], "full"), by(b["rows"], "full"), by(a["rows"], "baseline")
                    for s in sorted(set(F0) & set(H) & set(B0)):
                        dh.append(100 * (H[s]["accuracy"] - B0[s]["accuracy"]))
                        dd.append(100 * (H[s]["accuracy"] - F0[s]["accuracy"]))
            if dr:
                p(f"| {v} | {ci(dr)} | {ci(dh)} | {ci(dd)} |")
        p("")
    sy, e2c = load_cells("E5sym"), load_cells("E2")
    if sy:
        a = [x for (m, v, t), c in sy.items() for x in paired(c, "full", "baseline")]
        b = [x for (m, v, t), c in e2c.items() if v == "arxiv_year" and t == "" for x in paired(c, "full", "baseline")]
        p(f"**Warmup graph**: Arxiv-by-year full − frozen with a one-direction warmup graph {ci(a)} vs symmetric {ci(b)}.\n")
    if e2c:
        inj = [v for v in C.VARIANTS if v in ("arxiv", "cora", "pubmed", "photo", "cora_label", "pubmed_label", "arxiv_label")]
        nat = ["arxiv_year", "elliptic"]
        def count(vs, positive):
            tot = hit = 0
            for (m, v, t), c in e2c.items():
                if v in vs and t == "":
                    d = paired(c, "full", "baseline")
                    if len(d) < 3:
                        continue
                    tot += 1
                    pv = sig(d)
                    good = pv is not None and pv < 0.05 and np.mean(d) > 0
                    hit += good if positive else (not good)
            return hit, tot
        g, h = count(inj, False)
        k1, k2 = count(nat, True)
        p(f"**Injected vs natural** (`\\res{{g,h,k}}`): retraining fails to beat the frozen model in {g} of {h} injected-drift cells; "
          f"it beats it significantly in {k1} of {k2} natural-drift cells (paired t, 5 seeds, p < 0.05).\n")


def e7():
    files = sorted(glob.glob(str(RES / "E7" / "*.json")))
    if not files:
        return
    p("## E7. Inference latency while a repair runs on the same GPU\n")
    p("| model | condition | infer p50 ms | p99 ms | p99 / idle p99 | repair step p50 ms | repair nodes |")
    p("|---|---|---|---|---|---|---|")
    for f in files:
        d = json.loads(Path(f).read_text())
        conds = defaultdict(list)
        for s, rows in d["seeds"].items():
            for r in rows:
                conds[r["condition"]].append(r)
        idle = np.mean([r["p99_ms"] for r in conds.get("idle", [])]) if conds.get("idle") else np.nan
        for c in ("idle", "full", "selective", "global"):
            rr = conds.get(c, [])
            if not rr:
                continue
            rs = [r["repair_step_p50_ms"] for r in rr if r["repair_step_p50_ms"] is not None]
            if all(r.get("repair_oom") for r in rr):
                p(f"| {d['model']} | {c} | {np.mean([r['p50_ms'] for r in rr]):.2f} | {np.mean([r['p99_ms'] for r in rr]):.2f} | — | "
                  f"OOM (the update does not fit next to serving) | {rr[0]['nodes']:,} |")
                continue
            p(f"| {d['model']} | {c} | {np.mean([r['p50_ms'] for r in rr]):.2f} | {np.mean([r['p99_ms'] for r in rr]):.2f} | "
              f"{np.mean([r['p99_ms'] for r in rr]) / idle:.2f} | {np.mean(rs) if rs else float('nan'):.1f} | "
              f"{np.mean([r['nodes'] for r in rr if r['nodes']]) if rr[0]['nodes'] else 0:,.0f} |")
    p("")


def main():
    p(f"# experiments: numbers for intro.tex {'(SMOKE — not results)' if 'smoke' in str(RES) else ''}\n")
    p("Seeds 110–114 (calibration and tuning on 100–102). Generated by `python -m experiments.agg`; see CLAIMS.md for the "
      "intro line each section backs.\n")
    hp = C.read_json(C.ARMS_HP)
    if hp:
        p(f"Baseline hyperparameters (tuned on {hp['variant']}, {', '.join(hp['models'])}, seeds {hp['seeds']}): `{json.dumps(hp['hp'])}`\n")
    for f in (summary, e1, e2, e3a, e4, e5, e7):
        try:
            f()
        except Exception as ex:   # one broken section must not hide the others
            p(f"\n> {f.__name__} failed: {type(ex).__name__}: {ex}\n")
    out = RES / "intro_numbers.md"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text("\n".join(L) + "\n")
    print(f"wrote {out} ({len(L)} lines)")


if __name__ == "__main__":
    main()
