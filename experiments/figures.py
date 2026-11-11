"""The introduction's figure: three panels, one per gap, from experiments's result files (seeds 110-114).

    python -m experiments.figures      -> results/figures/intro_gaps.{pdf,png}

(a) Cost of one update, nodes materialised relative to sampled full retraining at the same budget (E2, mean over the
    streams each method ran on). Gap 2 / Gap 1 (whole-graph retraining).
(b) Gradient error of the same repair update vs the exact full-graph gradient (E4 part 1, float64, median and
    interquartile range over 4 backbones x 3 label-drift graphs x 5 seeds). Gap 3.
(c) Does retraining pay? A colour-coded table of sampled full retraining minus the never-retrained model (pp), per
    stream (rows, grouped by how the drift arises) and backbone (columns; "All" pools 4 backbones x 5 seeds), mean over
    seeds, * paired t p < 0.05 (E2). Diverging blue (helps) / red (hurts), gray at 0. Gap 4.
Colours (validated, dataviz reference palette, light): blue = GIST, orange = prior methods, aqua = whole-graph
retraining. The same in every panel.
"""
from __future__ import annotations

import glob
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy import stats

from . import common as C
from .agg import load_cells

E3A_REGION = 1                    # % of the nodes the drift touches in panel (b) of the plain figure
REPORT_SEEDS = (110, 111, 112, 113, 114)
OURS, PRIOR, WHOLE = "#2a78d6", "#eb6834", "#1baf7a"
DIV_POS, DIV_MID, DIV_NEG = "#2a78d6", "#f0efec", "#e34948"   # diverging pair (dataviz reference): gain / none / loss
INK, INK2, GRID = "#0b0b0b", "#52514e", "#e6e5e1"
plt.rcParams.update({
    "font.family": "serif", "font.serif": ["Nimbus Roman", "Liberation Serif", "DejaVu Serif"],
    "mathtext.fontset": "stix", "font.size": 7, "axes.titlesize": 7.5, "axes.labelsize": 7,
    "xtick.labelsize": 6.5, "ytick.labelsize": 6.5, "legend.fontsize": 6.5, "axes.edgecolor": INK2,
    "axes.labelcolor": INK, "xtick.color": INK2, "ytick.color": INK2, "text.color": INK, "axes.linewidth": 0.6,
    "xtick.major.width": 0.6, "ytick.major.width": 0.6, "pdf.fonttype": 42, "ps.fonttype": 42,
})


def style(ax, grid_axis="x"):
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    ax.grid(axis=grid_axis, color=GRID, linewidth=0.6, zorder=0)
    ax.set_axisbelow(True)
    ax.tick_params(length=2.5)


def cost_data():
    e2 = load_cells("E2")
    arms = [("selective", "GIST", OURS), ("full", "Full retraining", PRIOR), ("ewc", "EWC", PRIOR),
            ("twp", "TWP", PRIOR), ("ergnn", "ER-GNN", PRIOR),
            ("global_trigger", "InstantGNN, D3-GNN", WHOLE), ("roland", "ROLAND", WHOLE)]
    out = []
    for arm, label, col in arms:
        rat, oom = [], False
        for (m, v, t), c in e2.items():
            if t != "":
                continue
            A = [r["nodes"] for r in c["rows"] if r["arm"] == arm and r["events"] and not r.get("oom")]
            F = [r["nodes"] for r in c["rows"] if r["arm"] == "full" and r["events"]]
            oom |= any(r.get("oom") for r in c["rows"] if r["arm"] == arm)
            if A and F:
                rat.append(np.mean(A) / np.mean(F))
        out.append((label, float(np.mean(rat)), col, oom))
    return out


E4_METHODS = [("exact_lhop", "GIST", OURS), ("ns_25_15", "Sampling [25, 15]", PRIOR),
              ("ns_15_10", "GraphSAGE [15, 10]", PRIOR), ("ns_2_2", "Online CGL [2, 2]", PRIOR),
              ("cluster_gcn", "Cluster-GCN", PRIOR), ("saint_rw", "GraphSAINT", PRIOR), ("vrgcn", "VR-GCN", PRIOR),
              ("gas", "GNNAutoScale", PRIOR), ("one_hop", "1-hop subgraph", PRIOR)]
LABEL_GRAPHS = [("cora_label", "Cora"), ("pubmed_label", "PubMed"), ("arxiv_label", "ogbn-arxiv")]
# (c): graphs of 0.17M-3.8M nodes, at the paper's operating budget on ogbn-arxiv
EXPANSION_N = 8192
EXPANSION_GRAPHS = [("arxiv", "ogbn-arxiv"), ("web-stanford", "web-Stanford"), ("web-berkstan", "web-BerkStan"),
                    ("web-google", "web-Google"), ("youtube", "YouTube"), ("patents", "Patents")]
# (c): the E2 streams, grouped by how the drift arises
DRIFT_GROUPS = [("Natural drift", [("arxiv_year", "ogbn-arxiv (year)"), ("elliptic", "Elliptic")]),
                ("Injected drift", [("arxiv", "ogbn-arxiv"), ("cora", "Cora"), ("pubmed", "PubMed")]),
                ("Label-only drift", [("arxiv_label", "ogbn-arxiv"), ("cora_label", "Cora"), ("pubmed_label", "PubMed")])]
GROUP_SHORT = {"Natural drift": "Natural", "Injected drift": "Injected", "Label-only drift": "Labels only"}


def e4_records():
    recs = []
    for f in sorted(glob.glob(str(C.RESULTS / "E4" / "*.json"))):
        if f.endswith("_cal.json"):
            continue
        d = json.loads(Path(f).read_text())
        for s, r in d["seeds"].items():
            recs.append((d["variant"], d["model"], int(s), r))
    return recs


def main(c_kind="table", name="intro_gaps"):
    cost = cost_data()
    recs = e4_records()
    fig = plt.figure(figsize=(7.16, 2.0))
    # fixed panel boxes (figure fractions): label gutters sized to the tick labels, no collisions
    if c_kind == "matrix":      # the capability matrix needs a wider panel (c)
        axA = fig.add_axes([0.133, 0.205, 0.145, 0.66])
        axB = fig.add_axes([0.432, 0.205, 0.152, 0.66])
        axC = fig.add_axes([0.762, 0.205, 0.235, 0.53])
    else:
        axA = fig.add_axes([0.140, 0.205, 0.165, 0.66])
        axB = fig.add_axes([0.452, 0.205, 0.188, 0.66])
        axC = fig.add_axes([0.800, 0.235, 0.195, 0.56])

    # (a) cost of one update
    ax = axA
    style(ax)
    y = np.arange(len(cost))[::-1]
    for yi, (label, val, col, oom) in zip(y, cost):
        ax.barh(yi, val, height=0.64, color=col, edgecolor="white", linewidth=0.8, zorder=3)
        ax.text(max(val, 1.0) + 0.25, yi, f"{val:.2f}×" + ("$^{\dagger}$" if oom else ""), va="center", ha="left",
                fontsize=6.3, color=INK)
    ax.axvline(1.0, color=INK2, linewidth=0.6, linestyle=(0, (2, 2)), zorder=4)
    ax.set_yticks(y, [c[0] for c in cost])
    ax.get_yticklabels()[0].set_fontweight("bold")
    ax.set_xlim(0, 11.8)
    ax.set_xticks([0, 1, 5, 10])
    ax.set_xlabel("Nodes per update (× full retraining)")
    ax.set_title("(a) Cost of one update", loc="left", pad=4)

    # (b) gradient error of the same update
    ax = axB
    style(ax)
    ax.set_xscale("log")
    yb = np.arange(len(E4_METHODS))[::-1]
    for yi, (key, label, col) in zip(yb, E4_METHODS):
        v = np.clip(np.array([r["part1"][key]["rel_err"] for _, _, _, r in recs]), 1e-17, None)
        med, q1, q3 = np.median(v), np.percentile(v, 25), np.percentile(v, 75)
        ax.plot([1e-17, med], [yi, yi], color=col, linewidth=0.8, alpha=0.3, zorder=2)
        ax.plot([q1, q3], [yi, yi], color=col, linewidth=2.0, solid_capstyle="round", zorder=3)
        ax.scatter([med], [yi], s=20, color=col, edgecolor="white", linewidth=0.7, zorder=4)
        lab = (f"{med:.0e}".replace("e-", "e\N{MINUS SIGN}") if med < 1e-6 else f"{med:.1%}" if med < 0.1 else f"{med:.0%}")
        ax.text(med * 12 if med < 1e-6 else q3 * 2.6, yi, lab, va="center", ha="left", fontsize=6.2, color=INK)
    ax.axvspan(1e-17, 1e-12, color=OURS, alpha=0.08, zorder=0, linewidth=0)
    ax.text(2.2e-17, -0.48, "exact", fontsize=6, color=INK2, va="center", style="italic")
    ax.set_yticks(yb, [m[1] for m in E4_METHODS])
    ax.get_yticklabels()[0].set_fontweight("bold")
    ax.set_xlim(1e-17, 8)
    ax.set_ylim(-0.75, len(E4_METHODS) - 0.4)
    ax.set_xticks([1e-16, 1e-12, 1e-8, 1e-4, 1])
    ax.set_xlabel("Gradient error vs full-graph gradient")
    ax.set_title("(b) Gradient of the same repair", loc="left", pad=4)

    if c_kind == "matrix":
        capability_matrix(axC, cost, recs)
    else:
        retraining_table(axC)
    # one title row for the figure: every panel title at the same height, left-aligned with its plot area
    for ax in (axA, axB, axC):
        t = ax.get_title(loc="left")
        ax.set_title("", loc="left")
        fig.text(ax.get_position().x0, 0.975, t, ha="left", va="top", fontsize=plt.rcParams["axes.titlesize"], color=INK)
    out = C.RESULTS / "figures"
    out.mkdir(parents=True, exist_ok=True)
    fig.savefig(out / f"{name}.pdf")
    fig.savefig(out / f"{name}.png", dpi=300)
    print("cost:", [(c[0], round(c[1], 2), c[3]) for c in cost])
    print("wrote", out / f"{name}.pdf")


def retraining_table(ax):
    # (c) does retraining pay? A colour-coded table: full - no retraining (pp) per stream x backbone (E2): Gap 4
    from matplotlib.colors import LinearSegmentedColormap, TwoSlopeNorm
    cmap = LinearSegmentedColormap.from_list("div", [DIV_NEG, "#f6c9c6", DIV_MID, "#b7d3f6", DIV_POS])
    # colour shows the direction; the printed numbers carry the size. Gains are small (<= 1.7 pp), so the gain arm
    # saturates at +1 and the loss arm at -3 (two slopes), which keeps every gain visibly blue.
    lo, hi = -3.0, 1.0
    norm = TwoSlopeNorm(vmin=lo, vcenter=0.0, vmax=hi)
    e2 = load_cells("E2")
    cols = ["GAT", "SAGE", "GCN", "GIN", "All"]
    cells, sig, labels, groups = [], [], [], []
    for group, members in DRIFT_GROUPS:
        for v, label in members:
            row, srow, pooled = [], [], []
            for m in C.MODELS:
                c = e2.get((m, v, ""))
                A = {r["seed"]: r["accuracy"] for r in c["rows"] if r["arm"] == "full"}
                B = {r["seed"]: r["accuracy"] for r in c["rows"] if r["arm"] == "baseline"}
                d = np.array([100 * (A[s_] - B[s_]) for s_ in sorted(set(A) & set(B))])
                pooled += list(d)
                row.append(d.mean())
                srow.append(stats.ttest_1samp(d, 0).pvalue < 0.05)
            pooled = np.array(pooled)
            row.append(pooled.mean())
            srow.append(stats.ttest_1samp(pooled, 0).pvalue < 0.05)
            cells.append(row); sig.append(srow); labels.append(label); groups.append(group)
    cells = np.array(cells)
    # rows top to bottom with a small gap between drift groups
    ypos, y, prev = [], 0.0, None
    for g in groups:
        if prev is not None and g != prev:
            y += 0.35
        ypos.append(y); y += 1.0; prev = g
    ypos = np.array(ypos)
    for i, yi in enumerate(ypos):
        for j in range(len(cols)):
            val = cells[i, j]
            xj = j + (0.25 if j == len(cols) - 1 else 0.0)           # the pooled column stands a little apart
            face = cmap(norm(np.clip(val, lo, hi)))
            ax.add_patch(plt.Rectangle((xj - 0.47, yi - 0.45), 0.94, 0.9, facecolor=face, edgecolor="white",
                                       linewidth=1.0, zorder=2))
            lum = 0.2126 * face[0] + 0.7152 * face[1] + 0.0722 * face[2]   # text follows the cell's brightness
            dark = lum < 0.55
            txt = f"{val:+.1f}".replace("-", "\N{MINUS SIGN}") + ("*" if sig[i][j] else "")
            ax.text(xj, yi, txt, ha="center", va="center", fontsize=5.9, color="white" if dark else INK,
                    fontweight="bold" if j == len(cols) - 1 else "normal", zorder=3)
    ax.set_xlim(-0.55, len(cols) - 1 + 0.25 + 0.55)
    ax.set_ylim(ypos[-1] + 0.6, -0.95)
    ax.set_xticks([j + (0.25 if j == len(cols) - 1 else 0) for j in range(len(cols))], cols)
    ax.xaxis.tick_top()
    ax.tick_params(axis="x", length=0, pad=1.5, labelsize=6.2)
    ax.set_yticks(ypos, labels)
    ax.tick_params(axis="y", length=0, pad=2, labelsize=6.2)
    for sp in ax.spines.values():
        sp.set_visible(False)
    # group labels: a coloured bracket left of the row names
    for g in dict.fromkeys(groups):
        idx = [i for i, gg in enumerate(groups) if gg == g]
        y0g, y1g = ypos[idx[0]] - 0.42, ypos[idx[-1]] + 0.42
        ax.annotate("", xy=(-0.56, y0g), xytext=(-0.56, y1g), xycoords=("axes fraction", "data"),
                    arrowprops=dict(arrowstyle="-", color=INK2, linewidth=0.8))
        ax.text(-0.6, (y0g + y1g) / 2, GROUP_SHORT[g], transform=ax.get_yaxis_transform(), rotation=90,
                ha="right", va="center", fontsize=6.1, color=INK, fontweight="bold")
    ax.set_title("(c) Does retraining pay?", loc="left", pad=11)
    # key: what the colours mean in this panel
    for k, (col, txt) in enumerate(((DIV_POS, "retraining helps"), (DIV_NEG, "retraining hurts"))):
        kx = 0.02 + 0.52 * k
        ax.add_patch(plt.Rectangle((kx, -0.135), 0.07, 0.045, transform=ax.transAxes, facecolor=col,
                                   edgecolor="none", clip_on=False))
        ax.text(kx + 0.09, -0.112, txt, transform=ax.transAxes, fontsize=6.0, va="center", ha="left", color=INK)
    ax.text(0.02, -0.2, "pp vs no retraining;  * p < 0.05 over 5 seeds", transform=ax.transAxes, fontsize=5.6,
            color=INK2, va="center", ha="left", style="italic")



GOOD, BAD, NA = "#0ca30c", "#d03b3b", "#a3a29c"      # dataviz status palette: good / critical; muted for n/a


def capability_matrix(ax, cost, recs):
    """(c) for intro_gaps_2: which method families meet each requirement of a repair, with the measured value in each
    cell (E2 cost, E4 gradient error and drifted-node accuracy, E2/E3a out-of-memory). GIST is the only row with
    every requirement met. Pass/fail glyphs carry the meaning; colour repeats it (status palette)."""
    med = lambda key: float(np.median([r["part1"][key]["rel_err"] for _, _, _, r in recs]))
    drift = lambda key: 100 * float(np.mean([(r["frozen"]["acc_drifted"] if key == "frozen" else r["part2"][key]["20"]["acc_drifted"])
                                             for _, _, _, r in recs]))
    c = {k: v for k, v, _, _ in cost}
    approx = [med(k) for k in ("ns_15_10", "ns_2_2", "cluster_gcn", "saint_rw", "vrgcn", "gas")]
    rep_keys = [k for k in ("ns_15_10", "ns_2_2", "cluster_gcn", "saint_rw", "gas@1", "vrgcn@1", "iglu@1", "lmc@10")
                if all(k in r["part2"] for _, _, _, r in recs)]
    rep_approx = [drift(k) for k in rep_keys]
    whole = [c["InstantGNN, D3-GNN"], c["ROLAND"]]
    cgl = [c["EWC"], c["TWP"], c["ER-GNN"]]
    cols = ["Repairs\ndrift", "Exact\ngradient", "Cheaper\nthan full"]
    rows = [
        ("Incremental inference", [(False, f"{drift('frozen'):.1f}% acc."), (None, "no update"), (None, "no update")]),
        ("Whole-graph retraining", [(True, ""), (True, "full graph"), (False, f"{min(whole):.1f}–{max(whole):.1f}×")]),
        ("Continual learning", [(True, ""), (False, "sampled"), (False, f"{min(cgl):.2f}–{max(cgl):.2f}×")]),
        ("Sampling / history", [(True, f"{min(rep_approx):.0f}–{max(rep_approx):.0f}% acc."), (False, f"{min(approx):.0%}–{max(approx):.0%} err."), (True, "bounded")]),
        ("Full retraining", [(True, ""), (False, f"{med('ns_25_15'):.1%} err."), (False, "1.00×")]),
        ("GIST", [(True, f"{drift('exact_lhop'):.0f}% acc."), (True, f"{med('exact_lhop'):.0e} err.".replace("e-", "e\N{MINUS SIGN}")),
                         (True, f"{c['GIST']:.2f}×")]),
    ]
    nr, nc = len(rows), len(cols)
    ax.set_xlim(-0.5, nc - 0.5)
    ax.set_ylim(nr - 0.5, -0.5)
    ax.add_patch(plt.Rectangle((-0.5, nr - 1.5), nc, 1.0, facecolor=OURS, alpha=0.12, edgecolor="none", zorder=0))
    for i in range(nr + 1):
        ax.axhline(i - 0.5, color=GRID, linewidth=0.6, zorder=1)
    for i, (label, cells) in enumerate(rows):
        for j, (ok, note) in enumerate(cells):
            glyph, col = ("✓", GOOD) if ok else ("–", NA) if ok is None else ("✗", BAD)
            ax.text(j, i - (0.13 if note else 0), glyph, ha="center", va="center", fontsize=9.5, color=col,
                    fontfamily="DejaVu Sans", fontweight="bold", zorder=3)
            if note:
                ax.text(j, i + 0.27, note, ha="center", va="center", fontsize=5.0, color=INK2, zorder=3)
    ax.set_xticks(range(nc), cols)
    ax.xaxis.tick_top()
    ax.tick_params(axis="x", length=0, pad=1.5, labelsize=6.0)
    ax.set_yticks(range(nr), [r[0] for r in rows])
    ax.tick_params(axis="y", length=0, pad=2, labelsize=6.2)
    ax.get_yticklabels()[-1].set_fontweight("bold")
    for sp in ax.spines.values():
        sp.set_visible(False)
    ax.set_title("(c) What each kind of update achieves", loc="left")


def plain_figure(name="intro_gaps"):
    """The introduction's figure in common metrics (after review): each panel names WHO falls short, shows it in
    accuracy / time / changed predictions, and says WHY in one line. Data: E4 (label drift, 4 backbones x 3 graphs x
    5 seeds), E3a (time of one weight update, 4 backbones on ogbn-arxiv, 5 seeds) and E7 (inference slowdown while an
    update runs, GAT and SAGE on ogbn-arxiv)."""
    recs = e4_records()
    fig = plt.figure(figsize=(7.16, 2.35))
    axA = fig.add_axes([0.070, 0.20, 0.205, 0.52])
    axB = fig.add_axes([0.385, 0.20, 0.235, 0.52])
    axC = fig.add_axes([0.800, 0.20, 0.195, 0.52])
    TITLE_X = {"a": 0.012, "b": 0.330, "c": 0.655}     # each title starts at its panel's left edge, labels included

    def head(panel, title, why):
        fig.text(TITLE_X[panel], 0.975, title, ha="left", va="top", fontsize=7.5, color=INK)
        fig.text(TITLE_X[panel], 0.905, why, ha="left", va="top", fontsize=6.2, color=INK2, style="italic", linespacing=1.15)

    # (a) accuracy on drifted nodes: fixed weights vs GIST's update
    ax = axA
    style(ax, "y")
    groups = [("frozen", "Fixed weights\n(InkStream, Ripple,\nStreamTGN)", PRIOR), ("exact_lhop", "After a\nGIST\nupdate", OURS)]
    for xi, (key, label, col) in enumerate(groups):
        per = []
        for v, gname in LABEL_GRAPHS:
            vals = [100 * (r["frozen"]["acc_drifted"] if key == "frozen" else r["part2"][key]["20"]["acc_drifted"])
                    for vv, _, _, r in recs if vv == v]
            per.append(np.mean(vals))
        m = float(np.mean(per))
        ax.bar(xi, m, width=0.58, color=col, edgecolor="white", linewidth=0.8, zorder=3)
        ax.text(xi, m + 3, f"{m:.0f}%" if m >= 10 else f"{m:.1f}%", ha="center", va="bottom", fontsize=6.6, color=INK)
    ax.set_xticks([0, 1], [g[1] for g in groups], fontsize=6.0)
    ax.set_ylim(0, 75)
    ax.set_yticks([0, 25, 50, 75])
    ax.set_ylabel("Correct on drifted nodes (%)")
    ax.tick_params(axis="x", length=0)
    head("a", "(a) Refreshing embeddings does not fix drift", "the weights never change,\nso drifted nodes stay wrong")

    # (b) time of one weight update per backbone on ogbn-arxiv (E3a, whole graph, drift touching E3A_REGION % of the
    # nodes), log scale; the subtitle carries E7's inference slowdown during an update (GAT and GraphSAGE)
    ax = axB
    style(ax, "x")
    ax.set_xscale("log")
    slow = []
    for f in sorted(glob.glob(str(C.RESULTS / "E7" / "*.json"))):
        d = json.loads(Path(f).read_text())
        for rows in d["seeds"].values():
            idle = next(r["p99_ms"] for r in rows if r["condition"] == "idle")
            slow += [r["p99_ms"] / idle for r in rows if r["condition"] in ("full", "selective") and not r["repair_oom"]]
    e3 = {}
    for f in sorted(glob.glob(str(C.RESULTS / "E3a" / "arxiv_*.json"))):
        d = json.loads(Path(f).read_text())
        for r in d["rows"]:
            if r["fraction"] == 1.0 and r["seed"] in REPORT_SEEDS and r["region_pct"] in (None, E3A_REGION):
                e3.setdefault((d["model"], r["method"]), []).append(r)
    models = [("gcn", "GCN"), ("sage", "GraphSAGE"), ("gin", "GIN"), ("gat", "GAT")]
    bars = [("global", WHOLE), ("full", PRIOR), ("selective", OURS)]
    h = 0.27                                             # bar height; one backbone per unit, bars top to bottom
    for gi, (m, mlabel) in enumerate(models):
        for bi, (meth, col) in enumerate(bars):
            rr = e3[(m, meth)]
            y = -gi - (bi - 1) * h
            if all(r["oom"] for r in rr):
                ax.text(1.12, y, "✗ out of memory (48 GB)", va="center", ha="left", fontsize=5.6, color=INK)
                continue
            v = float(np.mean([r["ms"] for r in rr]))
            ax.barh(y, v - 1.0, left=1.0, height=h * 0.86, color=col, edgecolor="white", linewidth=0.6, zorder=3)
            ax.text(v * 1.1, y, f"{v:.0f}" if v >= 20 else f"{v:.1f}", va="center", ha="left", fontsize=5.6, color=INK)
    ax.set_yticks([-gi for gi in range(len(models))], [m[1] for m in models], fontsize=6.6)
    ax.tick_params(axis="y", length=0)
    ax.set_ylim(-(len(models) - 1) - 0.5, 0.5)
    ax.set_xlim(1.0, 3000)
    ax.set_xticks([1, 10, 100, 1000], ["1", "10", "100", "1000"])
    ax.set_xlabel("Time per weight update (ms, log)")
    hd = [plt.Rectangle((0, 0), 1, 1, color=c_) for _, c_ in bars]
    ax.legend(hd, ["Whole graph\n(InstantGNN, D3-GNN)", "Full retraining\n(EWC, TWP, ER-GNN)", "GIST"],
              loc="upper right", bbox_to_anchor=(1.02, 1.04), frameon=False, fontsize=5.5, handlelength=0.8,
              handleheight=0.7, borderaxespad=0.1, labelspacing=0.35)
    head("b", "(b) Updating the weights is slow", f"each update reprocesses every retrained node's neighbourhood,\n"
              f"and inference runs {min(slow):.1f}–{max(slow):.1f}× slower while it does")

    # (c) changed predictions after repair, vs the exact update (E4 part 2, 20 steps)
    ax = axC
    style(ax)
    meths = [("ns_25_15", "Neighbour sampling"), ("vrgcn@1", "VR-GCN"), ("ns_15_10", "GraphSAGE sampling"),
             ("cluster_gcn", "Cluster-GCN"), ("saint_rw", "GraphSAINT"), ("iglu@1", "IGLU"), ("gas@1", "GNNAutoScale"),
             ("ns_2_2", "Online CGL sampling"), ("exact_lhop", "GIST")]
    vals = []
    for key, label in meths:
        v = [100 * r["part2"][key]["20"]["disagree_exact"] for _, _, _, r in recs if key in r["part2"]]
        vals.append(float(np.mean(v)) if v else float("nan"))
    order = list(np.argsort(vals))[::-1]
    y = np.arange(len(meths))[::-1]
    for yi, k in zip(y, sorted(range(len(meths)), key=lambda i: (meths[i][0] == "exact_lhop", -vals[i]))):
        key, label = meths[k]
        col = OURS if key == "exact_lhop" else PRIOR
        ax.barh(yi, max(vals[k], 0.0), height=0.62, color=col, edgecolor="white", linewidth=0.8, zorder=3)
        ax.text(max(vals[k], 0) + 0.25, yi, "0 (exact)" if key == "exact_lhop" else f"{vals[k]:.1f}%", va="center",
                fontsize=6.0, color=INK)
        ax.text(-0.3, yi, label, va="center", ha="right", fontsize=6.3, color=INK,
                fontweight="bold" if key == "exact_lhop" else "normal")
    ax.set_yticks([])
    ax.set_xlim(0, 13)
    ax.set_xticks([0, 5, 10])
    ax.set_xlabel("Predictions that differ from\nthe exact update (%)")
    head("c", "(c) Faster training shortcuts change the result", "they approximate the update\ninstead of computing it")

    out = C.RESULTS / "figures"
    out.mkdir(parents=True, exist_ok=True)
    fig.savefig(out / f"{name}.pdf")
    fig.savefig(out / f"{name}.png", dpi=300)
    print("wrote", out / f"{name}.pdf", "| serving slowdown", [round(x, 2) for x in slow])


if __name__ == "__main__":
    import sys
    if "--matrix" in sys.argv:
        main("matrix", "intro_gaps_2")
    elif "--plain" in sys.argv:          # the reviewed version (common metrics; who + why per panel)
        plain_figure("intro_gaps_3")
    else:
        main()
