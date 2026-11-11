"""Intro figures of the GIST ICDE paper (Sec. 1.1), from the Phase D raw results (seeds 0-9). Reads only; never trains.

    OMP_NUM_THREADS=8 CUDA_VISIBLE_DEVICES= python analysis/make_intro_figures.py

intro_runtime   fig:insight-time. One repair on the Arxiv-year stream (natural drift, N = 8192, 10 seeds), D1:
                time per repair split into building the update batch and the rest of the repair (bars), and peak GPU
                memory of the repair (violet line, right axis), GAT and GraphSAGE, for every prior rule that updates
                the parameters.
intro_fidelity  fig:insight-exact. The same repair computed by each cost-bounding rule, E4 (label drift on Cora, PubMed,
                ogbn-arxiv; 4 backbones; 10 seeds): nodes it materialises relative to the exact L-hop subgraph of the
                retrained nodes (left), its gradient error and the share of predictions that differ from the exact
                repair after 20 steps (right). Same statistics as tab:fidelity (median error and node ratio, mean
                disagreement).
Writes results/figures/intro_runtime.{pdf,png}, intro_fidelity.{pdf,png} (600 dpi) and intro_values.md (every
plotted value). Style: the same fonts, sizes and colours as fig_eval.py / fig_speed.py.
"""
from __future__ import annotations

import glob
import json
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager
from matplotlib.lines import Line2D
from matplotlib.patches import Patch

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "results" / "figures" / "intro"
D1 = ROOT / "results" / "D1_main" / "raw" / "streams" / "D1"
E4 = ROOT / "results" / "D5_intro" / "raw" / "E4"
E2 = ROOT / "results" / "D5_intro" / "raw" / "streams" / "E2"
SEEDS = set(range(10))

NAVY, GREY, VERM, INK = "#0B2D5B", "#7A7A7A", "#D55E00", "#444444"
NAVY_T, GREY_T, VERM_T = "#9FB3CF", "#D2D2D2", "#F3C9A8"
VIOLET = "#7B3FA0"
# GIST in teal green (was navy), memory line in dark red (was violet), the "drops context" bars of
# Fig. 2 in leaf green (was orange); grey bars and the OOM marks unchanged.
GIST_G, GIST_GT, MEM_R = "#145A3C", "#52B788", "#C0392B"   # GIST in Fig. 1: deep green (rest of repair, labels) + fresh green (build batch) (2026-11-07)
GIST_F = "#145A3C"                                        # GIST in Fig. 2: the same deep green (2026-11-07)
DROP_G, DROP_GT = "#3A7A2C", "#CFE8C6"
SERIF = next((f for f in ("Times New Roman", "Nimbus Roman", "Nimbus Roman No9 L", "Liberation Serif")
              if any(e.name == f for e in font_manager.fontManager.ttflist)), "DejaVu Serif")
plt.rcParams.update({
    "font.family": SERIF, "font.size": 7, "axes.titlesize": 7, "axes.labelsize": 7, "xtick.labelsize": 6.5,
    "ytick.labelsize": 6.5, "legend.fontsize": 6.5, "pdf.fonttype": 42, "ps.fonttype": 42, "axes.linewidth": 0.5,
    "xtick.major.width": 0.5, "ytick.major.width": 0.5, "xtick.major.size": 2, "ytick.major.size": 2,
    "xtick.minor.size": 0, "ytick.minor.size": 1, "ytick.minor.width": 0.3,
    "axes.spines.top": False, "axes.spines.right": False, "axes.grid": True, "axes.axisbelow": True,
    "grid.linewidth": 0.3, "grid.color": "#DDDDDD", "mathtext.fontset": "stix",
    "hatch.linewidth": 0.4,
})
LOG = []
VAL = 5.6          # size of the value labels on the bars


def load(p):
    return json.loads(Path(p).read_text())


def save(fig, name):
    OUT.mkdir(exist_ok=True)
    fig.savefig(OUT / f"{name}.pdf")
    fig.savefig(OUT / f"{name}.png", dpi=600)
    plt.close(fig)


def check_layout(fig):
    """Fail on text that leaves the canvas or overlaps other text."""
    fig.canvas.draw()
    r = fig.canvas.get_renderer()
    items = []
    for ax in fig.axes:
        items += [t for t in ax.texts if t.get_text() and t.get_visible()]
        xt = [] if getattr(ax, "_twin_of", None) is not None else ax.get_xticklabels()   # a twin repeats the x labels
        yt = [t for t in ax.get_yticklabels() if ax.yaxis.get_ticks_position() != "unknown"]
        items += [t for t in xt + yt if t.get_text() and t.get_visible()]
        items += [ax.xaxis.label, ax.yaxis.label, ax.title]
    items += [t for t in fig.texts if t.get_text()]
    for lg in fig.legends:
        items += lg.get_texts()
    boxes = []
    for t in items:
        if not t.get_text():
            continue
        b = t.get_window_extent(r)
        pad = 0.15 * t.get_fontsize() * fig.dpi / 72
        boxes.append((t.get_text(), matplotlib.transforms.Bbox.from_extents(b.x0 + .5, b.y0 + pad, b.x1 - .5, b.y1 - pad),
                      t.get_rotation() != 0))
    W, H = fig.bbox.width, fig.bbox.height
    bad = [f"'{s}' leaves the canvas" for s, b, _ in boxes if b.x0 < -1 or b.y0 < -1 or b.x1 > W + 1 or b.y1 > H + 1]
    for i in range(len(boxes)):
        for j in range(i + 1, len(boxes)):
            if boxes[i][2] and boxes[j][2]:      # neighbouring rotated tick labels: their boxes are not their glyphs
                continue
            if boxes[i][1].overlaps(boxes[j][1]):
                bad.append(f"'{boxes[i][0]}' overlaps '{boxes[j][0]}'")
    if bad:
        raise SystemExit("layout: " + "; ".join(bad))


# --- Fig. A: time and memory of one repair ---------------------------------------------------------------------------
# (label, D1 arms pooled, family). EWC and TWP add a penalty to Full's sampled step; InstantGNN (trigger) and D3-GNN
# (timer) run the same whole-graph step. CaT is left out until its implementation is checked.
RULES = [
    ("Full", ("full",), "prior"),
    ("ER-GNN", ("ergnn",), "prior"),
    ("EWC/TWP", ("ewc", "twp"), "prior"),
    ("ROLAND", ("roland",), "prior"),
    ("Whole graph", ("global_trigger", "global_periodic"), "prior"),
    ("GIST", ("selective",), "gist"),
]
MODELS = (("gat", "GAT"), ("sage", "GraphSAGE"))


def runtime_data(model, stream="arxiv_year"):
    d = load(D1 / f"{model}_{stream}.json")
    rows = [r for r in d["rows"] if r["seed"] in SEEDS]
    out = []
    for label, arms, fam in RULES:
        rs = [r for r in rows if r["arm"] in arms]
        ok = [r for r in rs if not r["oom"] and r["events"] > 0]
        n_oom = len({(r["arm"], r["seed"]) for r in rs if r["oom"]})
        rec = dict(label=label, fam=fam, arms=arms, n=len(ok), n_oom=n_oom, n_rows=len(rs))
        if ok:
            tot = np.array([r["retrain_p50_ms"] for r in ok])
            host = np.minimum(np.array([r["region_sample_ms"] for r in ok]), tot)
            mem = np.array([r["retrain_peak_mb"] for r in ok]) / 1024
            nodes = np.array([r["nodes"] for r in ok])
            rec.update(total=float(np.median(tot)), host=float(np.median(host)), mem=float(np.median(mem)),
                       nodes=float(np.median(nodes)), total_rng=(float(tot.min()), float(tot.max())),
                       mem_rng=(float(mem.min()), float(mem.max())))
        out.append(rec)
    return out


def runtime():
    fig, axes = plt.subplots(1, 2, figsize=(3.45, 1.6),
                             gridspec_kw=dict(wspace=.16, left=.115, right=.9, top=.805, bottom=.2))
    LOG.append("## intro_runtime (fig:insight-time): Arxiv-year, N = 8192, D1 seeds 0-9\n")
    LOG.append("Time = median over seeds of the per-run retrain p50 (ms); host part = median of the per-run mean time "
               "to build the update batch (`region_sample_ms`: neighbour sampling of the roots; ER-GNN's replay batch "
               "is in the rest), capped at the total; memory = median `retrain_peak_mb` (violet line, right axis). OOM = "
               "the rule ran out of GPU memory during the stream on that many of the 10 seeds; its bar and memory point "
               "are not drawn when every seed failed.\n")
    LOG.append("| backbone | rule | arms | time ms (min-max) | host ms | rest ms | host share | peak GB | nodes/update | OOM seeds |")
    LOG.append("|---|---|---|---|---|---|---|---|---|---|")
    x = np.arange(len(RULES))
    w = .62
    for j, (m, name) in enumerate(MODELS):
        data = runtime_data(m)
        at = axes[j]
        am = at.twinx()
        am._twin_of = at
        mem = []
        for i, rec in enumerate(data):
            dark, light = (GIST_G, GIST_GT) if rec["fam"] == "gist" else (GREY, GREY_T)
            if "total" not in rec:
                at.text(i, .03, "OOM", transform=at.get_xaxis_transform(), rotation=90, ha="center", va="bottom",
                        color=VERM, fontsize=6.2, fontweight="bold")
                mem.append(np.nan)
                LOG.append(f"| {name} | {rec['label'].replace(chr(10), ' ')} | {', '.join(rec['arms'])} | OOM | | | | | | "
                           f"{rec['n_oom']}/{rec['n_rows']} runs |")
                continue
            host, rest = rec["host"], rec["total"] - rec["host"]
            at.bar(i, host, w, color=light, edgecolor=dark, hatch="////", lw=.5, zorder=3)
            at.bar(i, rest, w, bottom=host, color=dark, edgecolor=dark, lw=.5, zorder=3)
            at.text(i, rec["total"] * 1.12, f"{rec['total']:.0f}", ha="center",
                    va="bottom", fontsize=VAL, color=GIST_G if rec["fam"] == "gist" else INK,
                    fontweight="bold" if rec["fam"] == "gist" else "normal")
            mem.append(rec["mem"])
            LOG.append(f"| {name} | {rec['label'].replace(chr(10), ' ')} | {', '.join(rec['arms'])} | {rec['total']:.0f} "
                       f"({rec['total_rng'][0]:.0f}-{rec['total_rng'][1]:.0f}) | {host:.0f} | {rest:.0f} | "
                       f"{host / rec['total']:.0%} | {rec['mem']:.2f} | {rec['nodes']:.0f} | {rec['n_oom']}/{rec['n_rows']} runs |")
        # peak memory: violet line on the right axis, broken where a rule ran out of memory
        am.plot(x, mem, color=MEM_R, lw=1.0, marker="o", ms=3.2, markerfacecolor=MEM_R, markeredgecolor="white",
                markeredgewidth=.5, zorder=6, clip_on=False)
        for i, v in enumerate(mem):
            if np.isfinite(v):
                am.annotate(f"{v:.1f}" if v < 10 else f"{v:.0f}", (i, v), xytext=(0, -3.4), textcoords="offset points",
                            ha="center", va="top", fontsize=5.4, color=MEM_R, zorder=7,
                            bbox=dict(boxstyle="square,pad=0.05", facecolor="white", edgecolor="none", alpha=.85))
        am.set_yscale("log")
        # the two scales are banded: memory in the upper part of the panel, bars and their labels in the lower part
        am.set_ylim(5e-5, 120)
        am.set_yticks([.1, 1, 10, 100], ["0.1", "1", "10", "100"])
        am.minorticks_off()
        am.axhline(48, color=MEM_R, lw=.5, ls=(0, (3, 2)), zorder=2)
        am.grid(False)
        am.spines["right"].set_visible(True)
        am.spines["right"].set_color(MEM_R)
        am.tick_params(axis="y", colors=MEM_R, labelcolor=MEM_R)
        at.set_zorder(am.get_zorder() - 1)          # bars under the line
        at.patch.set_visible(False)
        if j == 1:
            am.text(len(RULES) - .45, 48 / 1.3, "48 GB GPU", ha="right", va="top", fontsize=5.4, color=MEM_R)
            am.set_ylabel("Peak memory (GB)", color=MEM_R, labelpad=2)
        else:
            am.tick_params(labelright=False)
        at.set_yscale("log")
        at.set_ylim(10, 2.5e5)
        at.set_yticks([10, 100, 1000], ["10", "100", "1K"])
        at.minorticks_off()
        at.grid(axis="x", visible=False)
        at.set_xlim(-.6, len(RULES) - .4)
        at.axvline(len(RULES) - 1.5, color="#BBBBBB", lw=.5, zorder=1)
        at.set_title(name, pad=2, fontweight="bold")
        at.set_xticks(x, [r[0] for r in RULES], rotation=30, ha="right", rotation_mode="anchor", fontsize=5.6,
                      linespacing=.85)
        at.tick_params(axis="x", pad=1)
        for t in at.get_xticklabels():
            if t.get_text() == "GIST":
                t.set_color(GIST_G)
                t.set_fontweight("bold")
        if j == 0:
            at.set_ylabel("Repair time (ms)", labelpad=1.5)
        else:
            at.tick_params(labelleft=False)
    fig.legend(handles=[Patch(facecolor=GREY_T, edgecolor=GREY, hatch="////", lw=.5, label="build update batch"),
                        Patch(facecolor=GREY, edgecolor=GREY, lw=.5, label="rest of repair"),
                        Line2D([], [], color=MEM_R, lw=1.0, marker="o", ms=3.2, markeredgecolor="white",
                               markeredgewidth=.5, label="peak memory")],
               ncol=3, loc="upper center", bbox_to_anchor=(.51, 1.015), frameon=False, handlelength=1.5,
               handleheight=.8, columnspacing=.9, handletextpad=.35)
    check_layout(fig)
    save(fig, "intro_runtime")
    LOG.append("")


# --- Fig. B: nodes materialised and fidelity of the same repair ------------------------------------------------------
# (label, E4 part-1 key for nodes and gradient error, part-2 key for the 20-step disagreement, family)
FID = [
    ("Whole graph", "whole", None, "whole"),
    ("Cluster-GCN", "cluster_gcn", "cluster_gcn", "prior"),
    ("Full (neighbor sampling)", "ns_25_15", "ns_25_15", "prior"),
    ("Online CGL", "ns_2_2", "ns_2_2", "prior"),
    ("GraphSAINT", "saint_rw", "saint_rw", "prior"),
    ("VR-GCN", "vrgcn", "vrgcn@1", "prior"),
    ("GNNAutoScale", "gas", "gas@1", "prior"),
    ("IGLU (lazy update)", "exact_lhop", "iglu@1", "stale"),
    ("GIST", "exact_lhop", "exact_lhop", "gist"),
]


def fidelity_data():
    n_nodes = {}
    for f in glob.glob(str(E2 / "*_label.json")):
        d = load(f)
        g = [r["nodes"] for r in d["rows"] if r["arm"] == "global_trigger" and not r["oom"]]
        if g:
            n_nodes[d["hyperparameters"]["variant"]] = max(g)
    acc = {lab: dict(ratio=[], err=[], dis=[]) for lab, *_ in FID}
    cells = []
    for f in sorted(glob.glob(str(E4 / "*_label.json"))):
        d = load(f)
        cells.append(f"{d['model']}/{d['variant']}")
        for s, r in d["seeds"].items():
            if int(s) not in SEEDS:
                continue
            p1, p2 = r["part1"], r["part2"]
            ex = p1["exact_lhop"]["nodes"]
            for lab, k1, k2, fam in FID:
                if fam == "whole":
                    acc[lab]["ratio"].append(n_nodes[d["variant"]] / ex)
                    continue
                acc[lab]["ratio"].append(p1[k1]["nodes"] / ex)
                if fam not in ("stale",):
                    acc[lab]["err"].append(p1[k1]["rel_err"])
                v = p2[k2]
                acc[lab]["dis"].append(v[max(v, key=int)]["disagree_exact"])
    return acc, cells, n_nodes


def fidelity():
    acc, cells, n_nodes = fidelity_data()
    LOG.append("## intro_fidelity (fig:insight-exact): E4, label drift, seeds 0-9\n")
    LOG.append(f"Cells: {', '.join(cells)} ({len(cells)}). Nodes = rule's nodes / exact 2-hop subgraph's (median; "
               f"whole graph: graph size {n_nodes} / exact), gradient error = median rel_err (max|g-g*|/max|g*|, "
               "float64, warm weights), differing = mean disagree_exact after 20 steps. Same statistics as tab:fidelity.\n")
    LOG.append("| rule | nodes / exact (median, IQR) | gradient error % (median) | differing % (mean) | n |")
    LOG.append("|---|---|---|---|---|")
    fig = plt.figure(figsize=(3.45, 1.6))
    gs = fig.add_gridspec(1, 2, width_ratios=[1.25, 1], left=.29, right=.985, top=.885, bottom=.21, wspace=.07)
    al, ar = fig.add_subplot(gs[0, 0]), fig.add_subplot(gs[0, 1])
    g_err = acc["GIST"]["err"]
    m, e = f"{max(g_err):.0e}".split("e")       # the largest of the 120 audited GIST errors, e.g. 8e-15
    gist_txt = rf"$\leq {m}\times10^{{{int(e)}}}$ / 0"
    y = np.arange(len(FID))[::-1]
    h = .62
    for yi, (lab, k1, k2, fam) in zip(y, FID):
        a = acc[lab]
        ratio = float(np.median(a["ratio"]))
        q1, q3 = np.percentile(a["ratio"], [25, 75])
        err = 100 * float(np.median(a["err"])) if a["err"] else None
        dis = 100 * float(np.mean(a["dis"])) if a["dis"] else None
        LOG.append(f"| {lab.replace(chr(10), ' ')} | {ratio:.2f} ({q1:.2f}-{q3:.2f}) | "
                   f"{'%.2e' % (err / 100) if fam == 'gist' else ('%.1f' % err if err is not None else 'stale' if fam == 'stale' else '0 (reference)')} | "
                   f"{'%.1f' % dis if dis is not None else 'not run'} | {len(a['ratio'])} |")
        # left: deviation from the nodes the gradient depends on (1x), on a log axis
        if fam == "gist":
            al.plot(1, yi, marker="D", ms=4.2, color=GIST_F, zorder=5, clip_on=False)
            al.text(1.3, yi, "1.00×, exact", va="center", ha="left", fontsize=VAL, color=GIST_F, fontweight="bold")
        elif fam == "stale":
            al.plot(1, yi, marker="D", ms=3.6, markerfacecolor="white", markeredgecolor=GREY, mew=.7, zorder=5)
            al.text(1.3, yi, "1.00×", va="center", ha="left", fontsize=VAL, color=INK)
        else:
            less = ratio < 1
            face, edge, hatch = (DROP_GT, DROP_G, "////") if less else ("white", GREY, "xxxx")
            al.barh(yi, ratio - 1, h, left=1, color=face, edgecolor=edge, hatch=hatch, lw=.5, zorder=3)
            al.plot([q1, q3], [yi, yi], color=INK, lw=.5, zorder=4, solid_capstyle="butt")
            tx = (q1 / 1.12, "right") if less else (q3 * 1.12, "left")
            al.text(tx[0], yi, f"{ratio:.2f}×", va="center", ha=tx[1], fontsize=VAL, color=INK)
        # right: gradient error and differing predictions
        if fam in ("gist", "whole"):
            ar.text(.6, yi, gist_txt if fam == "gist" else "0 (reference)",
                    va="center", ha="left", fontsize=VAL, color=GIST_F if fam == "gist" else INK,
                    fontweight="normal")
            continue
        if err is not None:
            ar.barh(yi + h / 4, err, h / 2, color=GREY, edgecolor=GREY, lw=.5, zorder=3)
        if dis is not None:
            ar.barh(yi - h / 4, dis, h / 2, color="white", edgecolor=INK, hatch="......", lw=.5, zorder=3)
        right = max(err or 0, dis or 0)
        ar.text(right + .8, yi, (f"{err:.1f} / " if err is not None else "– / ") + f"{dis:.1f}",
                va="center", ha="left", fontsize=VAL, color=INK)
    al.set_xscale("log")
    al.set_xlim(.025, 30)
    al.set_xticks([.03, .1, .3, 1, 3, 10], ["0.03", "0.1", "0.3", "1", "3", "10"])
    al.minorticks_off()
    al.axvline(1, color=GIST_F, lw=.7, zorder=2)
    al.set_yticks(y, [f[0] for f in FID], fontsize=6.2, linespacing=.9)
    for t in al.get_yticklabels():
        if t.get_text() == "GIST":
            t.set_color(GIST_F)
            t.set_fontweight("bold")
    al.grid(axis="y", visible=False)
    al.set_ylim(-.6, len(FID) + .25)
    al.set_xlabel("Nodes per update (× exact $L$-hop)", labelpad=1.5)
    al.text(.9, len(FID) - .1, "← drops context", ha="right", va="center", fontsize=5.6, color=DROP_G)
    al.text(1.11, len(FID) - .1, "extra nodes →", ha="left", va="center", fontsize=5.6, color=GREY)
    ar.set_xlim(0, 45)
    ar.set_xticks([0, 10, 20, 30, 40])
    ar.set_ylim(-.6, len(FID) + .25)
    ar.tick_params(labelleft=False, left=False)
    ar.grid(axis="y", visible=False)
    ar.set_xlabel("Error (%)", labelpad=1.5)
    fig.legend(handles=[Patch(facecolor=GREY, edgecolor=GREY, lw=.5, label="gradient error"),
                        Patch(facecolor="white", edgecolor=INK, hatch="......", lw=.5, label="differing predictions")],
               ncol=2, loc="upper center", bbox_to_anchor=(.61, 1.02), frameon=False, handlelength=1.3,
               handleheight=.8, columnspacing=.9, handletextpad=.35)
    check_layout(fig)
    save(fig, "intro_fidelity")
    LOG.append("")


if __name__ == "__main__":
    runtime()
    fidelity()
    (OUT / "intro_values.md").write_text("# Intro figures: every plotted value\n\nGenerated by "
                                         "`analysis/make_intro_figures.py` from the Phase D raw results.\n\n"
                                         + "\n".join(LOG) + "\n")
    print("\n".join(LOG))
