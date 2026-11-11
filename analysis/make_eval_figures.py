"""Evaluation figures of the GIST ICDE paper (Sec. 4, rework of 2026-11-06), from the Phase D raw results (seeds 0-9;
timing D3/E7 seeds 0-4). Reads only; never trains.

    OMP_NUM_THREADS=8 CUDA_VISIBLE_DEVICES= python analysis/make_eval_figures.py [exact cost ...]

eval_exact     fig:e-exact.    Digits of the parent-graph gradient that an update reproduces (-log10 of the largest relative
                               per-tensor difference), per backbone (D2), in-stream repairs (D1 audits) and link prediction
                               (D6 R4 audits): GIST's L-hop subgraph, the (L-1)-hop control, the subgraph without the fix.
eval_cost      fig:e-cost.     (a) share of the graph one update materialises, root rule x parent graph (D5 E1mix, N=8192);
                               (b) matched roots and whole-graph parent (D8 R1); (c) GIST's update against the size of the
                               repaired region (D5 E3a, GAT, N=2048).
eval_speed     fig:e-speed.    (a) repair speedup against N (D3); (b) mean repair at N=8192 split into host sampling and the
                               rest (D3); (c) peak repair memory, GAT on Arxiv and GraphSAGE on Products (D7); (d) serving
                               p99 during a repair (E7).
eval_accuracy  fig:e-acc.      (a) Full and GIST - Frozen per stream, natural vs injected drift (D1; Elliptic macro-F1 from
                               D8 R3); (b) every rule on the two Arxiv streams (D1); (c) drifted / clean nodes (D1).
eval_ablation  fig:e-ablation  (a) root rule (D1, D8, D9); (b) rehearsal share and budget (D4); (c) repair rate: cooldown,
                               threshold and a timer (D8); (d) breaking exactness inside the stream (D9).
eval_families  fig:families    per link-prediction graph: how much further uniform roots reach than regional ones (E1), the
                               nodes GIST saves, and Full - Frozen (D6).
Writes results/figures/eval/<name>.{pdf,png} (600 dpi) and eval_values.md (every plotted value).
Palette (validated with the dataviz checker, 2026-11-06): teal = GIST, dark red = Full, olive = whole graph / third series,
blue and amber = further series only, grey = neutral. Statistics as agg_draft.py / agg_review.py (paired by seed; mean +- 95% t CI).
"""
from __future__ import annotations

import glob
import json
import re
import sys
from pathlib import Path

import numpy as np
from scipy import stats

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(Path(__file__).resolve().parent))
import agg_draft as AD      # noqa: E402
import agg_review as AR     # noqa: E402
import fig_speed as FS      # noqa: E402
import fig_eval as FE       # noqa: E402

import matplotlib            # noqa: E402
matplotlib.use("Agg")
import matplotlib.pyplot as plt                 # noqa: E402
from matplotlib import font_manager             # noqa: E402
from matplotlib.lines import Line2D             # noqa: E402
from matplotlib.patches import Patch, Rectangle  # noqa: E402

F = ROOT / "results"
OUT = F / "figures" / "eval"
D1 = F / "D1_main/raw/streams/D1"
D4 = F / "D4_budget/raw/streams/D4"
D8 = F / "D8_review/raw/streams/D8"
D9 = F / "D9_ablation/raw/streams/D9"
MODELS = ("gat", "sage", "gcn", "gin")
NAME = dict(gat="GAT", sage="SAGE", gcn="GCN", gin="GIN")

BLUE, BLUE_T, BLUE_D = "#1B998B", "#BDE6E0", "#12695F"      # GIST: teal (name kept)
AMB, AMB_T, AMB_D = "#C0392B", "#F2C4BE", "#8E2A20"      # Full: dark red (name kept)
GRN, GRN_T, GRN_D = "#5DAE4B", "#CFE8C6", "#3A7A2C"      # third series: leaf green (name kept)
VIO, VIO_T, VIO_D = "#3B7DC4", "#C9DCF1", "#285A91"      # fourth series: blue (name kept)
YEL, YEL_T, YEL_D = "#E0A526", "#F4DFA6", "#A87A12"      # fifth series: amber
GRY, GRY_T = "#8A949C", "#DDE2E6"
INK = "#2F3B45"
SERIF = next((f for f in ("Times New Roman", "Nimbus Roman", "Nimbus Roman No9 L", "Liberation Serif")
              if any(e.name == f for e in font_manager.fontManager.ttflist)), "DejaVu Serif")
plt.rcParams.update({
    "font.family": SERIF, "font.size": 7, "axes.titlesize": 7, "axes.labelsize": 7, "xtick.labelsize": 6.5,
    "ytick.labelsize": 6.5, "legend.fontsize": 6.5, "pdf.fonttype": 42, "ps.fonttype": 42, "axes.linewidth": 0.5,
    "xtick.major.width": 0.5, "ytick.major.width": 0.5, "xtick.major.size": 2, "ytick.major.size": 2,
    "xtick.minor.size": 0, "ytick.minor.size": 0, "axes.spines.top": False, "axes.spines.right": False,
    "axes.grid": True, "axes.axisbelow": True, "grid.linewidth": 0.3, "grid.color": "#E3E6E8",
    "mathtext.fontset": "stix", "hatch.linewidth": 0.45, "legend.frameon": False, "legend.handletextpad": .3,
    "legend.columnspacing": .9, "legend.handlelength": 1.4, "axes.edgecolor": INK, "xtick.color": INK,
    "ytick.color": INK, "axes.labelcolor": INK, "text.color": INK,
})
VAL = 5.6
HATCH = "//////"
FULL_COST = ("#2E6B3A", "#C5DCC9", "#1C4426")              # eval_cost only: Full in dark green
COL = dict(full=(AMB, AMB_T, AMB_D), gist=(BLUE, BLUE_T, BLUE_D), grn=(GRN, GRN_T, GRN_D), vio=(VIO, VIO_T, VIO_D),
           yel=(YEL, YEL_T, YEL_D), gry=(GRY, GRY_T, "#5F6A72"))
HAT = dict(full="//////", gist="\\\\\\\\\\\\", grn=".....", vio="xxxxx", yel="ooo", gry="+++")


# Gradient for eval_cost / eval_exact: pale green -> light green ->
# green -> teal (GIST, solid, the darkest); thin dark hatches on every other bar; Full in red shades in eval_cost (b).
GR1, GR2, GR3 = "#EAF6E2", "#B9E0A5", "#6DBB5A"
RD1, RD2, RD3 = "#FADBD8", "#F1948A", "#C0392B"


def gb(fc, hatch=None, alpha=1.0, lw=.45):
    d = dict(facecolor=fc, edgecolor=INK, lw=lw, alpha=alpha)
    if hatch:
        d.update(hatch=hatch, hatchcolor=INK)
    return d


GRAD = False      # gradient style (eval_cost look) for the *_2 variants of eval_speed / eval_ablation
SUFFIX = ""
GRAD_STY = dict(   # series -> (main fill, main hatch, light fill, light hatch); GIST main is solid teal, the darkest
    gist=(BLUE, None, "#A8DED6", "////"), full=(RD3, "\\\\\\", RD1, "////"), grn=(GR3, "////", GR1, "ooo"),
    vio=("#3B7DC4", "xxxx", "#C9DCF1", "xxxx"), yel=(GR2, "ooo", GR1, "ooo"), gry=("#AEB6BD", "++", "#E1E5E8", "++"))


def sty(k, tint=False, hollow=False, col=None, hatch=None, lw=.4):
    if GRAD:
        fc, hh, lfc, lhh = GRAD_STY[k]
        if hollow:
            return gb("white", lhh or "////", lw=.45)
        return gb(lfc, lhh, lw=.45) if tint else gb(fc, hh, lw=.45)
    """Every bar is hatched (print). Solid fill + white hatch = the main condition; light fill + dark hatch = the second
    condition; white fill + dark hatch = the whole-graph parent."""
    c, t, d = col or COL[k]
    fc = "white" if hollow else (t if tint else c)
    return dict(facecolor=fc, edgecolor=d, hatch=hatch or HAT[k], hatchcolor=d if (tint or hollow) else "white", lw=lw)
LOG: list[str] = []


def log(s=""):
    LOG.append(s)


def load(p):
    p = Path(p)
    return json.loads(p.read_text()) if p.exists() else None


def mci(v):
    m, h, n = AD.mci(v)
    return m, (0.0 if not np.isfinite(h) else h)


def title(ax, s, x=0.0, pad=3):
    ax.set_title(s, loc="left", pad=pad, fontweight="bold", x=x)


def save(fig, name):
    name = name + SUFFIX
    OUT.mkdir(parents=True, exist_ok=True)
    check_layout(fig, name)
    fig.savefig(OUT / f"{name}.pdf")
    fig.savefig(OUT / f"{name}.png", dpi=600)
    plt.close(fig)


def check_layout(fig, name):
    """Warn about overlapping text (rotated pairs excepted) and text outside the figure."""
    fig.canvas.draw()
    r = fig.canvas.get_renderer()
    items = []
    for t in fig.findobj(matplotlib.text.Text):
        if not t.get_visible() or not t.get_text().strip():
            continue
        try:
            bb = t.get_window_extent(r)
        except Exception:
            continue
        if bb.width >= 1:
            items.append((t, bb))
    bad = 0
    for i in range(len(items)):
        for j in range(i + 1, len(items)):
            (a, ba), (b, bb) = items[i], items[j]
            if a.get_rotation() and b.get_rotation():
                continue
            if ba.overlaps(bb):
                ov = min(ba.x1, bb.x1) - max(ba.x0, bb.x0), min(ba.y1, bb.y1) - max(ba.y0, bb.y0)
                if ov[0] > 1.5 and ov[1] > 1.5:
                    bad += 1
                    print(f"[{name}] overlap: {a.get_text()!r} / {b.get_text()!r}")
    W, H = fig.get_size_inches() * fig.dpi
    for t, bb in items:
        if bb.x0 < -1 or bb.y0 < -1 or bb.x1 > W + 1 or bb.y1 > H + 1:
            print(f"[{name}] outside figure: {t.get_text()!r}")
    print(f"[{name}] layout check: {bad} overlaps")


def cells(base):
    out = {}
    for f in glob.glob(str(base / "*.json")):
        d = load(f)
        hp = d["hyperparameters"]
        out[(hp["model"], hp["variant"], hp.get("tag", ""))] = d
    return out


def arms_of(*ds):
    a = {}
    for d in ds:
        for k, v in AR.by_arm(d).items():
            a.setdefault(k, v)
    return a


def bar_label(ax, x, y, s, up=True, rot=0, size=VAL - .3, dy=.02):
    ax.text(x, y + (dy if up else -dy), s, ha="center", va="bottom" if up else "top", fontsize=size, rotation=rot, color=INK)


# =====================================================================================================================
# fig:e-exact: digits of agreement
# =====================================================================================================================
DEPTH = dict(gat=2, sage=2, gin=2, gcn=2, batchnorm=2, evolvegcn_o=2, tgcn=2, gconvgru_comb=4, gconvgru_sym=4)
ROWS_EXACT = (("gat", "GAT"), ("sage", "GraphSAGE"), ("gin", "GIN"), ("gcn", "GCN"), ("batchnorm", "BatchNorm"),
              ("evolvegcn_o", "EvolveGCN-O"), ("tgcn", "T-GCN"), ("gconvgru", "GConvGRU"))
F64 = -np.log10(np.finfo(np.float64).eps)        # 15.65 digits


def exact_values():
    rows = load(F / "D2_exact/raw/D2_backbones.json")["rows"]
    out = []
    for key, lab in ROWS_EXACT:
        bk = [b for b in DEPTH if b == key or b.startswith(key + "_")]
        L = DEPTH[bk[0]]
        sel = lambda fixed, hops: [r["per_tensor"] for r in rows if r["backbone"] in bk and r["hops"] == hops
                                   and (r["fixed"] is None or r["fixed"] == fixed)]
        g, c = sel(True, L), sel(True, L - 1)
        nf = [r["per_tensor"] for r in rows if r["backbone"] in bk and r["hops"] == L and r["fixed"] is False]
        out.append(dict(label=lab, gist=max(g), n=len(g), control=min(c), nofix=min(nf) if nf else None))
    aud = [a for d in cells(D1).values() for r in d["rows"] if r["arm"] == "selective" for a in r.get("audits") or []]
    out.append(dict(label="In-stream repairs", gist=max(a["rel_diff"] for a in aud), n=len(aud),
                    control=min(a["rel_diff_short"] for a in aud), nofix=None))
    lp = [r for f in glob.glob(str(F / "D6_graphs/raw/audit_*_s0.json")) for r in load(f)]
    out.append(dict(label="Link prediction", gist=max(r["rel_2hop"] for r in lp), n=len(lp),
                    control=min(r["rel_1hop"] for r in lp), nofix=None))
    return out


def fig_exact():
    v = exact_values()
    dig = lambda x: float(-np.log10(x))
    log("## eval_exact: digits of the parent-graph gradient reproduced = -log10(largest relative per-tensor difference);"
        " GIST = worst (max difference) over audits, controls = best (min difference)\n")
    for r in v:
        log(f"- {r['label']} (n={r['n']}): GIST {r['gist']:.1e} = {dig(r['gist']):.1f} digits; L-1 hops {r['control']:.1e} = "
            f"{dig(r['control']):.1f}" + (f"; no fix {r['nofix']:.1e} = {dig(r['nofix']):.1f}" if r["nofix"] else ""))
    fig, ax = plt.subplots(figsize=(3.45, 1.72))
    fig.subplots_adjust(left=.115, right=.995, top=.86, bottom=.31)
    W = .27
    xs = np.arange(len(v)).astype(float)
    xs[-2:] += .45
    ax.axhline(F64, color=BLUE_D, lw=.6, ls=(0, (3, 2)), zorder=1)
    ax.text(xs[-1] + .45, F64 + .35, "all digits of float64 agree", ha="right", va="bottom", fontsize=VAL, color=BLUE_D)
    for x, r in zip(xs, v):
        ax.bar(x - W, dig(r["gist"]), W * .92, zorder=2, **gb(BLUE, lw=.3))
        ax.bar(x, max(dig(r["control"]), .08), W * .92, zorder=2, **gb(GR3, lw=.3))
        bar_label(ax, x - .03, max(dig(r["control"]), .08), f"{dig(r['control']):.1f}", size=VAL - 1.0, dy=.15)
        if r["nofix"] is not None:
            ax.bar(x + W, dig(r["nofix"]), W * .92, zorder=2, **gb(GR2, lw=.3))
            bar_label(ax, x + W + .06, dig(r["nofix"]), f"{dig(r['nofix']):.1f}", size=VAL - 1.0, dy=.15)
    ax.axvline((xs[-3] + xs[-2]) / 2, color=INK, lw=.4, ls=(0, (2, 2)))
    ax.set_xticks(xs, [r["label"] for r in v], rotation=35, ha="right", rotation_mode="anchor", fontsize=6.0)
    ax.tick_params(axis="x", length=0, pad=1.5)
    ax.grid(axis="x", visible=False)
    ax.set_ylim(0, 17.2)
    ax.set_yticks([0, 4, 8, 12, 16])
    ax.set_xlim(xs[0] - .6, xs[-1] + .5)
    ax.set_ylabel("Digits that agree with\nthe parent-graph gradient", labelpad=1, fontsize=6.4)
    h = [Patch(**gb(BLUE, lw=.3), label="GIST ($L$ hops)"), Patch(**gb(GR3, lw=.3), label="one hop less"),
         Patch(**gb(GR2, lw=.3), label="no normalization fix")]
    fig.legend(handles=h, ncol=3, loc="upper center", bbox_to_anchor=(.55, 1.01), handlelength=1.1)
    save(fig, "eval_exact")


# =====================================================================================================================
# fig:e-cost
# =====================================================================================================================
R5_GRAPHS = (("arxiv", "Arxiv"), ("products", "Products"), ("elliptic", "Elliptic"), ("patents", "Patents"),
             ("web-google", "Web-Google"), ("youtube", "YouTube"))


def r5_values(N=8192):
    out = {}
    for g, _ in R5_GRAPHS:
        d = load(F / f"D5_intro/raw/E1mix/{g}.json")
        n = d["num_nodes"]
        m = lambda kind, key: 100 * float(np.mean([r[key] for r in d["rows"] if r["N"] == N and r["roots"] == kind])) / n
        out[g] = {k: (m(k, "fanout_25_15_nodes"), m(k, "exact_nodes")) for k in ("uniform", "region", "gist")}
    return out


R1_ARMS = ("full", "full_isoroots", "selective_topup", "selective_logged", "full_exact", "selective_exact")


def r1_values(v):
    out = {}
    for m in MODELS:
        a = arms_of(load(D8 / f"{m}_{v}.json"), load(D8 / f"{m}_{v}_exact.json"))
        f = AR.mean_of(a["full"], "nodes")
        out[m] = {arm: (AR.mean_of(a[arm], "nodes") / f if arm in a else float("nan")) for arm in R1_ARMS}
        roots = [x / 8192 for r in a["selective_logged"].values() for x in r.get("roots_per_event", [])]
        out[m]["roots"] = float(np.median(roots))
    return out


def fig_cost():
    fig = plt.figure(figsize=(7.0, 1.72))
    gs = fig.add_gridspec(1, 2, width_ratios=[1.25, 1], left=.05, right=.99, top=.70, bottom=.15, wspace=.16)
    ax = fig.add_subplot(gs[0, 0])
    v = r5_values()
    log("\n## eval_cost (a) share of the graph materialised, N=8192 (fan-out parent / whole-graph parent), E1mix seeds 0-9\n")
    W = .26
    for i, (g, lab) in enumerate(R5_GRAPHS):
        for j, (k, fc, hh) in enumerate((("uniform", GR1, "ooo"), ("region", GR3, "////"), ("gist", BLUE, None))):
            fo, wh = v[g][k]
            x = i + (j - 1) * W
            ax.bar(x, wh, W * .9, zorder=2, **gb(fc, "xxxx", alpha=.38, lw=.35))
            ax.bar(x, fo, W * .9, zorder=3, **gb(fc, hh))
            log(f"- {g} {k}: {fo:.1f}% / {wh:.1f}%")
    for xv in np.arange(.5, len(R5_GRAPHS) - .5):
        ax.axvline(xv, color=GRY, lw=.6, ls=(0, (1, 1.5)), zorder=1)
    ax.set_yscale("log")
    ax.set_ylim(1, 140)
    ax.set_yticks([1, 10, 100], ["1", "10", "100"])
    ax.set_xticks(range(len(R5_GRAPHS)), [l for _, l in R5_GRAPHS], fontsize=6.2)
    ax.tick_params(axis="x", length=0, pad=2)
    ax.grid(axis="x", visible=False)
    ax.set_xlim(-.5, len(R5_GRAPHS) - .5)
    ax.set_ylabel("Graph materialized (%)", labelpad=1)
    title(ax, "(a) Root rule $\\times$ parent graph, $N{=}8{,}192$", x=-.05, pad=22)
    h = [Patch(**gb(GR1, "ooo"), label="Uniform"), Patch(**gb(GR3, "////"), label="One region"), Patch(**gb(BLUE), label="GIST rule"),
         Patch(**gb(GR2, "xxxx", alpha=.38, lw=.35), label="faded top: whole-graph parent")]
    ax.legend(handles=h, ncol=4, loc="lower left", bbox_to_anchor=(-.03, 1.06), fontsize=5.8, handlelength=1.1,
              columnspacing=.6, borderaxespad=0)
    bx = fig.add_subplot(gs[0, 1])
    log("\n## eval_cost (b) nodes per repair relative to Full (mean over backbones; min-max), D8 R1\n")
    STY = (gb(RD3, "\\\\\\"), gb(RD2, "////"), gb(GR2, "ooo"), gb(BLUE), gb(RD1, "xxxx"), gb(GR1, "xxxx"))
    W = .135
    for i, st in enumerate(("arxiv_year", "arxiv")):
        r = r1_values(st)
        for j, (arm, st_kw) in enumerate(zip(R1_ARMS, STY)):
            vals = np.array([r[m][arm] for m in MODELS], float)
            ok = vals[np.isfinite(vals)]
            x = i + (j - 2.5) * W + (.04 if j >= 4 else -.04)
            bx.bar(x, ok.mean(), W * .88, zorder=2, **st_kw)
            if arm != "full":
                bx.plot([x, x], [ok.min(), ok.max()], color=INK, lw=.5, zorder=3)
            bx.text(x, max(ok.max(), ok.mean()) + .03, f"{ok.mean():.2f}", ha="center", va="bottom", fontsize=VAL - .6,
                    rotation=90, color=INK)
            log(f"- {st} {arm}: mean {ok.mean():.3f} (min {ok.min():.3f}, max {ok.max():.3f}, n={len(ok)}); "
                + ", ".join(f"{m} {r[m][arm]:.3f}" for m in MODELS))
        log(f"- {st} GIST |S|/N median per backbone: " + ", ".join(f"{m} {r[m]['roots']:.2f}" for m in MODELS))
    bx.axhline(1, color=INK, lw=.4, ls=(0, (2, 2)), zorder=1)
    bx.axvline(.5, color=GRY, lw=.6, ls=(0, (1, 1.5)), zorder=1)
    bx.set_xlim(-.5, 1.5)
    bx.set_xticks([0, 1], ["Arxiv-year", "Arxiv"])
    bx.tick_params(axis="x", length=0, pad=2)
    bx.grid(axis="x", visible=False)
    bx.set_ylim(0, 1.5)
    bx.set_yticks([0, .5, 1, 1.5])
    bx.set_ylabel("Nodes relative to Full", labelpad=1)
    title(bx, "(b) Matched roots and whole-graph parent", x=-.06, pad=22)
    h = [Patch(**STY[0], label="Full"), Patch(**STY[1], label="Full, GIST's $|S|$"), Patch(**STY[4], label="Full, whole graph"),
         Patch(**STY[3], label="GIST"), Patch(**STY[2], label="GIST, $|S|{=}N$"), Patch(**STY[5], label="GIST, whole graph")]
    bx.legend(handles=h, ncol=3, loc="lower left", bbox_to_anchor=(.03, .985), fontsize=5.6, handlelength=1.2,
              columnspacing=.7, handletextpad=.25, borderaxespad=0)
    save(fig, "eval_cost")


def fig_scaling():
    """The cost of an update grows with the L-hop expansion of its roots (fig_eval.cost_scaling, restyled)."""
    fig = plt.figure(figsize=(3.45, 1.62))
    gs = fig.add_gridspec(1, 2, left=.125, right=.985, top=.71, bottom=.22, wspace=.42)
    # (a) expansion against the budget (fig_eval.cost_scaling (a), E1): uniform vs regional roots
    ex = fig.add_subplot(gs[0, 0])
    log("\n## eval_scaling (a) share of the graph in the exact 2-hop expansion of N roots, uniform / one region (E1, seeds 0-9)\n")
    for (g, lab), col, mk in zip((("arxiv", "Arxiv"), ("products", "Products"), ("patents", "Patents"), ("elliptic", "Elliptic")),
                                 (BLUE, GR3, VIO, "#2E6B3A"), ("o", "s", "^", "D")):
        sh = FE.e1_share(g)
        for kind, ls, mfc in (("uniform", "-", col), ("region", (0, (2.5, 1.5)), "white")):
            Ns = sorted(sh[kind])
            ex.plot(Ns, [100 * sh[kind][n] for n in Ns], color=col, ls=ls, marker=mk, ms=2.3, mfc=mfc, mew=.6, lw=.85,
                    label=lab if kind == "uniform" else None)
            log(f"- {g} {kind}: " + ", ".join(f"N={n}: {100 * sh[kind][n]:.1f}%" for n in Ns))
    ex.set_xscale("log", base=2)
    ex.set_xticks([128, 1024, 8192, 32768], ["128", "1K", "8K", "32K"])
    ex.minorticks_off()
    ex.set_ylim(0, 100)
    ex.set_xlabel("Roots $N$", labelpad=1)
    ex.set_ylabel("2-hop share (%)", labelpad=1)
    title(ex, "(a) Expansion of $N$ roots", x=-.3, pad=23)
    hl = ex.legend(ncol=2, loc="lower left", bbox_to_anchor=(-.3, 1.04), fontsize=5.3, handlelength=1.2, columnspacing=.4,
                   borderaxespad=0, handletextpad=.2)
    ex.add_artist(hl)
    ex.legend(handles=[Line2D([], [], color=INK, lw=.85, label="uniform"), Line2D([], [], color=INK, lw=.85, ls=(0, (2.5, 1.5)), label="one region")],
              loc="upper left", fontsize=5.3, handlelength=1.6, borderaxespad=.1)
    cx = fig.add_subplot(gs[0, 1])
    log("\n## eval_scaling (b) GIST nodes / Full nodes against the repaired region (E3a, GAT, N=2048)\n")
    for (g, lab), col, mk, ls in zip((("arxiv", "Arxiv"), ("patents", "Patents"), ("web-google", "Web-Google"), ("youtube", "YouTube")),
                                     (BLUE, GR3, VIO, "#2E6B3A"), ("o", "s", "^", "D"), ("-", (0, (3, 1.5)), (0, (1, 1)), (0, (4, 1, 1, 1)))):
        full, sel, whole, oom = FE.e3a(g)
        p = sorted(sel)
        y = [sel[x] / full for x in p]
        cx.plot(p, y, color=col, marker=mk, ms=2.4, lw=.9, ls=ls, label=lab, mec=col)
        log(f"- {g}: Full {full:.0f}, whole {whole:.0f} ({whole / full:.1f}x; OOM {oom}); " + ", ".join(f"{x}%: {s:.2f}" for x, s in zip(p, y)))
    cx.axhline(1, color=RD3, lw=.8, zorder=1)
    cx.text(.98, 1.01, "Full", color=RD3, fontsize=VAL, va="bottom", ha="right", transform=cx.get_yaxis_transform())
    cx.set_xscale("log")
    cx.set_xticks([.5, 1, 5, 25], ["0.5", "1", "5", "25"])
    cx.set_ylim(.3, 1.12)
    cx.set_xlabel("Repaired region (% of graph)", labelpad=1)
    cx.set_ylabel("GIST / Full nodes", labelpad=1)
    title(cx, "(b) Cost against region size", x=-.3, pad=23)
    cx.legend(ncol=2, loc="lower left", bbox_to_anchor=(-.3, 1.04), fontsize=5.3, handlelength=1.7, columnspacing=.4, borderaxespad=0)
    save(fig, "eval_scaling")


# =====================================================================================================================
# fig:e-speed
# =====================================================================================================================
MK = dict(gat="o", sage="s", gcn="^", gin="v")
MCOL = dict(gat="#12695F", sage=BLUE, gcn=GR3, gin="#2E6B3A")


def fig_speed():
    d3 = FS.load_d3()
    rec = FS.summarise(d3)
    e7 = FS.load_e7()
    fig = plt.figure(figsize=(7.0, 1.6))
    gs = fig.add_gridspec(1, 4, width_ratios=[.8, 1.3, .95, .85], left=.058, right=.99, top=.76, bottom=.21, wspace=.45)
    ax = fig.add_subplot(gs[0, 0])
    log("\n## eval_speed (a) repair speedup Full/GIST (mean of per-seed ratios; min-max), D3 seeds 0-4\n")
    for k, m in enumerate(MODELS):
        r = [x for x in rec if x["model"] == m]
        Ns = [x["N"] * (1 + (k - 1.5) * .025) for x in r]
        s = [x["speedup"] for x in r]
        ax.errorbar(Ns, s, yerr=[[a["speedup"] - a["speedup_min"] for a in r], [a["speedup_max"] - a["speedup"] for a in r]],
                    color=MCOL[m], marker=MK[m], ms=2.9, lw=.95, elinewidth=.45, capsize=0, mec=INK if m == "gcn" else MCOL[m], mew=.4, label=NAME[m])
        log(f"- {m}: " + ", ".join(f"N={x['N']}: {x['speedup']:.2f} ({x['speedup_min']:.2f}-{x['speedup_max']:.2f}), net {x['speedup_net']:.2f}" for x in r))
    ax.axhline(1, color=INK, lw=.4, ls=(0, (2, 2)))
    ax.set_xscale("log", base=2)
    ax.set_xticks([2048, 4096, 8192], ["2K", "4K", "8K"])
    ax.minorticks_off()
    ax.set_xlim(1700, 8192 * 1.2)
    ax.set_ylim(.75, 2.0)
    ax.legend(ncol=2, loc="lower right", fontsize=5.6, handlelength=1.4, columnspacing=.4, borderaxespad=.1)
    ax.set_xlabel("Budget $N$", labelpad=1)
    ax.set_ylabel("Speedup over Full", labelpad=1)
    title(ax, "(a) Repair speedup", x=-.3, pad=11)
    bx = fig.add_subplot(gs[0, 1])
    log("\n## eval_speed (b) mean repair at N=8192 (ms): host sampling + rest\n")
    y, yt, yl = 0, [], []
    for m in MODELS[::-1]:
        r = next(x for x in rec if x["model"] == m and x["N"] == 8192)
        for arm in ("gist", "full"):
            s, u = r[f"{arm}_sample"], r[f"{arm}_update"]
            bx.barh(y, s, .78, **sty(arm, tint=True))
            bx.barh(y, u, .78, left=s, **sty(arm))
            bx.text(s + u + 4, y, f"{s + u:.0f}", va="center", fontsize=VAL - .2, color=INK)
            yt.append(y); yl.append("GIST" if arm == "gist" else "Full")
            log(f"- {m} {arm}: sampling {s:.1f} + rest {u:.1f} = {s + u:.1f} ms")
            y += 1
        bx.text(-58, y - 1.5, NAME[m], ha="right", va="center", fontsize=6.3, fontweight="bold", clip_on=False)
        y += .5
    bx.set_yticks(yt, yl, fontsize=5.8)
    bx.tick_params(axis="y", length=0, pad=1)
    bx.grid(axis="y", visible=False)
    bx.set_xlim(0, 340)
    bx.set_xlabel("Mean repair time at $N{=}8{,}192$ (ms)", labelpad=1)
    bx.legend(handles=[Patch(**sty("gry", tint=True, hatch="////"), label="host sampling (light)"),
                       Patch(**sty("gry", hatch="////"), label="rest of the repair")],
              ncol=2, loc="lower left", bbox_to_anchor=(-.02, .985), fontsize=5.8, borderaxespad=0)
    title(bx, "(b) Breakdown of repair time", x=-.27, pad=11)
    cx = fig.add_subplot(gs[0, 2])
    log("\n## eval_speed (c) peak repair memory at N=8192 (median GB over seeds; OOM = every seed), D7\n")
    H = (256, 512, 1024)
    grid = []
    for g, m, gl in (("arxiv", "gat", "GAT, Arxiv"), ("products", "sage", "SAGE, Products")):
        rows = load(F / f"D7_limits/raw/D7/memory_{g}_{m}.json")["rows"]
        for arm, al, col in (("full", "Full", AMB), ("selective", "GIST", BLUE), ("global", "Whole", GRN)):
            vals = []
            for h in H:
                sel = [x for x in rows if x["method"] == arm and x["hidden"] == h and (arm == "global" or x["N"] == 8192)]
                ok = [x["peak_mb"] / 1000 for x in sel if not x["oom"] and x.get("peak_mb")]
                vals.append(float(np.median(ok)) if ok and len(ok) == len(sel) else None)
                log(f"- {g} {m} {arm} h={h}: {'%.2f' % np.median(ok) if ok else 'OOM'} (OOM {len(sel) - len(ok)}/{len(sel)})")
            grid.append((gl, al, col, vals))
    for r, (gl, al, col, vals) in enumerate(grid):
        yy = len(grid) - 1 - r
        for k, val in enumerate(vals):
            if val is None:
                fc, txt, tc = GRY_T, "OOM", INK
            else:
                a = min(np.log10(max(val, .3) / .3) / np.log10(48 / .3), 1)
                fc = (*matplotlib.colors.to_rgb(col), .18 + .72 * a)
                txt, tc = (f"{val:.1f}" if val >= 1 else f"{val:.2f}"), ("white" if a > .78 else INK)
            cx.add_patch(Rectangle((k, yy), .94, .9, fc=fc, ec="none"))
            cx.text(k + .47, yy + .45, txt, ha="center", va="center", fontsize=VAL, color=tc, fontweight="bold" if al == "GIST" else None)
        cx.text(-.08, yy + .45, al, ha="right", va="center", fontsize=6.0)
    for gi, gl in enumerate(("GAT\nArxiv", "SAGE\nProducts")):
        cx.text(-1.05, 4.45 - 3 * gi, gl, ha="center", va="center", rotation=90, fontsize=5.3, fontweight="bold", linespacing=1.0)
    cx.axhline(2.95, color=INK, lw=.5)
    cx.set_xlim(0, 3)
    cx.set_ylim(0, 6)
    cx.set_xticks([.47, 1.47, 2.47], [str(h) for h in H])
    cx.set_yticks([])
    cx.tick_params(axis="x", length=0, pad=2)
    cx.grid(False)
    for sp in ("left", "bottom"):
        cx.spines[sp].set_visible(False)
    cx.set_xlabel("Hidden width", labelpad=1)
    title(cx, "(c) Peak memory (GB), $N{=}8{,}192$", x=-.45, pad=11)
    dx = fig.add_subplot(gs[0, 3])
    log("\n## eval_speed (d) serving p99 while a repair loops on the serving GPU (ms; ratio to idle), E7 seeds 0-4\n")
    W = .27
    for i, m in enumerate(("gat", "sage")):
        idle = e7[m]["idle"]["p99"]
        for j, (c, sk) in enumerate((("full", "full"), ("selective", "gist"), ("global", "grn"))):
            v = e7[m][c]
            x = i + (j - 1) * W
            if v["oom"] == v["n"]:
                dx.text(x, .3, "OOM", rotation=90, ha="center", va="bottom", fontsize=VAL - .4, color=INK)
                log(f"- {m} {c}: OOM on {v['oom']}/{v['n']}")
                continue
            dx.bar(x, v["p99"] / idle, W * .88, **sty(sk))
            dx.text(x, v["p99"] / idle + .1, f"{v['p99']:.1f}", ha="center", va="bottom", fontsize=VAL - .6, rotation=90, color=INK)
            log(f"- {m} {c}: p99 {v['p99']:.2f} ms ({v['p99'] / idle:.2f}x idle {idle:.2f} ms)")
    dx.axhline(1, color=INK, lw=.4, ls=(0, (2, 2)))
    dx.set_xticks([0, 1], ["GAT", "GraphSAGE"])
    dx.tick_params(axis="x", length=0, pad=2)
    dx.grid(axis="x", visible=False)
    dx.set_ylim(0, 7.6)
    dx.set_ylabel("Serving p99 / idle", labelpad=1)
    dx.legend(handles=[Patch(**sty("full"), label="Full"), Patch(**sty("gist"), label="GIST"), Patch(**sty("grn"), label="Whole")],
              ncol=1, loc="upper left", fontsize=5.6, handlelength=1.1)
    title(dx, "(d) Serving latency during a repair", x=-.32, pad=11)
    save(fig, "eval_speed")


# =====================================================================================================================
# fig:e-acc
# =====================================================================================================================
STREAMS = (("arxiv_year", "Arxiv-year"), ("elliptic", "Elliptic\n(F1)"), ("arxiv", "Arxiv"), ("cora", "Cora"),
           ("pubmed", "PubMed"), ("products", "Products"))
PRIOR = (("full", "Full", "full", None), ("selective", "GIST", "gist", None), ("global_trigger", "Whole\ngraph", "grn", None),
         ("roland", "ROLAND", "yel", None), ("ewc", "EWC", "vio", None), ("twp", "TWP", "vio", "+++"), ("ergnn", "ER-GNN", "gry", None))


def macro_f1_diff(m, arm):
    a = AR.by_arm(load(D8 / f"{m}_elliptic.json"))
    src = {"full": "full", "selective": "selective_logged"}[arm]
    met = {k: {s: AR.f1_bal(r["confusion"])[0] for s, r in rows.items()} for k, rows in a.items()}
    return [100 * (met[src][s] - met["baseline"][s]) for s in sorted(met[src]) if s in met["baseline"]]


def pooled(c, st, arm, ref="baseline"):
    return [x for m in MODELS if (m, st, "") in c for x in AD.diff(c[(m, st, "")]["rows"], arm, ref)]


def fig_accuracy():
    c = cells(D1)
    fig = plt.figure(figsize=(7.0, 1.7))
    gs = fig.add_gridspec(1, 3, width_ratios=[1.25, 1.0, 1.2], left=.052, right=.995, top=.72, bottom=.22, wspace=.24)
    # (a) Full and GIST per stream
    ax = fig.add_subplot(gs[0, 0])
    log("\n## eval_accuracy (a) Full / GIST - Frozen, pp, mean +- 95% CI over backbones x seeds (Elliptic: macro-F1, D8 R3)\n")
    W = .36
    ax.axvline(1.5, color=GRY, lw=.6, ls=(0, (1, 1.5)), zorder=1)
    ax.text(.5, -1.75, "natural drift", ha="center", va="bottom", fontsize=VAL + .2, color=GRN_D, fontweight="bold")
    ax.text(3.5, -1.75, "injected drift", ha="center", va="bottom", fontsize=VAL + .2, color=INK, fontweight="bold")
    for i, (st, lab) in enumerate(STREAMS):
        for k, (arm, sk) in enumerate((("full", "full"), ("selective", "gist"))):
            d = [x for m in MODELS for x in macro_f1_diff(m, arm)] if st == "elliptic" else pooled(c, st, arm)
            mu, h = mci(d)
            x = i + (k - .5) * W
            ax.bar(x, mu, W * .9, zorder=2, **sty(sk))
            ax.plot([x, x], [mu - h, mu + h], color=INK, lw=.55, zorder=3)
            ax.text(x, (mu + h + .06) if mu >= 0 else .06, f"{mu:+.1f}", ha="center", va="bottom", fontsize=VAL - .7, color=INK)
            log(f"- {st} {arm}: {mu:+.2f} +- {h:.2f} (n={len(d)})")
    ax.axhline(0, color=INK, lw=.5)
    ax.set_xticks(range(len(STREAMS)), [l for _, l in STREAMS], fontsize=5.7)
    ax.tick_params(axis="x", length=0, pad=2)
    ax.grid(axis="x", visible=False)
    ax.set_xlim(-.5, len(STREAMS) - .5)
    ax.set_ylim(-1.9, 3.15)
    ax.set_ylabel("Gain over frozen model (pp)", labelpad=1)
    ax.legend(handles=[Patch(**sty("full"), label="Full"), Patch(**sty("gist"), label="GIST")],
              ncol=2, loc="lower left", bbox_to_anchor=(-.02, 1.025), borderaxespad=0)
    title(ax, "(a) Accuracy change per stream", x=-.12, pad=17)
    # (b) every rule on the two Arxiv streams
    bx = fig.add_subplot(gs[0, 1])
    log("\n## eval_accuracy (b) every rule - Frozen on the Arxiv streams, pp (pooled over backbones x seeds; OOM seeds excluded)\n")
    W = .4
    for i, (arm, lab, sk, hh) in enumerate(PRIOR):
        for k, st in enumerate(("arxiv_year", "arxiv")):
            d = pooled(c, st, arm)
            mu, h = mci(d)
            x = i + (k - .5) * W
            bx.bar(x, mu, W * .88, zorder=2, **gb(GR3 if k == 0 else GR1, "////"))
            bx.plot([x, x], [mu - h, mu + h], color=INK, lw=.5, zorder=3)
            log(f"- {st} {arm}: {mu:+.2f} +- {h:.2f} (n={len(d)})")
    for i in range(len(PRIOR) - 1):
        bx.axvline(i + .5, color=GRY, lw=.5, ls=(0, (3, 2)), zorder=1)
    bx.axhline(0, color=INK, lw=.5)
    bx.set_xticks(range(len(PRIOR)), [p[1].replace("\n", " ") for p in PRIOR], fontsize=5.7, rotation=28, ha="right", rotation_mode="anchor")
    bx.tick_params(axis="x", length=0, pad=2)
    bx.grid(axis="x", visible=False)
    bx.set_xlim(-.55, len(PRIOR) - .45)
    bx.set_ylim(-1.9, 1.3)
    bx.legend(handles=[Patch(**gb(GR3, "////"), label="Arxiv-year (natural)"),
                       Patch(**gb(GR1, "////"), label="Arxiv (injected)")],
              ncol=2, loc="lower left", bbox_to_anchor=(-.02, 1.025), borderaxespad=0, handlelength=1.1)
    title(bx, "(b) Accuracy change per update rule", x=-.02, pad=17)
    # (c) drifted / clean nodes
    cx = fig.add_subplot(gs[0, 2])
    log("\n## eval_accuracy (c) drifted / clean holdout nodes: rule - Frozen, pp (pooled over backbones x seeds)\n")
    W = .2
    ST2 = [s for s in STREAMS]
    for i, (st, lab) in enumerate(ST2):
        for k, (arm, sk) in enumerate((("full", "full"), ("selective", "gist"))):
            vals = {}
            for part, idx in (("drifted", 0), ("clean", 1)):
                d = []
                for m in MODELS:
                    if (m, st, "") not in c:
                        continue
                    rows = c[(m, st, "")]["rows"]
                    A, B = AD.by(rows, arm), AD.by(rows, "baseline")
                    d += [100 * (AD.split_acc(A[s])[idx] - AD.split_acc(B[s])[idx]) for s in sorted(set(A) & set(B))]
                vals[part] = float(np.nanmean(d))
            xd = i - 1.5 * W + k * W
            xc = i + .5 * W + k * W
            cx.bar(xd, vals["drifted"], W * .9, zorder=2, **sty(sk))
            cx.bar(xc, vals["clean"], W * .9, zorder=2, **sty(sk, tint=True))
            log(f"- {st} {arm}: drifted {vals['drifted']:+.2f}, clean {vals['clean']:+.2f}")
    cx.axhline(0, color=INK, lw=.5)
    cx.set_xticks(range(len(ST2)), [l.split("\n")[0] for _, l in ST2], fontsize=5.9)
    cx.tick_params(axis="x", length=0, pad=2)
    cx.grid(axis="x", visible=False)
    cx.set_xlim(-.5, len(ST2) - .5)
    cx.set_ylim(-3.2, 2.6)
    cx.legend(handles=[Patch(**sty("gry", hatch="////"), label="drifted nodes"), Patch(**sty("gry", tint=True, hatch="////"), label="clean nodes (light)")],
              ncol=2, loc="lower left", bbox_to_anchor=(-.02, 1.025), borderaxespad=0, handlelength=1.1)
    title(cx, "(c) Drifted and clean nodes", x=-.02, pad=17)
    save(fig, "eval_accuracy")


# =====================================================================================================================
# fig:e-ablation
# =====================================================================================================================
def d9_arms(m, v):
    return AR.by_arm(load(D9 / f"{m}_{v}.json")), AR.by_arm(load(D9 / f"{m}_{v}_nofix.json"))


def fig_ablation():
    c = cells(D1)
    fig = plt.figure(figsize=(7.0, 1.65))
    outer = fig.add_gridspec(1, 3, width_ratios=[2.05, 1, .95], left=.055, right=.99, top=.78, bottom=.21, wspace=.36)
    inner = outer[0, 0].subgridspec(1, 2, width_ratios=[1.12, 1], wspace=.05)   # (a) and (b) share the y axis
    ax = fig.add_subplot(inner[0, 0])
    log("\n## eval_ablation (a) root rule: nodes / Full and accuracy - Full (pp), mean over backbones (seeds paired)\n")
    PTS = (("full_isoroots", "Full, GIST's $|S|$", AMB_T, AMB_D, "s"), ("selective_topup", "GIST, $|S|{=}N$", BLUE_T, BLUE_D, "D"),
           ("selective_logged", "GIST", BLUE, BLUE_D, "D"), ("random_region", "Random region", GRN, GRN_D, "o"),
           ("selective_trunc", "GIST, $L{-}1$ hops", VIO, VIO, "X"))
    for st, filled in (("arxiv_year", True), ("arxiv", False)):
        for arm, lab, col, ec, mk in PTS:
            xs, ys = [], []
            for m in MODELS:
                if arm == "random_region":
                    rows = c[(m, st, "")]["rows"]
                    xs.append(AD.ratio(rows, "nodes", "random_region", "full"))
                    ys += AD.diff(rows, "random_region", "full")
                    continue
                a = arms_of(load(D8 / f"{m}_{st}.json"))
                if arm == "selective_trunc":
                    t, _ = d9_arms(m, st)
                    if "selective_trunc" not in t:
                        continue
                    a = dict(a, selective_trunc=t["selective_trunc"])
                if arm not in a:
                    continue
                xs.append(AR.mean_of(a[arm], "nodes") / AR.mean_of(a["full"], "nodes"))
                ys += AR.paired(a[arm], a["full"])
            if not xs:
                continue
            mu, h = mci(ys)
            ax.errorbar(np.mean(xs), mu, yerr=h, color=ec, marker=mk, ms=3.4, mfc=col if filled else "white",
                        mec=ec, mew=.7, elinewidth=.5, capsize=0, ls="none", zorder=3)
            log(f"- {st} {arm}: nodes {np.mean(xs):.3f}, acc - Full {mu:+.2f} +- {h:.2f} (n={len(ys)})")
    ax.plot(1, 0, "s", color=AMB, mec=AMB_D, ms=3.4, zorder=3)
    ax.text(1.0, .12, "Full", ha="center", va="bottom", fontsize=VAL, color=AMB_D)
    ax.axhline(0, color=INK, lw=.4, ls=(0, (2, 2)))
    ax.set_xlim(.15, 1.08)
    ax.set_xticks([.2, .4, .6, .8, 1.0])
    ax.set_ylim(-2.2, .5)
    ax.set_xlabel("Nodes relative to Full", labelpad=1)
    ax.set_ylabel("Change vs. Full (pp)", labelpad=1)
    h = [Line2D([], [], color=ec, mfc=col, marker=mk, ms=3.2, ls="none", mew=.7, label=lab) for _, lab, col, ec, mk in PTS]
    ax.legend(handles=h, ncol=2, loc="lower right", fontsize=5.4, handletextpad=.1, borderaxespad=.1, columnspacing=.4)
    title(ax, "(a) Root selection (filled: Arxiv-year)", x=-.2, pad=10)
    bx = fig.add_subplot(inner[0, 1], sharey=ax)
    log("\n## eval_ablation (b) rehearsal share rho and budget N: GIST - Full (pp) and nodes / Full, GAT + SAGE x 2 streams\n")
    for N, col, ec, mk in ((2048, GRN, GRN_D, "o"), (4096, YEL, YEL_D, "s"), (8192, BLUE, BLUE_D, "D")):
        xs, ys, es = [], [], []
        for rho in ("rho0", "", "rho0.5"):
            nx, dd = [], []
            for m in ("gat", "sage"):
                for st in ("arxiv_year", "arxiv"):
                    base = load(D4 / f"{m}_{st}_N{N}.json")
                    d = load(D4 / f"{m}_{st}_N{N}{'_' + rho if rho else ''}.json")
                    rows = d["rows"] + [r for r in base["rows"] if r["arm"] == "full" and "full" not in AR.by_arm(d)]
                    nx.append(AD.ratio(rows, "nodes"))
                    dd += AD.diff(rows, "selective", "full")
            mu, h = mci(dd)
            xs.append(np.mean(nx)); ys.append(mu); es.append(h)
            log(f"- N={N} {rho or 'rho0.25'}: nodes {np.mean(nx):.3f}, GIST - Full {mu:+.2f} +- {h:.2f}")
        bx.errorbar(xs, ys, yerr=es, color=col, mec=ec, marker=mk, ms=3, lw=.9, elinewidth=.45, capsize=0, label=f"$N{{=}}{N // 1024}$K")
        dfx, dfy = [], []
        for m in ("gat", "sage"):
            base = load(D4 / f"{m}_arxiv_N{N}.json")
            d = load(D4 / f"{m}_arxiv_N{N}_diffuse.json")
            rows = d["rows"] + [r for r in base["rows"] if r["arm"] == "full" and "full" not in AR.by_arm(d)]
            dfx.append(AD.ratio(rows, "nodes")); dfy += AD.diff(rows, "selective", "full")
        bx.plot(np.mean(dfx), np.mean(dfy), marker=mk, mfc="white", mec=ec, ms=3.2, mew=.7, ls="none", zorder=4)
        log(f"- N={N} diffuse (Arxiv): nodes {np.mean(dfx):.3f}, GIST - Full {np.mean(dfy):+.2f}")
    for x, lab in ((.33, "$\\rho{=}0$"), (.67, "$0.25$"), (.83, "$0.5$")):
        bx.text(x, .25, lab, ha="center", fontsize=VAL, color=INK)
    bx.axhline(0, color=INK, lw=.4, ls=(0, (2, 2)))
    bx.set_xlim(.2, 1.0)
    bx.set_xticks([.4, .6, .8, 1.0])
    bx.set_ylim(-2.2, .5)
    bx.tick_params(labelleft=False)
    bx.set_xlabel("Nodes relative to Full", labelpad=1)
    hb = bx.legend(ncol=1, loc="upper left", bbox_to_anchor=(0, .9), fontsize=5.6, handlelength=1.5, borderaxespad=.1)
    bx.add_artist(hb)
    bx.legend(handles=[Line2D([], [], color=INK, marker="o", mfc="white", ms=3, mew=.7, ls="none", label="drift scattered")],
              loc="lower right", fontsize=5.6, borderaxespad=.1, handletextpad=.1)
    title(bx, "(b) Rehearsal share and budget", x=-.02, pad=10)
    cx = fig.add_subplot(outer[0, 1])
    log("\n## eval_ablation (c) repair rate on Arxiv-year: repairs per 1000 steps and rule - Frozen (pp), mean over backbones\n")
    SET = (("_k100", "$\\kappa{=}100$"), ("_tau1", "$\\tau_{1\\%}$"), ("", "default"), ("_tau10", "$\\tau_{10\\%}$"), ("_k25", "$\\kappa{=}25$"))
    for arm, col, ec, mk in (("full", AMB, AMB_D, "s"), ("selective_logged", BLUE, BLUE_D, "D")):
        pts = []
        for tag, lab in SET:
            ev, dd = [], []
            for m in MODELS:
                a = AR.by_arm(load(D8 / f"{m}_arxiv_year{tag}.json"))
                ev.append(AR.mean_of(a[arm], "events")); dd += AR.paired(a[arm], a["baseline"])
            mu, h = mci(dd)
            pts.append((np.mean(ev), mu, h, lab))
            log(f"- {arm} {lab}: repairs {np.mean(ev):.1f}, - Frozen {mu:+.2f} +- {h:.2f}")
        pts.sort()
        cx.errorbar([p[0] for p in pts], [p[1] for p in pts], yerr=[p[2] for p in pts], color=col, mec=ec, marker=mk, ms=3,
                    lw=.9, elinewidth=.45, capsize=0)
        if arm == "selective_logged":
            for k, p in enumerate(pts):
                cx.text(p[0], 1.36 if k % 2 == 0 else 1.22, p[3], ha="center", va="center", fontsize=VAL - .3, color=INK)
                cx.axvline(p[0], color="#EEF0F2", lw=.4, zorder=0)
    ev, dd = [], []
    for m in MODELS:
        a = AR.by_arm(load(D8 / f"{m}_arxiv_year.json"))
        ev.append(AR.mean_of(a["selective_periodic"], "events")); dd += AR.paired(a["selective_periodic"], a["baseline"])
    mu, h = mci(dd)
    cx.plot(np.mean(ev) + .6, mu, marker="D", mfc="white", mec=BLUE_D, ms=3.4, mew=.7, ls="none", zorder=4)
    log(f"- GIST on a timer (Full's schedule): repairs {np.mean(ev):.1f}, - Frozen {mu:+.2f} +- {h:.2f}")
    cx.axhline(0, color=INK, lw=.4, ls=(0, (2, 2)))
    cx.set_xlim(4, 25)
    cx.set_ylim(-.5, 1.45)
    cx.set_xlabel("Repairs per 1,000 steps", labelpad=1)
    cx.set_ylabel("Gain over frozen model (pp)", labelpad=1)
    cx.legend(handles=[Line2D([], [], color=AMB, mec=AMB_D, marker="s", ms=3, lw=.9, label="Full"),
                       Line2D([], [], color=BLUE, mec=BLUE_D, marker="D", ms=3, lw=.9, label="GIST, detector"),
                       Line2D([], [], color=BLUE_D, marker="D", mfc="white", ms=3.2, mew=.7, ls="none", label="GIST, timer")],
              ncol=1, loc="upper left", bbox_to_anchor=(0, .87), fontsize=5.6, borderaxespad=.1)
    title(cx, "(c) Repair rate (Arxiv-year)", x=-.2, pad=10)
    dx = fig.add_subplot(outer[0, 2])
    log("\n## eval_ablation (d) breaking exactness inside the stream: arm - exact GIST (pp), seeds paired\n")
    have = False
    for i, m in enumerate(MODELS):
        # GCN carries both broken conditions (L-1 hops and no normalization fix) in one group of four bars
        offs, Wb = ((-.33, -.11), .2) if m == "gcn" else ((-.18, .18), .32)
        for k, st in enumerate(("arxiv_year", "arxiv")):
            t, nf = d9_arms(m, st)
            if "selective_trunc" in t and "selective_logged" in t:
                d = AR.paired(t["selective_trunc"], t["selective_logged"])
                mu, h = mci(d)
                x = i + offs[k]
                dx.bar(x, mu, Wb * .88, **sty("vio", tint=k == 1))
                dx.plot([x, x], [mu - h, mu + h], color=INK, lw=.5)
                nr = AR.mean_of(t["selective_trunc"], "nodes") / AR.mean_of(t["selective_logged"], "nodes")
                log(f"- {m} {st} L-1 hops: {mu:+.2f} +- {h:.2f} (n={len(d)}); nodes {nr:.3f} of GIST")
                have = True
            if m == "gcn" and "selective_nofix" in nf and "selective_exact" in nf:
                d = AR.paired(nf["selective_nofix"], nf["selective_exact"])
                mu, h = mci(d)
                log(f"- gcn {st} no fix (whole-graph parent): {mu:+.2f} +- {h:.2f} (n={len(d)})")
                x = i + (.11, .33)[k]
                dx.bar(x, mu, Wb * .88, **sty("grn", tint=k == 1))
                dx.plot([x, x], [mu - h, mu + h], color=INK, lw=.5)
    if not have:
        dx.text(.5, .5, "D9 pending", transform=dx.transAxes, ha="center", va="center", color=INK)
    for xv in (.5, 1.5, 2.5):
        dx.axvline(xv, color=GRY, lw=.6, ls=(0, (1, 1.5)), zorder=1)
    dx.axhline(0, color=INK, lw=.5)
    dx.set_xlim(-.5, 3.5)
    dx.set_xticks([0, 1, 2, 3], ["GAT", "SAGE", "GCN", "GIN"], fontsize=6.0)
    dx.tick_params(axis="x", length=0, pad=1.5)
    dx.grid(axis="x", visible=False)
    dx.set_ylabel("Change vs. exact GIST (pp)", labelpad=1)
    dx.legend(handles=[Patch(**sty("vio"), label="$L{-}1$ hops"), Patch(**sty("grn"), label="no fix"),
                       Patch(**sty("gry", tint=True, hatch="////"), label="Arxiv")], ncol=3, loc="lower left",
              fontsize=5.4, handlelength=1.0, columnspacing=.4, borderaxespad=.1)
    title(dx, "(d) Inexact updates", x=-.2, pad=10)
    save(fig, "eval_ablation")


# =====================================================================================================================
# fig:families
# =====================================================================================================================
FAM = dict(social=("Social", VIO, VIO_D, "vio"), web=("Web", BLUE, BLUE_D, "gist"), citation=("Citation", GRN, GRN_D, "grn"),
           p2p=("P2P", YEL, YEL_D, "yel"), road=("Road", GRY, "#5F6A72", "gry"))
GNAME = {"facebook": "Facebook", "twitter": "Twitter", "youtube": "YouTube", "web-stanford": "Web-Stanford",
         "web-google": "Web-Google", "web-berkstan": "Web-BerkStan", "gnutella25": "Gnutella25", "gnutella30": "Gnutella30",
         "gnutella31": "Gnutella31", "road-pa": "Road-PA", "road-tx": "Road-TX", "road-ca": "Road-CA", "hepth": "HepTh",
         "hepph": "HepPh", "patents": "Patents"}


def families_rows():
    rows = []
    for line in (ROOT / "results/D6_graphs/d6.md").read_text().split("## Detection")[0].splitlines():
        m = re.match(r"\| (social|web|p2p|road|citation) \| ([\w-]+) \|", line)
        if not m:
            continue
        c = [x.strip() for x in line.strip("|").split("|")]
        g, gci = re.match(r"([+-][\d.]+) ± ([\d.]+)", c[4]).groups()
        sf = re.match(r"([+-][\d.]+) ± ([\d.]+)", c[5]).groups()
        rows.append(dict(family=m.group(1), graph=m.group(2), G=float(g), Gci=float(gci), SF=float(sf[0]), cost=float(c[10])))
    assert len(rows) == 15
    for r in rows:
        N = load(F / f"D6_graphs/raw/gat_{r['graph']}_final.json")["cfg"]["budget"]
        s = FE.e1_share(r["graph"])
        near = min(s["uniform"], key=lambda n: abs(np.log2(n) - np.log2(N)))
        r.update(N=N, reach=s["uniform"][near] / s["region"][near])
    return rows


def fig_timeline():
    """Accuracy along the stream against the frozen model (fig_eval.timeline, restyled): 50-step moving mean over
    backbones x seeds, without seeds whose update ran out of memory (as in fig:e-acc)."""
    Wn = 50
    kernel = np.ones(Wn) / Wn
    fig, axes = plt.subplots(1, 2, figsize=(3.45, 1.62), sharey=True)
    fig.subplots_adjust(left=.115, right=.985, top=.78, bottom=.2, wspace=.07)
    log("\n## eval_timeline (pp against Frozen, 50-step mean over backbones x seeds)\n")
    for ax, stream, ttl in zip(axes, ("arxiv_year", "arxiv"), ("(a) Arxiv-year (natural)", "(b) Arxiv (injected)")):
        cur = {a: [] for a in ("full", "selective", "roland")}
        rep_, drift = [], []
        for m in MODELS:
            d = load(D1 / f"{m}_{stream}.json")
            base = {r["seed"]: np.asarray(r["step_accuracy"], float) for r in d["rows"] if r["arm"] == "baseline"}
            for arm in cur:
                cur[arm] += [100 * (np.asarray(r["step_accuracy"], float) - base[r["seed"]]) for r in d["rows"]
                             if r["arm"] == arm and not r.get("oom")]
            rep_ += [x for r in d["rows"] if r["arm"] == "full" for x in r["retrain_steps"]]
            if stream == "arxiv":
                drift += [x for vv in d["meta"].values() for x in vv["drift_steps"]]
        for arm, col, lab, lw, ls in (("roland", VIO, "ROLAND", .8, (0, (3, 1.5))), ("full", RD3, "Full", 1.0, "-"),
                                      ("selective", BLUE, "GIST", 1.1, "-")):
            y = np.mean(cur[arm], axis=0)
            z = np.convolve(y, kernel, mode="valid")
            xs = np.arange(Wn - 1, len(y))
            ax.plot(xs, z, color=col, lw=lw, ls=ls, label=lab, zorder=3)
            log(f"- {stream} {arm}: {len(cur[arm])} pairs, whole-stream mean {y.mean():+.2f} pp; end (last 50) {z[-1]:+.2f}; "
                f"first half {y[:500].mean():+.2f}, second half {y[500:].mean():+.2f}")
        ax.axhline(0, color=INK, lw=.5)
        ax.text(990, .07, "Frozen", ha="right", va="bottom", fontsize=5.8, color=INK)
        h, _ = np.histogram(rep_, bins=np.arange(0, 1001, 25))
        ax.vlines(np.arange(12.5, 1000, 25), -2.6, -2.6 + .5 * h / h.max(), color=GRY, lw=1.0, alpha=.8)
        ax.text(20, -2.05, "Full's repairs", fontsize=5.5, color=GRY, ha="left", va="bottom")
        if drift:
            hd, _ = np.histogram(drift, bins=np.arange(0, 1001, 25))
            ax.vlines(np.arange(12.5, 1000, 25), 2.3 - .35 * hd / hd.max(), 2.3, color=YEL_D, lw=1.0, alpha=.85)
            ax.text(25, 2.33, "injected drift events", fontsize=5.5, color=YEL_D, va="bottom", ha="left")
        ax.set_xlim(0, 1000)
        ax.set_ylim(-2.6, 2.75)
        ax.set_xticks([0, 500, 1000], ["0", "500", "1K"])
        ax.set_xlabel("Stream step", labelpad=1)
        ax.set_title(ttl, pad=2, fontsize=6.6, fontweight="bold")
    axes[0].set_ylabel("Gain over frozen model (pp)", labelpad=1.5)
    hh, ll = axes[0].get_legend_handles_labels()
    order = [2, 1, 0]
    fig.legend([hh[i] for i in order], [ll[i] for i in order], ncol=3, loc="upper center", bbox_to_anchor=(.55, 1.02),
               handlelength=1.6, columnspacing=1)
    save(fig, "eval_timeline")


def fig_families():
    rows = families_rows()
    order = ("web", "citation", "social", "p2p", "road")
    seq = [r for f in order for r in sorted((x for x in rows if x["family"] == f), key=lambda x: x["cost"])]
    log("\n## eval_families: reach = uniform / regional 2-hop share (E1, budget nearest the graph's); saved = 1 - GIST/Full "
        "nodes; G = Full - Frozen MRR x100 (D6)\n")
    for r in seq:
        log(f"- {r['family']} {r['graph']}: reach {r['reach']:.2f}x, saved {100 * (1 - r['cost']):.1f}%, G {r['G']:+.2f} +- "
            f"{r['Gci']:.2f}, GIST - Full {r['SF']:+.2f}")
    no_road = [r for r in rows if r["family"] != "road"]
    a = stats.spearmanr([r["reach"] for r in no_road], [r["cost"] for r in no_road])
    log(f"- Spearman reach vs GIST/Full nodes without road (12): {a.statistic:.2f} (p={a.pvalue:.4f})")
    fig = plt.figure(figsize=(3.45, 2.6))
    gs = fig.add_gridspec(1, 3, width_ratios=[1, 1, 1.15], left=.255, right=.985, top=.89, bottom=.165, wspace=.12)
    axs = [fig.add_subplot(gs[0, i]) for i in range(3)]
    ys, y, starts = [], 0.0, []
    for k, r in enumerate(seq):
        if k and r["family"] != seq[k - 1]["family"]:
            y += .55
        if not k or r["family"] != seq[k - 1]["family"]:
            starts.append((r["family"], y))
        ys.append(y)
        y += 1
    ys = np.array(ys)
    top = ys.max()
    for ax in axs:
        ax.set_ylim(top + .7, -.7)
        ax.grid(axis="y", visible=False)
        ax.tick_params(axis="y", length=0)
    for r, yy in zip(seq, ys):
        _, col, ec, sk = FAM[r["family"]]
        axs[0].barh(yy, r["reach"], .72, **sty(sk, tint=True))
        axs[1].barh(yy, 100 * (1 - r["cost"]), .72, **sty(sk))
        axs[2].barh(yy, r["G"], .72, **sty(sk, tint=True))
        axs[2].plot([r["G"] - r["Gci"], r["G"] + r["Gci"]], [yy, yy], color=INK, lw=.5)
    axs[0].set_yticks(ys, [GNAME[r["graph"]] for r in seq], fontsize=5.7)
    for ax in axs[1:]:
        ax.set_yticks(ys, [""] * len(ys))
    for fam, y0 in starts:
        n = sum(1 for r in seq if r["family"] == fam)
        lab, col, ec, _ = FAM[fam]
        axs[0].text(-1.02, y0 + (n - 1) / 2, lab, transform=axs[0].get_yaxis_transform(), rotation=90, ha="center", va="center",
                    fontsize=6.0, fontweight="bold", color=ec)
        for ax in axs:
            ax.axhspan(y0 - .45, y0 + n - .55, color=matplotlib.colors.to_rgba(col, .07), lw=0, zorder=0)
    axs[0].axvline(1, color=INK, lw=.4, ls=(0, (2, 2)))
    axs[0].set_xlim(0, 8.2)
    axs[0].set_xticks([1, 4, 8], ["1$\\times$", "4$\\times$", "8$\\times$"])
    axs[0].set_xlabel("Uniform reach /\nregional reach", labelpad=1, fontsize=6.3)
    axs[1].set_xlim(0, 52)
    axs[1].set_xticks([0, 20, 40], ["0", "20", "40"])
    axs[1].set_xlabel("Nodes GIST\nsaves (%)", labelpad=1, fontsize=6.3)
    axs[2].axvline(0, color=INK, lw=.5)
    axs[2].set_xlim(-1.6, 2.9)
    axs[2].set_xticks([-1, 0, 1, 2])
    axs[2].set_xlabel("Retraining gain\n(MRR points)", labelpad=1, fontsize=6.3)
    for ax, t in zip(axs, ("(a) Reach of\nuniform roots", "(b) Nodes\nsaved", "(c) Gain from\nretraining")):
        ax.set_title(t, fontweight="bold", pad=3, fontsize=6.4, linespacing=1.05)
    save(fig, "fig_families")


def _grad(fn):
    global GRAD, SUFFIX
    GRAD, SUFFIX = True, "_2"
    try:
        fn()
    finally:
        GRAD, SUFFIX = False, ""


def fig_speed2():
    _grad(fig_speed)


def fig_ablation2():
    _grad(fig_ablation)


def main():
    which = sys.argv[1:] or ["exact", "cost", "scaling", "speed", "speed2", "ablation2", "accuracy", "timeline", "ablation", "families"]
    for w in which:
        globals()[f"fig_{w}"]()
    OUT.mkdir(parents=True, exist_ok=True)
    old = (OUT / "eval_values.md").read_text().split("\n## ") if (OUT / "eval_values.md").exists() else []
    (OUT / "eval_values.md").write_text("\n".join(LOG) + "\n")
    print("\n".join(LOG))


if __name__ == "__main__":
    main()
