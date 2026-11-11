"""Evaluation figures of the GIST ICDE paper from the Phase D raw results (seeds 0-9). Reads only; never trains.

    OMP_NUM_THREADS=8 CUDA_VISIBLE_DEVICES= python analysis/fig_eval.py

Writes to results/figures/ (PDF + 600 dpi PNG), and fig_eval_values.md with every plotted value:
  fig_cost_scaling  E1 2-hop share vs N; E3a nodes per update vs region size
  fig_timeline      D1 holdout accuracy along the stream, against Frozen
  fig_frontier      D4 cost-accuracy frontier
  fig_families      E1 + D6 over-expansion vs node ratio; G per graph
  fig_memory        D7 peak repair memory
fig_speed is analysis/fig_speed.py.
"""
from __future__ import annotations

import glob
import json
import re
from pathlib import Path

import numpy as np
from scipy import stats
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager
from matplotlib.lines import Line2D

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "results" / "figures"
D5 = ROOT / "results" / "D5_intro" / "raw"
NAVY, GREY, VERM, GREEN, PINK, BLUE = "#0B2D5B", "#7A7A7A", "#D55E00", "#009E73", "#CC79A7", "#56B4E9"
INK = "#444444"
NAME = dict(gat="GAT", sage="GraphSAGE", gcn="GCN", gin="GIN")
LOG = []

SERIF = next((f for f in ("Times New Roman", "Nimbus Roman", "Nimbus Roman No9 L", "Liberation Serif")
              if any(e.name == f for e in font_manager.fontManager.ttflist)), "DejaVu Serif")
plt.rcParams.update({
    "font.family": SERIF, "font.size": 7, "axes.titlesize": 7, "axes.labelsize": 7, "xtick.labelsize": 6.5,
    "ytick.labelsize": 6.5, "legend.fontsize": 6.5, "pdf.fonttype": 42, "ps.fonttype": 42, "axes.linewidth": 0.5,
    "xtick.major.width": 0.5, "ytick.major.width": 0.5, "xtick.major.size": 2, "ytick.major.size": 2,
    "xtick.minor.size": 0, "ytick.minor.size": 1, "ytick.minor.width": 0.3,
    "axes.spines.top": False, "axes.spines.right": False, "axes.grid": True, "axes.axisbelow": True,
    "grid.linewidth": 0.3, "grid.color": "#DDDDDD", "mathtext.fontset": "stix",
})


def load(p):
    return json.loads(Path(p).read_text())


def save(fig, name):
    OUT.mkdir(exist_ok=True)
    fig.savefig(OUT / f"{name}.pdf")
    fig.savefig(OUT / f"{name}.png", dpi=600)
    plt.close(fig)


def ci(v):
    v = np.asarray(v, float)
    return float(stats.t.ppf(0.975, len(v) - 1) * v.std(ddof=1) / np.sqrt(len(v)))


# --- fig:cost-scaling ------------------------------------------------------------------------------------------------
def e1_share(graph):
    d = load(D5 / "E1" / f"{graph}.json")
    out = {}
    for kind in ("uniform", "region"):
        Ns = sorted({r["N"] for r in d["rows"] if r["roots"] == kind})
        out[kind] = {N: float(np.mean([r["exact"]["2"] for r in d["rows"] if r["roots"] == kind and r["N"] == N])) / d["num_nodes"] for N in Ns}
    return out


def e3a(graph):
    rows = load(D5 / "E3a" / f"{graph}_gat.json")["rows"]
    rows = [r for r in rows if r["fraction"] == 1.0]
    full = float(np.mean([r["nodes"] for r in rows if r["method"] == "full"]))
    pct = sorted({r["region_pct"] for r in rows if r["method"] == "selective"})
    sel = {p: float(np.mean([r["nodes"] for r in rows if r["method"] == "selective" and r["region_pct"] == p])) for p in pct}
    whole = float(np.mean([r["nodes"] for r in rows if r["method"] == "global"]))
    oom = [m for m in ("gat", "sage", "gcn", "gin")
           if all(r["oom"] for r in load(D5 / "E3a" / f"{graph}_{m}.json")["rows"] if r["fraction"] == 1.0 and r["method"] == "global")]
    return full, sel, whole, oom


def cost_scaling():
    fig = plt.figure(figsize=(7.0, 1.62))
    gs = fig.add_gridspec(1, 9, width_ratios=[1, 1, 1, 1, .62, 1, 1, 1, 1], left=.05, right=.995, top=.76, bottom=.215, wspace=.09)
    LOG.append("## fig_cost_scaling\n")
    first = None
    for i, (g, nm) in enumerate((("arxiv", "Arxiv"), ("products", "Products"), ("patents", "Patents"), ("elliptic", "Elliptic"))):
        ax = fig.add_subplot(gs[0, i], sharey=first)
        first = first or ax
        s = e1_share(g)
        for kind, c, mk in (("uniform", GREY, "s"), ("region", NAVY, "D")):
            Ns = sorted(s[kind])
            ax.plot(Ns, [100 * s[kind][n] for n in Ns], color=c, marker=mk, ms=2.6, lw=.9)
            LOG.append(f"- E1 {g} {kind}: " + ", ".join(f"N={n}: {100 * s[kind][n]:.1f}%" for n in Ns))
        ax.set_xscale("log", base=2)
        ax.set_xticks([512, 8192], ["512", "8K"])
        ax.minorticks_off()
        ax.set_ylim(0, 100)
        ax.set_title(nm, pad=2)
        if i:
            ax.tick_params(labelleft=False)
        else:
            ax.set_ylabel("2-hop share (%)", labelpad=1.5)
            ax.text(-.3, 1.17, "(a) Expansion against budget $N$", transform=ax.transAxes, fontweight="bold", va="bottom")
        if i == 1:
            ax.set_xlabel("Roots $N$", labelpad=1, x=1.05)
    fig.legend(handles=[Line2D([], [], color=GREY, marker="s", ms=2.8, lw=.9, label="Uniform roots"),
                        Line2D([], [], color=NAVY, marker="D", ms=2.8, lw=.9, label="Roots from one region")],
               ncol=2, loc="upper left", bbox_to_anchor=(.205, 1.03), frameon=False, handlelength=1.5, columnspacing=.8, handletextpad=.3)
    first = None
    for i, (g, nm) in enumerate((("arxiv", "Arxiv"), ("patents", "Patents"), ("web-google", "Web-Google"), ("youtube", "YouTube"))):
        ax = fig.add_subplot(gs[0, 5 + i], sharey=first)
        first = first or ax
        full, sel, whole, oom = e3a(g)
        p = sorted(sel)
        ax.plot(p, [sel[x] / 1e3 for x in p], color=NAVY, marker="D", ms=2.6, lw=.9)
        ax.plot([p[0], p[-1]], [full / 1e3] * 2, color=GREY, marker="s", ms=2.6, lw=.9)
        ax.axhline(whole / 1e3, color=INK, lw=.6, ls=(0, (3, 2)))
        tag = "OOM: all" if len(oom) == 4 else "OOM: " + ", ".join(m.upper() for m in oom)
        ax.text(.97, whole / 1e3 * 1.18, tag, transform=ax.get_yaxis_transform(), ha="right", va="bottom", fontsize=5.8, color=INK)
        ax.set_xscale("log")
        ax.set_xticks([1, 10], ["1", "10"])
        ax.minorticks_off()
        ax.set_yscale("log")
        ax.set_ylim(20, 1.2e4)
        ax.set_yticks([100, 1000, 10000], ["100K", "1M", "10M"])
        ax.minorticks_off()
        ax.set_title(nm, pad=2)
        LOG.append(f"- E3a {g} N=2048: full {full:.0f}, whole {whole:.0f}, OOM {oom}; GIST " + ", ".join(f"{x}%: {sel[x]:.0f}" for x in p))
        if i:
            ax.tick_params(labelleft=False)
        else:
            ax.set_ylabel("Nodes per update", labelpad=1.5)
            ax.text(-.32, 1.17, "(b) Update size vs region size", transform=ax.transAxes, fontweight="bold", va="bottom")
        if i == 1:
            ax.set_xlabel("Region size (% of graph)", labelpad=1, x=1.05)
    fig.legend(handles=[Line2D([], [], color=GREY, marker="s", ms=2.8, lw=.9, label="Full"),
                        Line2D([], [], color=NAVY, marker="D", ms=2.8, lw=.9, label="GIST"),
                        Line2D([], [], color=INK, lw=.6, ls=(0, (3, 2)), label="Whole graph")],
               ncol=3, loc="upper right", bbox_to_anchor=(1.0, 1.03), frameon=False, handlelength=1.5, columnspacing=.8, handletextpad=.3)
    save(fig, "fig_cost_scaling")


# --- fig:timeline ----------------------------------------------------------------------------------------------------
def timeline():
    W = 50
    kernel = np.ones(W) / W
    fig, axes = plt.subplots(1, 2, figsize=(3.4, 1.62), sharey=True)
    fig.subplots_adjust(left=.115, right=.985, top=.79, bottom=.2, wspace=.07)
    LOG.append("\n## fig_timeline (pp against Frozen, mean over backbones x seeds; whole-stream mean must match tab:accuracy)\n")
    for ax, stream, ttl in zip(axes, ("arxiv_year", "arxiv"), ("(a) Arxiv-year (natural drift)", "(b) Arxiv (injected drift)")):
        cur = {a: [] for a in ("full", "selective", "roland")}
        rep, drift = [], []
        for m in ("gat", "sage", "gcn", "gin"):
            d = load(ROOT / f"results/D1_main/raw/streams/D1/{m}_{stream}.json")
            base = {r["seed"]: np.asarray(r["step_accuracy"], float) for r in d["rows"] if r["arm"] == "baseline"}
            for arm in cur:
                # as tab:accuracy: pooled over backbones x seeds, without the seeds whose update ran out of memory
                cur[arm] += [100 * (np.asarray(r["step_accuracy"], float) - base[r["seed"]]) for r in d["rows"] if r["arm"] == arm and not r.get("oom")]
            rep += [s for r in d["rows"] if r["arm"] == "full" for s in r["retrain_steps"]]
            if stream == "arxiv":
                drift += [s for v in d["meta"].values() for s in v["drift_steps"]]
        for arm, c, lab, lw in (("roland", GREEN, "ROLAND", .8), ("full", GREY, "Full", 1.0), ("selective", NAVY, "GIST", 1.0)):
            y = np.mean(cur[arm], axis=0)
            z = np.convolve(y, kernel, mode="valid")
            ax.plot(np.arange(W - 1, len(y)), z, color=c, lw=lw, label=lab, ls=(0, (3, 1.5)) if arm == "roland" else "-")
            LOG.append(f"- {stream} {arm}: {len(cur[arm])} backbone-seed pairs, whole-stream mean {y.mean():+.2f} pp; smoothed min {z.min():+.2f} max {z.max():+.2f}; "
                       f"first half {y[:500].mean():+.2f}, second half {y[500:].mean():+.2f}")
        ax.axhline(0, color=INK, lw=.5)
        ax.text(990, .07, "Frozen", ha="right", va="bottom", fontsize=5.8, color=INK)
        h, _ = np.histogram(rep, bins=np.arange(0, 1001, 25))
        ax.bar(np.arange(12.5, 1000, 25), -2.6 + 0.5 * h / h.max(), bottom=0, width=0, lw=0)
        ax.vlines(np.arange(12.5, 1000, 25), -2.6, -2.6 + .55 * h / h.max(), color=GREY, lw=1.0, alpha=.7)
        if drift:
            hd, _ = np.histogram(drift, bins=np.arange(0, 1001, 25))
            ax.vlines(np.arange(12.5, 1000, 25), 2.3 - .35 * hd / hd.max(), 2.3, color=VERM, lw=1.0, alpha=.8)
            ax.text(25, 2.33, "drift events", fontsize=5.6, color=VERM, va="bottom", ha="left")
        ax.text(20, -2.0, "Full's repairs", fontsize=5.6, color=GREY, ha="left", va="bottom")
        ax.set_xlim(0, 1000)
        ax.set_ylim(-2.6, 2.7)
        ax.set_xticks([0, 500, 1000], ["0", "500", "1K"])
        ax.set_xlabel("Stream step", labelpad=1)
        ax.set_title(ttl, pad=2, fontsize=6.6, fontweight="bold")
    axes[0].set_ylabel(f"Accuracy $-$ Frozen (pp)", labelpad=1.5)
    handles, labels = axes[0].get_legend_handles_labels()
    order = [2, 1, 0]
    fig.legend([handles[i] for i in order], [labels[i] for i in order], ncol=3, loc="upper center", bbox_to_anchor=(.55, 1.02),
               frameon=False, handlelength=1.5, columnspacing=1, handletextpad=.3)
    save(fig, "fig_timeline")


# --- fig:frontier ----------------------------------------------------------------------------------------------------
def frontier():
    fig, axes = plt.subplots(2, 2, figsize=(3.4, 2.5), sharex=True, sharey=True)
    fig.subplots_adjust(left=.115, right=.985, top=.885, bottom=.125, wspace=.07, hspace=.17)
    SUF = {0: "_rho0", .25: "", .5: "_rho0.5"}
    STY = {2048: ("o", GREEN), 4096: ("s", VERM), 8192: ("D", NAVY)}
    LOG.append("\n## fig_frontier (GIST - Full pp +- 95% CI / nodes ratio)\n")

    def point(m, s, N, suf):
        d = load(ROOT / f"results/D4_budget/raw/streams/D4/{m}_{s}_N{N}{suf}.json")
        g = {r["seed"]: r for r in d["rows"] if r["arm"] == "selective"}
        f = {r["seed"]: r for r in d["rows"] if r["arm"] == "full"}
        if not f:                     # the rho variants reuse the Full arm of the default run at this N
            f = {r["seed"]: r for r in load(ROOT / f"results/D4_budget/raw/streams/D4/{m}_{s}_N{N}.json")["rows"] if r["arm"] == "full"}
        k = sorted(set(g) & set(f))
        assert len(k) == 10, (m, s, N, suf, len(k))
        gap = [100 * (g[i]["accuracy"] - f[i]["accuracy"]) for i in k]
        return float(np.mean([g[i]["nodes"] for i in k]) / np.mean([f[i]["nodes"] for i in k])), float(np.mean(gap)), ci(gap)

    for r, (s, sn) in enumerate((("arxiv", "Arxiv"), ("arxiv_year", "Arxiv-year"))):
        for c, m in enumerate(("gat", "sage")):
            ax = axes[r, c]
            ax.axhline(0, color=INK, lw=.5)
            for N, (mk, col) in STY.items():
                pts = [point(m, s, N, SUF[rho]) for rho in (0, .25, .5)]
                ax.errorbar([p[0] for p in pts], [p[1] for p in pts], yerr=[p[2] for p in pts], color=col, marker=mk, ms=2.8, lw=.8,
                            elinewidth=.45, capsize=1, capthick=.4)
                LOG.append(f"- {m} {s} N={N}: " + ", ".join(f"rho={rho}: {p[1]:+.2f}+-{p[2]:.2f} / {p[0]:.3f}" for rho, p in zip((0, .25, .5), pts)))
                if s == "arxiv":
                    q = point(m, s, N, "_diffuse")
                    ax.errorbar([q[0]], [q[1]], yerr=[q[2]], color=col, marker=mk, ms=3.2, mfc="white", mew=.7, elinewidth=.45, capsize=1, capthick=.4, ls="none")
                    LOG.append(f"- {m} {s} N={N} diffuse: {q[1]:+.2f}+-{q[2]:.2f} / {q[0]:.3f}")
            ax.set_xlim(.2, .95)
            ax.set_ylim(-4.0, .5)
            ax.set_yticks([-3, -2, -1, 0])
            ax.set_title(f"{NAME[m]}, {sn}", pad=2)
            for x, t in ((.31, r"$\rho=0$"), (.66, r"$\rho=0.25$"), (.84, r"$\rho=0.5$")):
                ax.text(x, -3.95, t, ha="center", va="bottom", fontsize=5.8, color=INK)
    fig.supxlabel("Nodes relative to Full", fontsize=7, y=.015, x=.55)
    fig.supylabel("GIST $-$ Full (pp)", fontsize=7, x=.005, y=.52)
    h = [Line2D([], [], color=STY[N][1], marker=STY[N][0], ms=2.8, lw=.8, label=f"$N={N:,}$".replace(",", "{,}")) for N in STY]
    h.append(Line2D([], [], color=INK, marker="o", ms=3, mfc="white", mew=.7, ls="none", label="Diffuse"))
    fig.legend(handles=h, ncol=4, loc="upper center", bbox_to_anchor=(.55, 1.015), frameon=False, handlelength=1.5, columnspacing=.9, handletextpad=.3)
    save(fig, "fig_frontier")


# --- fig:families ----------------------------------------------------------------------------------------------------
def families():
    rows = []
    for line in (ROOT / "results/D6_graphs/d6.md").read_text().split("## Detection")[0].splitlines():
        m = re.match(r"\| (social|web|p2p|road|citation) \| ([\w-]+) \|", line)
        if not m:
            continue
        c = [x.strip() for x in line.strip("|").split("|")]
        g, gci = re.match(r"([+-][\d.]+) ± ([\d.]+)", c[4]).groups()
        rows.append(dict(family=m.group(1), graph=m.group(2), G=float(g), Gci=float(gci), cost=float(c[10])))
    assert len(rows) == 15
    for r in rows:
        N = load(ROOT / f"results/D6_graphs/raw/gat_{r['graph']}_final.json")["cfg"]["budget"]
        s = e1_share(r["graph"])
        near = min(s["uniform"], key=lambda n: abs(np.log2(n) - np.log2(N)))
        r.update(N=N, E1N=near, uniform=s["uniform"][near], region=s["region"][near], x=s["uniform"][near] / s["region"][near])
    LOG.append("\n## fig_families (x = uniform / region 2-hop share at the E1 budget nearest the graph's D6 budget)\n")
    for r in sorted(rows, key=lambda r: r["cost"]):
        LOG.append(f"- {r['family']} {r['graph']}: D6 N={r['N']} E1 N={r['E1N']} uniform {100 * r['uniform']:.1f}% region {100 * r['region']:.1f}% x={r['x']:.2f} cost {r['cost']:.3f} G {r['G']:+.2f}+-{r['Gci']:.2f}")
    no_road = [r for r in rows if r["family"] != "road"]
    a = stats.spearmanr([r["x"] for r in rows], [r["cost"] for r in rows])
    b = stats.spearmanr([r["x"] for r in no_road], [r["cost"] for r in no_road])
    LOG.append(f"- Spearman x vs cost: all 15 rho={a.statistic:.2f} p={a.pvalue:.3f}; without road (12) rho={b.statistic:.2f} p={b.pvalue:.4f}")

    FAM = dict(social=("o", VERM, "Social"), web=("s", NAVY, "Web"), citation=("^", GREEN, "Citation"), p2p=("v", PINK, "P2P"), road=("X", INK, "Road"))
    fig, (ax, bx) = plt.subplots(1, 2, figsize=(3.4, 2.0), gridspec_kw=dict(width_ratios=[1.45, 1]))
    fig.subplots_adjust(left=.105, right=.985, top=.865, bottom=.165, wspace=.52)
    for r in rows:
        mk, col, _ = FAM[r["family"]]
        ax.plot(r["x"], r["cost"], marker=mk, color=col, ms=3.6, ls="none", mew=.3)
    OFF = {"web-google": (3, -4), "web-stanford": (3, 2), "web-berkstan": (-4, -5), "patents": (3, 1), "youtube": (3, 0), "road-pa": (-1, 4)}
    NM = {"web-google": "Web-Google", "web-stanford": "Web-Stanford", "web-berkstan": "Web-BerkStan", "patents": "Patents", "youtube": "YouTube", "road-pa": "Road"}
    for r in rows:
        if r["graph"] in OFF:
            dx, dy = OFF[r["graph"]]
            ax.annotate(NM[r["graph"]], (r["x"], r["cost"]), xytext=(dx, dy), textcoords="offset points", fontsize=5.6, ha="right" if dx < 0 else "left", color=INK)
    ax.set_xscale("log", base=2)
    ax.set_xticks([1, 2, 4, 8], ["1", "2", "4", "8"])
    ax.minorticks_off()
    ax.set_xlim(1, 11)
    ax.set_ylim(.45, 1.03)
    ax.set_xlabel("Uniform / region 2-hop share", labelpad=1)
    ax.set_ylabel("GIST nodes / Full nodes", labelpad=1.5)
    ax.set_title("(a) Saving vs over-expansion", pad=2, fontweight="bold", fontsize=6.6)
    order = ["social", "web", "citation", "p2p", "road"]
    seq = [r for f in order for r in rows if r["family"] == f][::-1]
    for i, r in enumerate(seq):
        mk, col, _ = FAM[r["family"]]
        bx.errorbar(r["G"], i, xerr=r["Gci"], color=col, marker=mk, ms=2.8, elinewidth=.5, ls="none", mew=.3)
    bx.axvline(0, color=INK, lw=.5)
    bx.set_yticks(range(15), [r["graph"].replace("gnutella", "gnut.").replace("web-", "w-").replace("road-", "r-") for r in seq], fontsize=5.3)
    bx.tick_params(axis="y", pad=1)
    bx.set_ylim(-.7, 14.7)
    bx.grid(axis="y", visible=False)
    bx.set_xlim(-1.8, 3.0)
    bx.set_xticks([-1, 0, 1, 2])
    bx.set_xlabel("Full $-$ Frozen (MRR pts)", labelpad=1)
    bx.set_title("(b) Does retraining pay?", pad=2, fontweight="bold", fontsize=6.6)
    h = [Line2D([], [], color=FAM[f][1], marker=FAM[f][0], ms=3.4, ls="none", label=FAM[f][2]) for f in order]
    fig.legend(handles=h, ncol=5, loc="upper center", bbox_to_anchor=(.53, 1.02), frameon=False, handletextpad=0, columnspacing=.7)
    save(fig, "fig_families")


# --- fig:memory ------------------------------------------------------------------------------------------------------
def memory():
    fig, axes = plt.subplots(2, 2, figsize=(3.4, 2.65), sharex=True, sharey=True)
    fig.subplots_adjust(left=.115, right=.985, top=.84, bottom=.125, wspace=.07, hspace=.17)
    H = (256, 512, 1024)
    LS = {2048: (0, (1, 1)), 8192: (0, (3, 1.5)), 32768: "-"}
    OFF = {2048: -.09, 8192: 0, 32768: .09}
    ARM = {"full": (GREY, "s", GREY), "selective": (NAVY, "D", NAVY), "global": (INK, "s", "white")}
    LOG.append("\n## fig_memory (median peak GB over non-OOM seeds; n OOM of seeds)\n")
    for r, g in enumerate(("arxiv", "products")):
        for c, m in enumerate(("gat", "sage")):
            ax = axes[r, c]
            rows = load(ROOT / f"results/D7_limits/raw/D7/memory_{g}_{m}.json")["rows"]
            ax.axhline(48, color=VERM, lw=.6, ls=(0, (3, 2)))
            for arm, (col, mk, face) in ARM.items():
                for N, ls in LS.items():
                    xs, ys = [], []
                    for i, h in enumerate(H):
                        sel = [x for x in rows if x["method"] == arm and x["N"] == N and x["hidden"] == h]
                        if not sel:
                            continue
                        ok = [x["peak_mb"] / 1000 for x in sel if not x["oom"] and x.get("peak_mb")]
                        noom = len(sel) - len(ok)
                        LOG.append(f"- {g} {m} {arm} N={N} h={h}: {'%.2f GB' % np.median(ok) if ok else 'OOM'} (OOM {noom}/{len(sel)})")
                        if ok and noom == 0:
                            xs.append(i + OFF[N]); ys.append(float(np.median(ok)))
                        elif ok:                 # some seeds fit: draw the point, flag the partial failure
                            xs.append(i + OFF[N]); ys.append(float(np.median(ok)))
                            ax.annotate(f"OOM {noom}/{len(sel)}", (i + OFF[N], ys[-1]), xytext=(-2, 3), textcoords="offset points", fontsize=5.2, ha="right", color=col)
                        else:
                            ax.plot(i + OFF[N], 75 if arm == "global" else (62 if arm == "full" else 92), marker="x", color=col, ms=3, mew=.8, clip_on=False)
                    if xs:
                        ax.plot(xs, ys, color=col, ls=ls, lw=.8, marker=mk, ms=2.6, mfc=face, mew=.6)
            ax.set_yscale("log")
            ax.set_ylim(.12, 120)
            ax.set_yticks([.1, 1, 10, 48], ["0.1", "1", "10", "48"])
            ax.minorticks_off()
            ax.set_xticks(range(3), [str(h) for h in H])
            ax.set_xlim(-.3, 2.3)
            ax.set_title(f"{NAME[m]}, {'Arxiv' if g == 'arxiv' else 'Products'}", pad=2)
    fig.supxlabel("Hidden width", fontsize=7, y=.015, x=.55)
    fig.supylabel("Peak repair memory (GB)", fontsize=7, x=.005, y=.5)
    h = [Line2D([], [], color=GREY, marker="s", ms=2.8, ls="none", label="Full"), Line2D([], [], color=NAVY, marker="D", ms=2.8, ls="none", label="GIST"),
         Line2D([], [], color=INK, marker="s", mfc="white", ms=2.8, mew=.6, ls="none", label="Whole graph"), Line2D([], [], color=INK, marker="x", ms=3, mew=.8, ls="none", label="OOM")]
    h2 = [Line2D([], [], color=INK, ls=LS[N], lw=.8, label=f"$N={N:,}$".replace(",", "{,}")) for N in LS]
    l1 = fig.legend(handles=h, ncol=4, loc="upper center", bbox_to_anchor=(.55, 1.02), frameon=False, handletextpad=.2, columnspacing=.9)
    fig.add_artist(l1)
    fig.legend(handles=h2, ncol=3, loc="upper center", bbox_to_anchor=(.55, .96), frameon=False, handlelength=2.2, handletextpad=.3, columnspacing=.9)
    save(fig, "fig_memory")


def main():
    cost_scaling()
    timeline()
    frontier()
    families()
    memory()
    (OUT / "fig_eval_values.md").write_text("\n".join(LOG) + "\n")
    print("\n".join(LOG))


if __name__ == "__main__":
    main()
