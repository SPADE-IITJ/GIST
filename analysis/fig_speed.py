"""fig:speed of the GIST ICDE paper: repair time vs budget and serving latency, Phase D timing runs.

    OMP_NUM_THREADS=8 python analysis/fig_speed.py

Reads only results/D3_speed/raw/<model>.json (Arxiv injected, seeds 0-4, N in 2048/4096/8192, sync-on stage times,
sync-off wall clock) and results/D5_intro/raw/E7/<model>_arxiv.json (seeds 0-4). Rows marked contended are refused.
Writes fig_speed.pdf, fig_speed.png (600 dpi), speed_data.csv and speed_tables.md to results/D3_speed/figures/.
Per-seed statistics: speedup is the mean of per-seed ratios; error bars are the min-max over seeds.
"""
from __future__ import annotations

import csv
import json
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager
from matplotlib.lines import Line2D
from matplotlib.legend_handler import HandlerTuple
from matplotlib.patches import Patch

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "results" / "D3_speed" / "raw"
E7 = ROOT / "results" / "D5_intro" / "raw" / "E7"
OUT = ROOT / "results" / "D3_speed" / "figures"
MODELS = ("gat", "sage", "gcn", "gin")
NAME = dict(gat="GAT", sage="SAGE", gcn="GCN", gin="GIN")
BUDGETS = (2048, 4096, 8192)
NAVY, GREY = "#0B2D5B", "#7A7A7A"
NAVY_T, GREY_T = "#9FB3CF", "#D2D2D2"
SHAPE = dict(gat="o", sage="s", gcn="^", gin="v")
COLOR = dict(gat="#0B2D5B", sage="#D55E00", gcn="#009E73", gin="#CC79A7")

SERIF = next((f for f in ("Times New Roman", "Nimbus Roman", "Nimbus Roman No9 L", "Liberation Serif")
              if any(e.name == f for e in font_manager.fontManager.ttflist)), "DejaVu Serif")
plt.rcParams.update({
    "font.family": SERIF, "font.size": 7, "axes.titlesize": 7, "axes.labelsize": 7, "xtick.labelsize": 6.5,
    "ytick.labelsize": 6.5, "legend.fontsize": 6.5, "pdf.fonttype": 42, "ps.fonttype": 42, "axes.linewidth": 0.5,
    "xtick.major.width": 0.5, "ytick.major.width": 0.5, "xtick.major.size": 2, "ytick.major.size": 2,
    "axes.spines.top": False, "axes.spines.right": False, "axes.grid": True, "axes.axisbelow": True,
    "mathtext.fontset": "stix", "grid.linewidth": 0.3, "grid.color": "#DDDDDD",
})


def ms(r, stage, field="p50_ms"):
    return r["latency"].get(stage, {}).get(field, 0.0) or 0.0


def load_d3():
    out = {}
    for m in MODELS:
        d = json.load(open(RAW / f"{m}.json"))
        rows = [r for r in d["rows"] if r["kind"] == "main"]
        assert not any(r["contended"] for r in d["rows"]), m
        out[m] = (rows, [r for r in d["rows"] if r["kind"] == "repeat"])
    return out


def pick(rows, **k):
    return sorted((r for r in rows if all(r[a] == b for a, b in k.items())), key=lambda r: r["seed"])


def summarise(d3):
    """One record per model x budget; every value is per-seed then averaged over seeds 0-4."""
    rec = []
    for m in MODELS:
        rows = d3[m][0]
        for N in BUDGETS:
            f = pick(rows, arm="full", budget=N, sync_stages=True, detection=True)
            g = pick(rows, arm="selective", budget=N, sync_stages=True, detection=True)
            assert [r["seed"] for r in f] == [r["seed"] for r in g] == list(range(5)), (m, N)
            ff, gg = [ms(r, "retrain") for r in f], [ms(r, "retrain") for r in g]
            ratio = np.array(ff) / np.array(gg)
            comm = [1000 * (ms(r, "community", "total_s") + ms(r, "community_bg", "total_s")) / max(r["events"], 1) for r in g]
            net = np.array(ff) / (np.array(gg) + np.array(comm))
            wall = lambda arm, det=True: [r["wall_s"] for r in pick(rows, arm=arm, budget=N, sync_stages=False, detection=det)]
            wf, wg, wp = wall("full"), wall("selective"), wall("periodic", False)
            rec.append(dict(
                model=m, N=N,
                full_ms=float(np.mean(ff)), full_min=min(ff), full_max=max(ff),
                gist_ms=float(np.mean(gg)), gist_min=min(gg), gist_max=max(gg),
                speedup=float(ratio.mean()), speedup_sd=float(ratio.std(ddof=1)), speedup_min=float(ratio.min()),
                speedup_max=float(ratio.max()), speedup_net=float(net.mean()), community_ms=float(np.mean(comm)),
                node_ratio=float(np.mean([b["nodes"] / a["nodes"] for a, b in zip(f, g)])),
                full_nodes=float(np.mean([r["nodes"] for r in f])), gist_nodes=float(np.mean([r["nodes"] for r in g])),
                # mean over events, so the two parts add up to the mean repair (not the p50)
                full_sample=float(np.mean([r["region_sample_ms"] for r in f])),
                full_update=float(np.mean([ms(r, "retrain", "mean_ms") - r["region_sample_ms"] for r in f])),
                gist_sample=float(np.mean([r["region_sample_ms"] for r in g])),
                gist_update=float(np.mean([ms(r, "retrain", "mean_ms") - r["region_sample_ms"] for r in g])),
                wall_full=float(np.mean(wf)), wall_gist=float(np.mean(wg)), wall_periodic_nodet=float(np.mean(wp)),
                gist_lt_full=sum(a < b for a, b in zip(wg, wf)),
                gist_le_periodic=sum(a <= b for a, b in zip(wg, wp)),
            ))
    return rec


def overhead(d3):
    out = {}
    for m in MODELS:
        rows, rep = d3[m]
        b = pick(rows, arm="baseline", sync_stages=True, detection=True)
        on, off = pick(rows, arm="baseline", sync_stages=False, detection=True), pick(rows, arm="baseline", sync_stages=False, detection=False)
        twin = pick(rows, seed=rep[0]["seed"], arm=rep[0]["arm"], budget=rep[0]["budget"], sync_stages=rep[0]["sync_stages"], detection=rep[0]["detection"])[0]
        out[m] = dict(
            infer=float(np.mean([ms(r, "infer") for r in b])), detect=float(np.mean([ms(r, "detect") for r in b])),
            sample=float(np.mean([ms(r, "sample") for r in b])),
            pct_infer=100 * float(np.mean([ms(r, "detect") / ms(r, "infer") for r in b])),
            pct_step=100 * float(np.mean([ms(r, "detect", "total_s") / r["wall_s"] for r in b])),
            wall_on=float(np.mean([r["wall_s"] for r in on])), wall_off=float(np.mean([r["wall_s"] for r in off])),
            repeat_gap=abs(rep[0]["wall_s"] - twin["wall_s"]),
        )
    return out


def load_e7():
    out = {}
    for m in ("gat", "sage"):
        e = json.load(open(E7 / f"{m}_arxiv.json"))
        by = {}
        for rows in e["seeds"].values():
            for r in rows:
                by.setdefault(r["condition"], []).append(r)
        out[m] = {c: dict(
            n=len(g), oom=sum(r["repair_oom"] for r in g),
            p50=float(np.mean([r["p50_ms"] for r in g])), p99=float(np.mean([r["p99_ms"] for r in g])),
            requests=float(np.mean([r["requests"] for r in g])),
            step=float(np.mean([r["repair_step_p50_ms"] for r in g if r["repair_step_p50_ms"] is not None])) if not all(r["repair_step_p50_ms"] is None for r in g) else None,
            nodes=None if g[0]["nodes"] is None else float(np.mean([r["nodes"] for r in g])),
        ) for c, g in by.items()}
    return out


def k_fmt(N):
    return f"{N // 1024}K"


def plot(rec, e7):
    fig = plt.figure(figsize=(7.0, 1.9))
    # (a) four panels, (b), (c), (d) two panels; spacer columns between the groups
    gs = fig.add_gridspec(1, 12, width_ratios=[.78, .78, .78, .78, .44, 1.08, .44, 1.42, .44, 1.02, .28, 1.02],
                          left=.05, right=.997, top=.80, bottom=.2, wspace=.10)
    get = lambda m, N: next(r for r in rec if r["model"] == m and r["N"] == N)

    # (a) repair time vs budget, one panel per backbone, shared y
    first = None
    for i, m in enumerate(MODELS):
        ax = fig.add_subplot(gs[0, i], sharey=first)
        first = first or ax
        x = np.arange(len(BUDGETS))
        for arm, c, mk, kw in (("full", GREY, "s", dict(markerfacecolor=GREY)), ("gist", NAVY, "D", dict(markerfacecolor=NAVY))):
            y = [get(m, N)[f"{arm}_ms"] for N in BUDGETS]
            lo = [get(m, N)[f"{arm}_ms"] - get(m, N)[f"{arm}_min"] for N in BUDGETS]
            hi = [get(m, N)[f"{arm}_max"] - get(m, N)[f"{arm}_ms"] for N in BUDGETS]
            ax.errorbar(x, y, yerr=[lo, hi], color=c, marker=mk, ms=3, lw=.9, elinewidth=.5, capsize=1, capthick=.5, **kw)
        ax.set_title(NAME[m], pad=2)
        ax.set_xticks(x, [k_fmt(N) for N in BUDGETS])
        ax.set_xlim(-.35, 2.35)
        ax.set_ylim(0, 340)
        if i:
            ax.tick_params(labelleft=False)
        else:
            ax.set_ylabel("Repair time (ms)", labelpad=1.5)
            ax.text(-.5, 1.16, "(a) Repair time vs budget", transform=ax.transAxes, fontweight="bold", va="bottom")
        if i == 1:
            ax.set_xlabel("Budget $N$ (roots per repair)", labelpad=1, x=1.05)

    # (b) speedup vs budget
    ax = fig.add_subplot(gs[0, 5])
    x = np.arange(len(BUDGETS))
    for m in MODELS:
        ax.plot(x, [get(m, N)["speedup"] for N in BUDGETS], marker=SHAPE[m], ms=3.2, lw=.9, color=COLOR[m], label=NAME[m])
    ax.axhline(1, color="#444444", lw=.5, ls=(0, (3, 2)))
    ax.text(2.3, 1.015, "no speedup", ha="right", va="bottom", fontsize=6, color="#444444")
    ax.set_xticks(x, [k_fmt(N) for N in BUDGETS])
    ax.set_xlim(-.2, 2.35)
    ax.set_ylim(.9, 2.25)
    ax.set_yticks([1, 1.25, 1.5, 1.75, 2], ["1", "1.25", "1.5", "1.75", "2"])
    ax.set_xlabel("Budget $N$", labelpad=1)
    ax.set_ylabel("Full / GIST", labelpad=1.5)
    ax.legend(ncol=2, loc="upper left", frameon=False, handlelength=1.2, columnspacing=.7, borderaxespad=.05, handletextpad=.3, labelspacing=.15)
    ax.text(-.3, 1.16, "(b) Speedup", transform=ax.transAxes, fontweight="bold", va="bottom")

    # (c) what a repair spends its time on at N = 8192
    ax = fig.add_subplot(gs[0, 7])
    w = .36
    for i, m in enumerate(MODELS):
        r = get(m, 8192)
        for off, arm, dark, tint in ((-w / 2 - .02, "full", GREY, GREY_T), (w / 2 + .02, "gist", NAVY, NAVY_T)):
            ax.bar(i + off, r[f"{arm}_sample"], w, color=tint, lw=0)
            ax.bar(i + off, r[f"{arm}_update"], w, bottom=r[f"{arm}_sample"], color=dark, lw=0)
    ax.set_xticks(range(4), [NAME[m] for m in MODELS], fontsize=6)
    ax.set_xlim(-.55, 3.55)
    ax.set_ylim(0, 440)
    ax.set_yticks([0, 100, 200, 300])
    ax.grid(axis="x", visible=False)
    ax.set_ylabel("Mean repair (ms)", labelpad=1.5)
    ax.legend(handles=[(Patch(color=GREY), Patch(color=NAVY)), (Patch(color=GREY_T), Patch(color=NAVY_T))],
              labels=["Rest of repair", "Host sampling"], handler_map={tuple: HandlerTuple(ndivide=None, pad=0)},
              loc="upper right", frameon=False, handlelength=1.8, handleheight=.8, borderaxespad=.05,
              handletextpad=.4, labelspacing=.15, fontsize=6)
    ax.text(-.3, 1.16, "(c) Split at $N=8192$", transform=ax.transAxes, fontweight="bold", va="bottom")

    # (d) serving p99 next to a continuous repair, own y scale per model; the values are printed
    conds = (("idle", "Idle", GREY_T, GREY_T), ("full", "Full", GREY, GREY), ("selective", "GIST", NAVY, NAVY), ("global", "Whole", "white", GREY))
    for j, m in enumerate(("gat", "sage")):
        ax = fig.add_subplot(gs[0, 9 + 2 * j])
        top = max(e7[m][c]["p99"] for c, *_ in conds)
        for i, (c, _, face, edge) in enumerate(conds):
            v = e7[m][c]
            if v["oom"]:
                assert v["oom"] == v["n"], (m, c)        # never a partial OOM: then the bar would mean something
                ax.plot([i], [top * .1], marker="x", color=GREY, ms=4, mew=1)
                ax.text(i, top * .19, "OOM", ha="center", va="bottom", fontsize=6)
                continue
            ax.bar(i, v["p99"], .72, color=face, edgecolor=edge, lw=.5)
            ax.text(i, v["p99"] + top * .025, f"{v['p99']:.1f}" if m == "gat" else f"{v['p99']:.2f}", ha="center", va="bottom", fontsize=5.5)
        ax.set_xticks(range(4), [n for _, n, *_ in conds], fontsize=5.8, rotation=90)
        ax.tick_params(axis="x", pad=1)
        ax.set_xlim(-.6, 3.6)
        ax.set_ylim(0, top * 1.17)
        ax.grid(axis="x", visible=False)
        ax.set_title(NAME[m], pad=2)
        if j == 0:
            ax.set_ylabel("Serving p99 (ms)", labelpad=1.5)
            ax.text(-.4, 1.16, "(d) Serving during repair", transform=ax.transAxes, fontweight="bold", va="bottom")

    fig.legend(handles=[Line2D([], [], color=GREY, marker="s", ms=3, lw=.9, label="Full"),
                        Line2D([], [], color=NAVY, marker="D", ms=3, lw=.9, label="GIST")],
               ncol=2, loc="upper left", bbox_to_anchor=(.215, 1.01), frameon=False, handlelength=1.5, columnspacing=.8, handletextpad=.3)
    return fig


def tables(rec, ov, e7):
    get = lambda m, N: next(r for r in rec if r["model"] == m and r["N"] == N)
    L = ["## Repair time (sync on, ms, mean of seed p50s; speedup mean of per-seed ratios +- sd)\n"]
    for m in MODELS:
        L.append(f"- {NAME[m]}: " + " | ".join(
            f"N={N}: {get(m, N)['full_ms']:.0f}->{get(m, N)['gist_ms']:.0f} x{get(m, N)['speedup']:.2f}+-{get(m, N)['speedup_sd']:.2f} "
            f"[{get(m, N)['speedup_min']:.2f},{get(m, N)['speedup_max']:.2f}] net of community x{get(m, N)['speedup_net']:.2f} "
            f"nodes {get(m, N)['node_ratio'] * 100:.1f}% ({get(m, N)['full_nodes'] / 1000:.1f}K->{get(m, N)['gist_nodes'] / 1000:.1f}K) community {get(m, N)['community_ms']:.1f} ms"
            for N in BUDGETS))
    L.append("\n## Mean repair split (ms): sampling + model update\n")
    for m in MODELS:
        L.append(f"- {NAME[m]}: " + " | ".join(
            f"N={N}: Full {get(m, N)['full_sample']:.0f}+{get(m, N)['full_update']:.0f} GIST {get(m, N)['gist_sample']:.0f}+{get(m, N)['gist_update']:.0f}" for N in BUDGETS))
    L.append("\n## End to end (sync off, ms/step)\n")
    for m in MODELS:
        L.append(f"- {NAME[m]}: " + " | ".join(
            f"N={N}: full {get(m, N)['wall_full']:.2f} gist {get(m, N)['wall_gist']:.2f} periodic-no-detect {get(m, N)['wall_periodic_nodet']:.2f} "
            f"gist<full {get(m, N)['gist_lt_full']}/5 gist<=periodic {get(m, N)['gist_le_periodic']}/5" for N in BUDGETS))
    L.append("\n## Overhead on the inference path (baseline arm)\n")
    for m in MODELS:
        o = ov[m]
        L.append(f"- {NAME[m]}: infer p50 {o['infer']:.2f} detect p50 {o['detect']:.2f} = {o['pct_infer']:.1f}% of infer, {o['pct_step']:.2f}% of step; "
                 f"sampling p50 {o['sample']:.1f}; wall/1000 steps det on {o['wall_on']:.2f} s off {o['wall_off']:.2f} s; repeat-run gap {o['repeat_gap']:.2f} s")
    L.append("\n## E7 serving (seeds 0-4, 20 s window per condition)\n")
    for m in ("gat", "sage"):
        idle = e7[m]["idle"]["p99"]
        for c, v in e7[m].items():
            L.append(f"- {NAME[m]} {c}: n={v['n']} oom={v['oom']} p50 {v['p50']:.2f} p99 {v['p99']:.2f} (x{v['p99'] / idle:.2f} idle) requests {v['requests']:.0f} "
                     f"repair step p50 {v['step'] if v['step'] is None else round(v['step'], 1)} nodes {v['nodes']}")
    return "\n".join(L) + "\n"


def main():
    OUT.mkdir(exist_ok=True)
    d3 = load_d3()
    rec = summarise(d3)
    ov = overhead(d3)
    e7 = load_e7()
    with open(OUT / "speed_data.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rec[0]))
        w.writeheader()
        w.writerows(rec)
    (OUT / "speed_tables.md").write_text(tables(rec, ov, e7))
    fig = plot(rec, e7)
    fig.savefig(OUT / "fig_speed.pdf")
    fig.savefig(OUT / "fig_speed.png", dpi=600)
    print((OUT / "speed_tables.md").read_text())


if __name__ == "__main__":
    main()
