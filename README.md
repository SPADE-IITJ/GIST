# GIST: Gradient-Identical Subgraph Training for Streaming Graph Neural Networks

GIST is a runtime that repairs a deployed graph neural network while it keeps serving predictions on an evolving graph.
When drift is detected, GIST updates the model's parameters on the subgraph induced by the $L$-hop neighborhood of the
nodes it retrains. Because the output of an $L$-layer message-passing GNN at a node depends only on its $L$-hop
neighborhood, the gradient computed on this subgraph is identical to the gradient on the parent graph, so the saving
comes only from work that the update never needed.

The paper is included as [`GIST.pdf`](GIST.pdf).

**Main results** (four backbones, six node-classification streams, fifteen link-prediction graphs, ten seeds):

- The localized update matches the parent-graph gradient to float64 round-off on every backbone, including normalized
  and snapshot-based temporal GNNs, and to at most $5.2\times10^{-14}$ in every audited repair inside the streams.
- At the same root budget as sampled retraining, a repair materializes up to 49% fewer nodes (29–36% fewer on the two
  Arxiv streams), uses up to 43% less GPU memory, and runs 1.59–1.77× faster at $N=8{,}192$.
- Holdout accuracy is statistically equivalent to sampled retraining (two one-sided tests, 2 pp margin) in 22 of 23
  backbone–stream cells.

## Repository layout

```
├── gist/
│   ├── stream/
│   ├── drift/
│   ├── adaptation/
│   ├── models/
│   ├── experiments/
│   └── evaluation/
├── configs/
├── experiments/
│   └── calibration/
├── linkpred/
├── analysis/
├── results/
│   ├── archives/
│   ├── summary/
│   └── figures/
├── scripts/
│   ├── commands/
│   └── prep_products.py
└── tests/
```

## Installation

Python 3.11. The evaluation used PyTorch 2.5.1 with CUDA 12.1 and PyTorch Geometric 2.8 on NVIDIA L40S GPUs.

```bash
python -m venv .venv && source .venv/bin/activate
pip install torch==2.5.1 --index-url https://download.pytorch.org/whl/cu121
pip install torch-scatter torch-sparse -f https://data.pyg.org/whl/torch-2.5.1+cu121.html
pip install -r requirements.txt
```

## Data

Datasets are stored under `data/` (git-ignored).

| Graph | How to obtain |
|---|---|
| Cora, PubMed, ogbn-arxiv | downloaded automatically on first use (PyG and OGB) |
| Elliptic | the Bitcoin transaction CSVs (Kaggle release, also hosted by PyG) in `data/elliptic/raw/` |
| ogbn-products | download `ogbn-products.zip` from the DGL GraphBolt mirror, then `python scripts/prep_products.py <unzipped dir> data/ogbn_products_pyg/data.pt` |
| SNAP graphs (link prediction) | the edge lists from https://snap.stanford.edu/data/ in `linkpred/data/raw/`, with the file names listed in `linkpred/data.py` |

## Quick start

Run the frozen model, sampled retraining (`full`), and GIST (`selective`) on 200 steps of the injected-drift Arxiv
stream for GAT:

```bash
python -m experiments.streams --exp demo --model gat --variant arxiv \
    --arms baseline full selective --seeds 110 --steps 200
```

The results are written to `experiments/results/streams/demo/gat_arxiv.json`: holdout accuracy over the stream, and
for every repair its nodes, FLOPs, peak memory, and time.

## Reproducing the paper

All reported runs use the evaluation seeds 0–9 and go through `experiments.final`, which writes to
`results/<experiment>/raw/`. Thresholds, warm-up lengths, and baseline hyperparameters were fixed beforehand on the
calibration seeds 100–102 and are stored in `experiments/calibration/`. The exact command line of every run is listed
in `scripts/commands/`:

| File | Experiments |
|---|---|
| `main_evaluation.txt` | main matrix (D1), exactness of every backbone (D2), repair speed and serving latency (D3), rehearsal share and budget (D4), cost-bounding baselines and evaluation practices (D5), link prediction (D6), memory and depth limits (D7) |
| `review_experiments.txt` | matched roots, whole-graph parent, timer and oracle triggers, cooldown and threshold sensitivity, Elliptic macro-F1, link-prediction audits (D8) |
| `exactness_ablation.txt` | repairs that break one exactness condition (D9) |
| `drift_ranges.txt` | drift size, sudden shocks, endless drift, and severity-adaptive scheduling (D10) |

For example, one cell of the main matrix:

```bash
python -m experiments.final D1_main streams --exp D1 --model gat --variant arxiv_year \
    --arms baseline full selective periodic random_region iso_full global_periodic global_trigger \
           roland ergnn ewc twp cat --seeds 0 1 2 3 4 5 6 7 8 9 --audit 3
```

Timing runs (D3) need a GPU to themselves, since the reported times come only from runs without other processes on
the card.

## Regenerating the figures and tables from the archived results

```bash
for a in results/archives/*.tar.gz; do tar -C results -xzf "$a"; done
python analysis/make_eval_figures.py        # evaluation figures -> results/figures/eval/
python analysis/make_intro_figures.py       # introduction figures -> results/figures/intro/
python analysis/agg_draft.py                # main matrix, exactness, budget, and limits tables
python analysis/agg_review.py               # review experiments
```

Every value plotted in a figure is also written next to it (`results/figures/*/*_values.md`).

## Tests

```bash
python -m pytest tests -q
```

The tests include the invariants of the evaluation protocol (one replayed stream per seed, matched budgets,
disjoint training and holdout nodes) and float64 checks that the update on the induced subgraph equals the gradient on
the parent graph for every supported backbone, while one hop less does not. Tests of the snapshot-based temporal
backbones are skipped unless `torch-geometric-temporal` is installed.

<!-- ## Citation

```bibtex
@inproceedings{gist2027,
  title     = {{GIST}: Gradient-Identical Subgraph Training for Streaming Graph Neural Networks},
  author    = {Narkhede, Sahil and Nijhara, Prajjwal and Banerjee, Dip Sankar},
  booktitle = {Proceedings of the IEEE International Conference on Data Engineering (ICDE)},
  year      = {2027}
}
``` -->