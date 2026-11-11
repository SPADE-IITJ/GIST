# Review experiments: matched roots, parent graph, trigger, Elliptic macro-F1, link-prediction audits

## R1. Matched roots, root rule × parent graph, timer trigger (Arxiv streams, seeds 0–9)

Nodes = mean nodes materialised per repair; ratio to `full` in brackets. |S| = distinct roots per repair. `*_exact` = parent is the whole current graph. Accuracy: pp against Frozen, mean ± 95% CI over seeds.

### arxiv_year

| model | GIST \|S\| / N (median, min–max per repair) | full | GIST (D1 rule) | GIST, \|S\|=N | Full, GIST's \|S\| | Full, whole-graph parent | GIST, whole-graph parent | OOM |
|---|---|---|---|---|---|---|---|---|
| GAT | 0.59 (0.27–0.98), 0% at N | 80.5K | 55.7K (0.69) | 72.6K (0.90) | 71.5K (0.89) | nanK (nan) | nanK (nan) | full_exact 10/10, selective_exact 10/10 |
| SAGE | 0.61 (0.27–0.98), 0% at N | 81.4K | 57.4K (0.71) | 71.6K (0.88) | 75.0K (0.92) | 93.8K (1.15) | 77.0K (0.95) | none |
| GCN | 0.60 (0.27–0.98), 0% at N | 80.9K | 56.6K (0.70) | 71.7K (0.89) | 73.4K (0.91) | 92.3K (1.14) | 74.4K (0.92) | none |
| GIN | 0.95 (0.27–0.98), 0% at N | 81.0K | 57.7K (0.71) | 66.6K (0.82) | 76.4K (0.94) | 92.8K (1.15) | 76.3K (0.94) | none |

| model | arm | − Frozen (pp) | − Full (pp) | TOST 2 pp vs Full | repairs | peak MB |
|---|---|---|---|---|---|---|
| GAT | full | +1.03 ± 0.23 |  |  | 14.4 | 14648 |
| GAT | selective_logged | +0.52 ± 0.31 | -0.51 ± 0.12 | yes (p=0.000) | 14.7 | 8545 |
| GAT | selective_topup | +0.73 ± 0.26 | -0.30 ± 0.07 | yes (p=0.000) | 14.4 | 12780 |
| GAT | full_isoroots | +0.99 ± 0.21 | -0.04 ± 0.06 | yes (p=0.000) | 14.6 | 11659 |
| GAT | selective_periodic | +0.44 ± 0.23 | -0.59 ± 0.13 | yes (p=0.000) | 14.4 | 8662 |
| GAT | full_exact | - | - | - | nan | nan |
| GAT | selective_exact | - | - | - | nan | nan |
| SAGE | full | +0.53 ± 0.17 |  |  | 13.8 | 730 |
| SAGE | selective_logged | +0.18 ± 0.15 | -0.35 ± 0.07 | yes (p=0.000) | 13.7 | 472 |
| SAGE | selective_topup | +0.29 ± 0.17 | -0.25 ± 0.09 | yes (p=0.000) | 14.0 | 621 |
| SAGE | full_isoroots | +0.54 ± 0.17 | +0.01 ± 0.03 | yes (p=0.000) | 13.8 | 640 |
| SAGE | selective_periodic | +0.17 ± 0.19 | -0.36 ± 0.10 | yes (p=0.000) | 13.8 | 468 |
| SAGE | full_exact | +0.47 ± 0.18 | -0.06 ± 0.03 | yes (p=0.000) | 13.9 | 1537 |
| SAGE | selective_exact | +0.14 ± 0.16 | -0.40 ± 0.08 | yes (p=0.000) | 13.8 | 1390 |
| GCN | full | +0.16 ± 0.13 |  |  | 15.2 | 1115 |
| GCN | selective_logged | -0.06 ± 0.15 | -0.22 ± 0.09 | yes (p=0.000) | 15.0 | 703 |
| GCN | selective_topup | +0.04 ± 0.14 | -0.11 ± 0.07 | yes (p=0.000) | 15.0 | 964 |
| GCN | full_isoroots | +0.07 ± 0.13 | -0.09 ± 0.04 | yes (p=0.000) | 15.1 | 943 |
| GCN | selective_periodic | -0.06 ± 0.10 | -0.22 ± 0.07 | yes (p=0.000) | 15.2 | 711 |
| GCN | full_exact | +0.41 ± 0.08 | +0.26 ± 0.09 | yes (p=0.000) | 14.9 | 2617 |
| GCN | selective_exact | +0.11 ± 0.08 | -0.04 ± 0.13 | yes (p=0.000) | 14.7 | 2336 |
| GIN | full | +0.63 ± 0.53 |  |  | 14.2 | 807 |
| GIN | selective_logged | -0.15 ± 0.41 | -0.78 ± 0.24 | yes (p=0.000) | 14.5 | 548 |
| GIN | selective_topup | -0.15 ± 0.47 | -0.78 ± 0.16 | yes (p=0.000) | 14.7 | 649 |
| GIN | full_isoroots | +0.65 ± 0.49 | +0.02 ± 0.10 | yes (p=0.000) | 14.3 | 741 |
| GIN | selective_periodic | -0.06 ± 0.50 | -0.69 ± 0.29 | yes (p=0.000) | 14.2 | 563 |
| GIN | full_exact | -2.29 ± 0.90 | -2.92 ± 0.54 | no (p=0.998) | 14.5 | 1607 |
| GIN | selective_exact | -2.72 ± 0.84 | -3.35 ± 0.68 | no (p=0.999) | 14.5 | 1440 |

Trigger: `selective_periodic` (timer at Full's realised rate) − `selective_logged` (detector), pp, paired: GAT -0.08 ± 0.21; SAGE -0.01 ± 0.10; GCN +0.00 ± 0.11; GIN +0.09 ± 0.26

Consistency with D1 `selective`: GAT: nodes within 0.00%, accuracy within 0.00 pp; SAGE: nodes within 0.00%, accuracy within 0.00 pp; GCN: nodes within 0.00%, accuracy within 0.00 pp; GIN: nodes within 0.00%, accuracy within 0.00 pp

### arxiv

| model | GIST \|S\| / N (median, min–max per repair) | full | GIST (D1 rule) | GIST, \|S\|=N | Full, GIST's \|S\| | Full, whole-graph parent | GIST, whole-graph parent | OOM |
|---|---|---|---|---|---|---|---|---|
| GAT | 0.62 (0.59–0.99), 0% at N | 91.8K | 58.5K (0.64) | 78.9K (0.86) | 79.1K (0.86) | 112.5K (1.23) | 88.6K (0.97) | none |
| SAGE | 0.62 (0.59–0.99), 0% at N | 91.8K | 59.6K (0.65) | 75.5K (0.82) | 83.6K (0.91) | 112.5K (1.23) | 89.9K (0.98) | none |
| GCN | 0.61 (0.59–0.99), 0% at N | 91.8K | 58.5K (0.64) | 78.9K (0.86) | 79.3K (0.86) | 112.5K (1.23) | 88.2K (0.96) | none |
| GIN | 0.62 (0.59–0.99), 0% at N | 91.8K | 58.5K (0.64) | 75.2K (0.82) | 82.4K (0.90) | 112.5K (1.23) | 89.3K (0.97) | none |

| model | arm | − Frozen (pp) | − Full (pp) | TOST 2 pp vs Full | repairs | peak MB |
|---|---|---|---|---|---|---|
| GAT | full | -0.52 ± 0.63 |  |  | 14.7 | 12998 |
| GAT | selective_logged | -0.91 ± 0.49 | -0.40 ± 0.47 | yes (p=0.000) | 14.8 | 7348 |
| GAT | selective_topup | -0.81 ± 0.61 | -0.29 ± 0.46 | yes (p=0.000) | 14.8 | 10913 |
| GAT | full_isoroots | -0.49 ± 0.65 | +0.03 ± 0.04 | yes (p=0.000) | 14.9 | 10226 |
| GAT | selective_periodic | -1.13 ± 0.53 | -0.62 ± 0.59 | yes (p=0.000) | 14.7 | 7466 |
| GAT | full_exact | -0.51 ± 0.61 | +0.00 ± 0.05 | yes (p=0.000) | 14.7 | 37400 |
| GAT | selective_exact | -0.99 ± 0.69 | -0.48 ± 0.49 | yes (p=0.000) | 15.0 | 32809 |
| SAGE | full | -0.46 ± 0.50 |  |  | 15.0 | 700 |
| SAGE | selective_logged | -0.89 ± 0.52 | -0.43 ± 0.38 | yes (p=0.000) | 14.8 | 433 |
| SAGE | selective_topup | -0.82 ± 0.52 | -0.36 ± 0.34 | yes (p=0.000) | 15.2 | 562 |
| SAGE | full_isoroots | -0.47 ± 0.51 | -0.01 ± 0.04 | yes (p=0.000) | 15.1 | 619 |
| SAGE | selective_periodic | -1.03 ± 0.89 | -0.58 ± 0.56 | yes (p=0.000) | 15.0 | 431 |
| SAGE | full_exact | -0.47 ± 0.49 | -0.02 ± 0.03 | yes (p=0.000) | 15.0 | 1526 |
| SAGE | selective_exact | -0.98 ± 0.60 | -0.52 ± 0.42 | yes (p=0.000) | 15.0 | 1324 |
| GCN | full | -0.53 ± 0.71 |  |  | 15.0 | 1026 |
| GCN | selective_logged | -0.40 ± 0.40 | +0.12 ± 0.35 | yes (p=0.000) | 15.0 | 612 |
| GCN | selective_topup | -0.34 ± 0.45 | +0.19 ± 0.28 | yes (p=0.000) | 15.0 | 870 |
| GCN | full_isoroots | -0.60 ± 0.71 | -0.07 ± 0.03 | yes (p=0.000) | 15.1 | 829 |
| GCN | selective_periodic | -0.57 ± 0.61 | -0.04 ± 0.37 | yes (p=0.000) | 15.0 | 622 |
| GCN | full_exact | -0.03 ± 0.43 | +0.50 ± 0.32 | yes (p=0.000) | 15.0 | 2590 |
| GCN | selective_exact | -0.19 ± 0.37 | +0.34 ± 0.50 | yes (p=0.000) | 15.2 | 2251 |
| GIN | full | -0.96 ± 0.59 |  |  | 14.7 | 792 |
| GIN | selective_logged | -1.65 ± 0.57 | -0.70 ± 0.41 | yes (p=0.000) | 14.8 | 499 |
| GIN | selective_topup | -1.40 ± 0.52 | -0.44 ± 0.32 | yes (p=0.000) | 15.2 | 637 |
| GIN | full_isoroots | -0.94 ± 0.60 | +0.02 ± 0.05 | yes (p=0.000) | 14.7 | 690 |
| GIN | selective_periodic | -2.07 ± 0.64 | -1.11 ± 0.60 | yes (p=0.004) | 14.7 | 496 |
| GIN | full_exact | -3.27 ± 0.54 | -2.31 ± 0.68 | no (p=0.836) | 14.6 | 1636 |
| GIN | selective_exact | -4.07 ± 0.81 | -3.11 ± 0.78 | no (p=0.995) | 14.2 | 1408 |

Trigger: `selective_periodic` (timer at Full's realised rate) − `selective_logged` (detector), pp, paired: GAT -0.22 ± 0.25; SAGE -0.14 ± 0.49; GCN -0.17 ± 0.33; GIN -0.42 ± 0.31

Consistency with D1 `selective`: GAT: nodes within 0.00%, accuracy within 0.00 pp; SAGE: nodes within 0.00%, accuracy within 0.00 pp; GCN: nodes within 0.00%, accuracy within 0.00 pp; GIN: nodes within 0.00%, accuracy within 0.00 pp

## R2. Oracle trigger (Arxiv, injected; fires on the known event steps, same cooldown)

| model | trigger | repairs | Full − Frozen | GIST − Frozen | GIST − Full |
|---|---|---|---|---|---|
| GAT | detector | 14.7 / 14.8 | -0.52 ± 0.63 | -0.91 ± 0.49 | -0.40 ± 0.47 |
| GAT | oracle | 7.4 / 7.4 | -0.53 ± 0.59 | -1.02 ± 1.11 | -0.49 ± 0.66 |
| SAGE | detector | 15.0 / 14.8 | -0.46 ± 0.50 | -0.89 ± 0.52 | -0.43 ± 0.38 |
| SAGE | oracle | 7.4 / 7.4 | -0.26 ± 0.47 | -0.32 ± 0.38 | -0.06 ± 0.35 |
| GCN | detector | 15.0 / 15.0 | -0.53 ± 0.71 | -0.40 ± 0.40 | +0.12 ± 0.35 |
| GCN | oracle | 7.4 / 7.4 | -0.39 ± 0.55 | -0.37 ± 0.45 | +0.02 ± 0.21 |
| GIN | detector | 14.7 / 14.8 | -0.96 ± 0.59 | -1.65 ± 0.57 | -0.70 ± 0.41 |
| GIN | oracle | 7.4 / 7.4 | -1.38 ± 0.56 | -1.93 ± 0.58 | -0.55 ± 0.34 |

## R3. Elliptic: macro-F1 and balanced accuracy (holdout, whole stream, seeds 0–9)

| model | arm | accuracy | macro-F1 | balanced acc. | minority recall | macro-F1 − Frozen (pp) | macro-F1 GIST − Full (pp) |
|---|---|---|---|---|---|---|---|
| GAT | baseline | 92.91 | 77.56 | 73.21 | 48.1 |  | - |
| GAT | full | 93.27 | 81.63 | 80.10 | 63.3 | +4.08 ± 1.91 | - |
| GAT | selective_logged | 93.00 | 81.01 | 79.77 | 62.9 | +3.45 ± 1.80 | -0.62 ± 0.62 |
| SAGE | baseline | 93.68 | 82.97 | 82.04 | 67.3 |  | - |
| SAGE | full | 93.44 | 83.12 | 83.76 | 71.5 | +0.15 ± 0.29 | - |
| SAGE | selective_logged | 93.25 | 82.81 | 83.80 | 71.9 | -0.16 ± 0.49 | -0.31 ± 0.39 |
| GCN | baseline | 87.63 | 71.51 | 74.58 | 58.2 |  | - |
| GCN | full | 89.01 | 74.27 | 77.35 | 62.7 | +2.77 ± 0.62 | - |
| GCN | selective_logged | 88.56 | 73.86 | 77.62 | 64.0 | +2.36 ± 0.78 | -0.41 ± 0.54 |
| GIN | baseline | 92.28 | 72.94 | 67.69 | 36.3 |  | - |
| GIN | full | 90.81 | 75.03 | 74.11 | 52.8 | +2.09 ± 1.51 | - |
| GIN | selective_logged | 90.76 | 74.54 | 73.34 | 51.1 | +1.59 ± 1.29 | -0.50 ± 0.47 |

## R4. Link-prediction exactness on evaluation seed 0 (float64, 3 updates per model × graph)

180 audited updates in 60 model × graph cells: 2-hop subgraph max relative difference 9.8e-16; 1-hop control min 8.3e-05, median 2.1e-01.

## R5. Static 2 × 2: root rule × parent graph (share of the graph materialised; seeds 0–9; ratio to Full = uniform roots, fan-out parent)

| graph | N | uniform, fan-out (Full) | uniform, whole graph | region, fan-out | region, whole graph | GIST rule, fan-out | GIST rule, whole graph |
|---|---|---|---|---|---|---|---|
| arxiv | 2048 | 41.7% (1.00) | 65.9% (1.58) | 15.5% (0.37) | 41.5% (0.99) | 24.8% (0.59) | 54.3% (1.30) |
| arxiv | 8192 | 70.7% (1.00) | 83.8% (1.19) | 37.0% (0.52) | 59.8% (0.85) | 53.0% (0.75) | 72.3% (1.02) |
| elliptic | 2048 | 8.8% (1.00) | 16.0% (1.81) | 1.8% (0.20) | 2.1% (0.24) | 3.7% (0.42) | 6.9% (0.78) |
| elliptic | 8192 | 26.7% (1.00) | 37.4% (1.40) | 5.6% (0.21) | 6.9% (0.26) | 12.8% (0.48) | 20.2% (0.76) |
| patents | 2048 | 4.0% (1.00) | 6.1% (1.53) | 1.3% (0.33) | 1.7% (0.44) | 2.1% (0.52) | 2.9% (0.72) |
| patents | 8192 | 14.1% (1.00) | 19.6% (1.39) | 2.8% (0.20) | 3.6% (0.26) | 6.4% (0.45) | 9.1% (0.64) |
| products | 2048 | 13.3% (1.00) | 65.1% (4.91) | 2.0% (0.15) | 14.1% (1.07) | 5.7% (0.43) | 43.8% (3.31) |
| products | 8192 | 35.0% (1.00) | 82.2% (2.35) | 4.7% (0.14) | 25.8% (0.74) | 15.7% (0.45) | 67.7% (1.93) |
| web-google | 2048 | 9.1% (1.00) | 44.7% (4.93) | 0.9% (0.10) | 5.6% (0.62) | 3.3% (0.37) | 32.8% (3.62) |
| web-google | 8192 | 26.4% (1.00) | 60.0% (2.27) | 2.9% (0.11) | 24.9% (0.94) | 10.4% (0.39) | 45.9% (1.74) |
| youtube | 2048 | 4.3% (1.00) | 36.8% (8.58) | 1.5% (0.35) | 19.4% (4.51) | 2.7% (0.63) | 30.1% (7.01) |
| youtube | 8192 | 11.6% (1.00) | 51.9% (4.48) | 5.7% (0.49) | 36.8% (3.18) | 7.2% (0.62) | 43.1% (3.72) |

## Sensitivity: cooldown κ and detector threshold τ (Arxiv-year, seeds 0–9)

| model | setting | repairs (Full / GIST) | Full − Frozen | GIST − Frozen | GIST − Full |
|---|---|---|---|---|---|
| GAT | κ=50, τ at 5% (default) | 14.4 / 14.7 | +1.03 ± 0.23 | +0.52 ± 0.31 | -0.51 ± 0.12 |
| GAT | κ=25 | 21.5 / 21.4 | +1.29 ± 0.27 | +0.72 ± 0.33 | -0.57 ± 0.17 |
| GAT | κ=100 | 8.7 / 8.8 | +0.64 ± 0.16 | +0.19 ± 0.22 | -0.44 ± 0.17 |
| GAT | τ at 1% | 6.6 / 6.6 | +0.66 ± 0.31 | +0.29 ± 0.25 | -0.37 ± 0.18 |
| GAT | τ at 10% | 16.5 / 16.5 | +1.05 ± 0.21 | +0.48 ± 0.23 | -0.57 ± 0.15 |
| SAGE | κ=50, τ at 5% (default) | 13.8 / 13.7 | +0.53 ± 0.17 | +0.18 ± 0.15 | -0.35 ± 0.07 |
| SAGE | κ=25 | 20.7 / 20.6 | +0.66 ± 0.14 | +0.25 ± 0.17 | -0.41 ± 0.07 |
| SAGE | κ=100 | 8.4 / 8.2 | +0.45 ± 0.16 | +0.04 ± 0.13 | -0.41 ± 0.16 |
| SAGE | τ at 1% | 6.6 / 6.5 | +0.40 ± 0.15 | +0.07 ± 0.20 | -0.33 ± 0.16 |
| SAGE | τ at 10% | 16.9 / 16.7 | +0.61 ± 0.14 | +0.19 ± 0.18 | -0.43 ± 0.12 |
| GCN | κ=50, τ at 5% (default) | 15.2 / 15.0 | +0.16 ± 0.13 | -0.06 ± 0.15 | -0.22 ± 0.09 |
| GCN | κ=25 | 22.9 / 22.5 | +0.21 ± 0.14 | -0.07 ± 0.14 | -0.28 ± 0.06 |
| GCN | κ=100 | 8.6 / 8.7 | +0.10 ± 0.14 | -0.14 ± 0.17 | -0.24 ± 0.10 |
| GCN | τ at 1% | 7.7 / 7.6 | +0.07 ± 0.12 | -0.12 ± 0.11 | -0.19 ± 0.13 |
| GCN | τ at 10% | 16.9 / 17.0 | +0.16 ± 0.13 | -0.13 ± 0.15 | -0.28 ± 0.08 |
| GIN | κ=50, τ at 5% (default) | 14.2 / 14.5 | +0.63 ± 0.53 | -0.15 ± 0.41 | -0.78 ± 0.24 |
| GIN | κ=25 | 22.3 / 22.2 | +1.14 ± 0.49 | +0.37 ± 0.61 | -0.77 ± 0.25 |
| GIN | κ=100 | 8.7 / 8.8 | +0.09 ± 0.53 | -0.68 ± 0.55 | -0.76 ± 0.31 |
| GIN | τ at 1% | 7.1 / 7.3 | +0.17 ± 0.43 | -0.65 ± 0.54 | -0.82 ± 0.43 |
| GIN | τ at 10% | 16.7 / 16.7 | +0.81 ± 0.51 | +0.01 ± 0.45 | -0.80 ± 0.21 |

## Trigger behaviour in D1 (no new run): how often the detector fires, and how many repairs the cooldown allows

| stream | flagged steps (%) | 50-step windows with a flag (%) | Full repairs / 1000 steps | cooldown ceiling |
|---|---|---|---|---|
| arxiv_year | 4.6 | 93 | 14.4 | 19.6 |
| arxiv | 5.0 | 92 | 14.8 | 19.6 |
| elliptic | 5.2 | 96 | 15.1 | 19.6 |
| cora | 5.3 | 96 | 15.1 | 19.6 |
| pubmed | 5.3 | 97 | 15.3 | 19.6 |
| products | 4.9 | 95 | 14.8 | 19.6 |

