# Link prediction on 15 SNAP graphs (seeds 0-9)

MRR x100 differences, paired by seed, mean ± 95% CI over models x seeds. Cost = selective / full per update.

| Family | Graph | models | baseline MRR | full − base (G) | sel − full (C6) | full − periodic (C2) | sel − random (C3) | drifted: full − base | clean: sel − base | nodes | FLOPs | peak mem | train ms |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| social | facebook | gat,gcn,gin,sage | 0.1017 | +0.07 ± 0.04 | -0.02 ± 0.02 | +0.02 ± 0.02 | +0.01 ± 0.01 | +0.15 ± 0.05 | -0.03 ± 0.05 | 0.803 | 0.800 | 0.745 | 0.982 |
| social | twitter | gat,gcn,gin,sage | 0.4619 | +0.33 ± 0.11 | -0.40 ± 0.11 | -0.01 ± 0.01 | -0.21 ± 0.06 | +1.41 ± 0.11 | -0.28 ± 0.06 | 0.788 | 0.781 | 0.681 | 0.793 |
| social | youtube | gat,gcn,gin,sage | 0.1564 | +2.13 ± 0.46 | -0.72 ± 0.18 | -0.23 ± 0.11 | -0.41 ± 0.18 | +2.29 ± 0.37 | +1.34 ± 0.60 | 0.732 | 0.734 | 0.723 | 0.769 |
| web | web-stanford | gat,gcn,gin,sage | 0.5309 | +0.38 ± 0.31 | -0.56 ± 0.16 | -0.02 ± 0.07 | -0.22 ± 0.10 | +1.80 ± 0.57 | -0.30 ± 0.18 | 0.616 | 0.618 | 0.629 | 0.804 |
| web | web-google | gat,gcn,gin,sage | 0.7750 | +1.12 ± 0.53 | -0.75 ± 0.17 | -0.10 ± 0.06 | -0.40 ± 0.10 | +1.28 ± 0.48 | +0.34 ± 0.38 | 0.527 | 0.531 | 0.552 | 0.604 |
| web | web-berkstan | gat,gcn,gin,sage | 0.5959 | +0.31 ± 0.24 | -0.33 ± 0.14 | -0.03 ± 0.04 | -0.13 ± 0.07 | +1.56 ± 0.49 | -0.14 ± 0.13 | 0.574 | 0.574 | 0.572 | 0.652 |
| p2p | gnutella25 | gat,gcn,gin,sage | 0.0140 | -0.04 ± 0.05 | +0.01 ± 0.01 | -0.00 ± 0.02 | +0.00 ± 0.01 | -0.01 ± 0.06 | -0.09 ± 0.06 | 0.932 | 0.933 | 0.913 | 0.984 |
| p2p | gnutella30 | gat,gcn,gin,sage | 0.0175 | -0.05 ± 0.07 | -0.00 ± 0.01 | -0.04 ± 0.03 | +0.01 ± 0.01 | +0.04 ± 0.09 | -0.12 ± 0.08 | 0.976 | 0.976 | 0.969 | 1.017 |
| p2p | gnutella31 | gat,gcn,gin,sage | 0.0136 | -0.04 ± 0.05 | +0.00 ± 0.01 | -0.02 ± 0.03 | +0.00 ± 0.01 | +0.01 ± 0.05 | -0.06 ± 0.05 | 0.989 | 0.988 | 0.985 | 1.001 |
| road | road-pa | gat,gcn,gin,sage | 0.1487 | -0.85 ± 0.45 | -0.10 ± 0.10 | +0.43 ± 0.19 | -0.05 ± 0.07 | -0.49 ± 0.29 | -1.00 ± 0.48 | 0.946 | 0.946 | 0.976 | 1.031 |
| road | road-tx | gat,gcn,gin,sage | 0.1468 | -0.84 ± 0.56 | -0.14 ± 0.08 | +0.40 ± 0.14 | -0.06 ± 0.10 | -0.46 ± 0.35 | -1.03 ± 0.58 | 0.961 | 0.962 | 0.985 | 0.994 |
| road | road-ca | gat,gcn,gin,sage | 0.1459 | -0.85 ± 0.62 | -0.27 ± 0.09 | +0.29 ± 0.13 | -0.09 ± 0.08 | -0.42 ± 0.39 | -1.17 ± 0.64 | 0.932 | 0.929 | 0.974 | 0.994 |
| citation | hepth | gat,gcn,gin,sage | 0.2232 | +1.21 ± 0.43 | -0.27 ± 0.19 | +0.08 ± 0.04 | -0.05 ± 0.06 | +1.21 ± 0.43 | - | 0.859 | 0.856 | 0.819 | 1.018 |
| citation | hepph | gat,gcn,gin,sage | 0.2685 | +0.74 ± 0.41 | -0.27 ± 0.18 | +0.22 ± 0.13 | -0.06 ± 0.08 | +0.74 ± 0.41 | - | 0.838 | 0.834 | 0.759 | 1.002 |
| citation | patents | gat,gcn,gin,sage | 0.3177 | +1.83 ± 0.80 | -0.17 ± 0.08 | -0.00 ± 0.01 | +0.00 ± 0.08 | +1.83 ± 0.80 | - | 0.656 | 0.654 | 0.703 | 0.671 |

(temporal graphs: 'drifted' = edges with a recently arrived endpoint, 'clean' = both endpoints old)

## Detection (injected drift; baseline arm's scan)

| Family | Graph | events | hit rate | median delay | false flags / 1000 steps | top-1 region precision | chance |
|---|---|---|---|---|---|---|---|
| social | facebook | 320 | 0.89 | 20 | 1.2 | 0.62 | 0.125 |
| social | twitter | 320 | 1.00 | 18 | 2.6 | 0.65 | 0.025 |
| social | youtube | 320 | 0.94 | 26 | 1.9 | 0.51 | 0.008 |
| web | web-stanford | 320 | 0.91 | 21 | 2.0 | 0.63 | 0.008 |
| web | web-google | 320 | 0.99 | 20 | 1.9 | 0.66 | 0.008 |
| web | web-berkstan | 320 | 0.94 | 16 | 1.8 | 0.63 | 0.008 |
| p2p | gnutella25 | 280 | 0.29 | 41 | 0.3 | 0.56 | 0.087 |
| p2p | gnutella30 | 320 | 0.49 | 38 | 0.9 | 0.37 | 0.054 |
| p2p | gnutella31 | 320 | 0.23 | 42 | 0.5 | 0.24 | 0.032 |
| road | road-pa | 320 | 0.40 | 40 | 0.3 | 0.59 | 0.008 |
| road | road-tx | 320 | 0.55 | 29 | 0.6 | 0.47 | 0.008 |
| road | road-ca | 320 | 0.68 | 35 | 0.7 | 0.49 | 0.008 |

