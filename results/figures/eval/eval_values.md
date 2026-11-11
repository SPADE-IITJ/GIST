## eval_exact: digits of the parent-graph gradient reproduced = -log10(largest relative per-tensor difference); GIST = worst (max difference) over audits, controls = best (min difference)

- GAT (n=10): GIST 5.8e-16 = 15.2 digits; L-1 hops 7.7e-01 = 0.1
- GraphSAGE (n=10): GIST 3.0e-16 = 15.5 digits; L-1 hops 3.8e-01 = 0.4
- GIN (n=10): GIST 4.4e-16 = 15.4 digits; L-1 hops 4.2e-01 = 0.4
- GCN (n=10): GIST 4.9e-16 = 15.3 digits; L-1 hops 2.9e-01 = 0.5; no fix 4.7e-02 = 1.3
- BatchNorm (n=10): GIST 3.2e-16 = 15.5 digits; L-1 hops 2.7e-01 = 0.6; no fix 7.0e-01 = 0.2
- EvolveGCN-O (n=10): GIST 3.3e-16 = 15.5 digits; L-1 hops 4.5e-01 = 0.3; no fix 3.2e-02 = 1.5
- T-GCN (n=10): GIST 4.2e-16 = 15.4 digits; L-1 hops 2.7e-01 = 0.6; no fix 2.7e-02 = 1.6
- GConvGRU (n=20): GIST 7.9e-16 = 15.1 digits; L-1 hops 2.2e-06 = 5.7; no fix 4.5e-04 = 3.4
- In-stream repairs (n=69): GIST 5.2e-14 = 13.3 digits; L-1 hops 4.8e-02 = 1.3
- Link prediction (n=180): GIST 9.8e-16 = 15.0 digits; L-1 hops 8.3e-05 = 4.1

## eval_cost (a) share of the graph materialised, N=8192 (fan-out parent / whole-graph parent), E1mix seeds 0-9

- arxiv uniform: 70.7% / 83.8%
- arxiv region: 37.0% / 59.8%
- arxiv gist: 53.0% / 72.3%
- products uniform: 35.0% / 82.2%
- products region: 4.7% / 25.8%
- products gist: 15.7% / 67.7%
- elliptic uniform: 26.7% / 37.4%
- elliptic region: 5.6% / 6.9%
- elliptic gist: 12.8% / 20.2%
- patents uniform: 14.1% / 19.6%
- patents region: 2.8% / 3.6%
- patents gist: 6.4% / 9.1%
- web-google uniform: 26.4% / 60.0%
- web-google region: 2.9% / 24.9%
- web-google gist: 10.4% / 45.9%
- youtube uniform: 11.6% / 51.9%
- youtube region: 5.7% / 36.8%
- youtube gist: 7.2% / 43.1%

## eval_cost (b) nodes per repair relative to Full (mean over backbones; min-max), D8 R1

- arxiv_year full: mean 1.000 (min 1.000, max 1.000, n=4); gat 1.000, sage 1.000, gcn 1.000, gin 1.000
- arxiv_year full_isoroots: mean 0.915 (min 0.888, max 0.943, n=4); gat 0.888, sage 0.921, gcn 0.908, gin 0.943
- arxiv_year selective_topup: mean 0.872 (min 0.822, max 0.902, n=4); gat 0.902, sage 0.880, gcn 0.886, gin 0.822
- arxiv_year selective_logged: mean 0.702 (min 0.692, max 0.712, n=4); gat 0.692, sage 0.705, gcn 0.699, gin 0.712
- arxiv_year full_exact: mean 1.146 (min 1.140, max 1.152, n=3); gat nan, sage 1.152, gcn 1.140, gin 1.145
- arxiv_year selective_exact: mean 0.936 (min 0.920, max 0.946, n=3); gat nan, sage 0.946, gcn 0.920, gin 0.942
- arxiv_year GIST |S|/N median per backbone: gat 0.59, sage 0.61, gcn 0.60, gin 0.95
- arxiv full: mean 1.000 (min 1.000, max 1.000, n=4); gat 1.000, sage 1.000, gcn 1.000, gin 1.000
- arxiv full_isoroots: mean 0.884 (min 0.862, max 0.911, n=4); gat 0.862, sage 0.911, gcn 0.864, gin 0.898
- arxiv selective_topup: mean 0.840 (min 0.819, max 0.860, n=4); gat 0.860, sage 0.823, gcn 0.859, gin 0.819
- arxiv selective_logged: mean 0.640 (min 0.637, max 0.650, n=4); gat 0.637, sage 0.650, gcn 0.637, gin 0.637
- arxiv full_exact: mean 1.226 (min 1.226, max 1.226, n=4); gat 1.226, sage 1.226, gcn 1.226, gin 1.226
- arxiv selective_exact: mean 0.970 (min 0.961, max 0.980, n=4); gat 0.965, sage 0.980, gcn 0.961, gin 0.973
- arxiv GIST |S|/N median per backbone: gat 0.62, sage 0.62, gcn 0.61, gin 0.62

## eval_scaling (a) share of the graph in the exact 2-hop expansion of N roots, uniform / one region (E1, seeds 0-9)

- arxiv uniform: N=128: 35.2%, N=512: 48.9%, N=2048: 66.4%, N=8192: 83.8%, N=32768: 96.0%
- arxiv region: N=128: 14.4%, N=512: 27.6%, N=2048: 42.8%, N=8192: 61.4%, N=32768: 80.5%
- products uniform: N=128: 16.0%, N=512: 39.4%, N=2048: 64.7%, N=8192: 82.2%, N=32768: 91.5%
- products region: N=128: 4.0%, N=512: 7.8%, N=2048: 9.7%, N=8192: 25.1%, N=32768: 56.7%
- patents uniform: N=128: 0.4%, N=512: 1.6%, N=2048: 6.0%, N=8192: 19.6%, N=32768: 47.5%
- patents region: N=128: 0.3%, N=512: 0.7%, N=2048: 1.5%, N=8192: 3.2%, N=32768: 11.7%
- elliptic uniform: N=128: 1.5%, N=512: 5.1%, N=2048: 16.0%, N=8192: 37.4%, N=32768: 71.7%
- elliptic region: N=128: 0.5%, N=512: 0.7%, N=2048: 2.2%, N=8192: 7.3%, N=32768: 25.2%

## eval_scaling (b) GIST nodes / Full nodes against the repaired region (E3a, GAT, N=2048)

- arxiv: Full 70659, whole 169343 (2.4x; OOM ['gat']); 0.5%: 0.78, 1%: 0.60, 2%: 0.67, 5%: 0.74, 10%: 0.80, 25%: 0.89
- patents: Full 150204, whole 3774768 (25.1x; OOM ['gat', 'sage', 'gcn', 'gin']); 0.5%: 0.64, 1%: 0.68, 2%: 0.77, 5%: 0.87, 10%: 0.92, 25%: 0.97
- web-google: Full 79122, whole 875713 (11.1x; OOM ['gat']); 0.5%: 0.42, 1%: 0.46, 2%: 0.54, 5%: 0.65, 10%: 0.76, 25%: 0.89
- youtube: Full 48612, whole 1134890 (23.3x; OOM ['gat']); 0.5%: 0.69, 1%: 0.72, 2%: 0.79, 5%: 0.86, 10%: 0.93, 25%: 0.99

## eval_speed (a) repair speedup Full/GIST (mean of per-seed ratios; min-max), D3 seeds 0-4

- gat: N=2048: 1.31 (1.16-1.59), net 1.18, N=4096: 1.46 (1.29-1.74), net 1.36, N=8192: 1.62 (1.40-1.83), net 1.53
- sage: N=2048: 1.44 (1.10-1.75), net 1.21, N=4096: 1.55 (1.14-1.84), net 1.36, N=8192: 1.65 (1.41-1.88), net 1.48
- gcn: N=2048: 1.26 (1.11-1.50), net 1.06, N=4096: 1.49 (1.24-1.75), net 1.32, N=8192: 1.59 (1.47-1.73), net 1.46
- gin: N=2048: 1.66 (1.51-1.83), net 1.34, N=4096: 1.74 (1.68-1.90), net 1.51, N=8192: 1.77 (1.67-1.87), net 1.61

## eval_speed (b) mean repair at N=8192 (ms): host sampling + rest

- gin gist: sampling 87.5 + rest 19.2 = 106.7 ms
- gin full: sampling 167.4 + rest 18.2 = 185.6 ms
- gcn gist: sampling 98.4 + rest 24.2 = 122.6 ms
- gcn full: sampling 169.8 + rest 18.3 = 188.1 ms
- sage gist: sampling 95.0 + rest 20.4 = 115.4 ms
- sage full: sampling 171.4 + rest 17.3 = 188.6 ms
- gat gist: sampling 98.1 + rest 84.9 = 183.0 ms
- gat full: sampling 170.8 + rest 124.8 = 295.6 ms

## eval_speed (c) peak repair memory at N=8192 (median GB over seeds; OOM = every seed), D7

- arxiv gat full h=256: 12.98 (OOM 0/5)
- arxiv gat full h=512: 25.80 (OOM 0/5)
- arxiv gat full h=1024: OOM (OOM 5/5)
- arxiv gat selective h=256: 9.34 (OOM 0/5)
- arxiv gat selective h=512: 18.72 (OOM 0/5)
- arxiv gat selective h=1024: 36.42 (OOM 0/5)
- arxiv gat global h=256: 43.15 (OOM 0/5)
- arxiv gat global h=512: OOM (OOM 5/5)
- arxiv gat global h=1024: OOM (OOM 5/5)
- products sage full h=256: 4.42 (OOM 0/5)
- products sage full h=512: 8.10 (OOM 0/5)
- products sage full h=1024: 15.51 (OOM 0/5)
- products sage selective h=256: 2.13 (OOM 0/5)
- products sage selective h=512: 3.91 (OOM 0/5)
- products sage selective h=1024: 7.32 (OOM 0/5)
- products sage global h=256: OOM (OOM 5/5)
- products sage global h=512: OOM (OOM 5/5)
- products sage global h=1024: OOM (OOM 5/5)

## eval_speed (d) serving p99 while a repair loops on the serving GPU (ms; ratio to idle), E7 seeds 0-4

- gat full: p99 19.42 ms (2.45x idle 7.94 ms)
- gat selective: p99 19.36 ms (2.44x idle 7.94 ms)
- gat global: OOM on 5/5
- sage full: p99 2.81 ms (5.27x idle 0.53 ms)
- sage selective: p99 2.84 ms (5.33x idle 0.53 ms)
- sage global: p99 2.83 ms (5.31x idle 0.53 ms)

## eval_speed (a) repair speedup Full/GIST (mean of per-seed ratios; min-max), D3 seeds 0-4

- gat: N=2048: 1.31 (1.16-1.59), net 1.18, N=4096: 1.46 (1.29-1.74), net 1.36, N=8192: 1.62 (1.40-1.83), net 1.53
- sage: N=2048: 1.44 (1.10-1.75), net 1.21, N=4096: 1.55 (1.14-1.84), net 1.36, N=8192: 1.65 (1.41-1.88), net 1.48
- gcn: N=2048: 1.26 (1.11-1.50), net 1.06, N=4096: 1.49 (1.24-1.75), net 1.32, N=8192: 1.59 (1.47-1.73), net 1.46
- gin: N=2048: 1.66 (1.51-1.83), net 1.34, N=4096: 1.74 (1.68-1.90), net 1.51, N=8192: 1.77 (1.67-1.87), net 1.61

## eval_speed (b) mean repair at N=8192 (ms): host sampling + rest

- gin gist: sampling 87.5 + rest 19.2 = 106.7 ms
- gin full: sampling 167.4 + rest 18.2 = 185.6 ms
- gcn gist: sampling 98.4 + rest 24.2 = 122.6 ms
- gcn full: sampling 169.8 + rest 18.3 = 188.1 ms
- sage gist: sampling 95.0 + rest 20.4 = 115.4 ms
- sage full: sampling 171.4 + rest 17.3 = 188.6 ms
- gat gist: sampling 98.1 + rest 84.9 = 183.0 ms
- gat full: sampling 170.8 + rest 124.8 = 295.6 ms

## eval_speed (c) peak repair memory at N=8192 (median GB over seeds; OOM = every seed), D7

- arxiv gat full h=256: 12.98 (OOM 0/5)
- arxiv gat full h=512: 25.80 (OOM 0/5)
- arxiv gat full h=1024: OOM (OOM 5/5)
- arxiv gat selective h=256: 9.34 (OOM 0/5)
- arxiv gat selective h=512: 18.72 (OOM 0/5)
- arxiv gat selective h=1024: 36.42 (OOM 0/5)
- arxiv gat global h=256: 43.15 (OOM 0/5)
- arxiv gat global h=512: OOM (OOM 5/5)
- arxiv gat global h=1024: OOM (OOM 5/5)
- products sage full h=256: 4.42 (OOM 0/5)
- products sage full h=512: 8.10 (OOM 0/5)
- products sage full h=1024: 15.51 (OOM 0/5)
- products sage selective h=256: 2.13 (OOM 0/5)
- products sage selective h=512: 3.91 (OOM 0/5)
- products sage selective h=1024: 7.32 (OOM 0/5)
- products sage global h=256: OOM (OOM 5/5)
- products sage global h=512: OOM (OOM 5/5)
- products sage global h=1024: OOM (OOM 5/5)

## eval_speed (d) serving p99 while a repair loops on the serving GPU (ms; ratio to idle), E7 seeds 0-4

- gat full: p99 19.42 ms (2.45x idle 7.94 ms)
- gat selective: p99 19.36 ms (2.44x idle 7.94 ms)
- gat global: OOM on 5/5
- sage full: p99 2.81 ms (5.27x idle 0.53 ms)
- sage selective: p99 2.84 ms (5.33x idle 0.53 ms)
- sage global: p99 2.83 ms (5.31x idle 0.53 ms)

## eval_ablation (a) root rule: nodes / Full and accuracy - Full (pp), mean over backbones (seeds paired)

- arxiv_year full_isoroots: nodes 0.915, acc - Full -0.02 +- 0.03 (n=40)
- arxiv_year selective_topup: nodes 0.872, acc - Full -0.36 +- 0.09 (n=40)
- arxiv_year selective_logged: nodes 0.702, acc - Full -0.47 +- 0.09 (n=40)
- arxiv_year random_region: nodes 0.706, acc - Full -0.42 +- 0.09 (n=40)
- arxiv_year selective_trunc: nodes 0.274, acc - Full -0.67 +- 0.23 (n=40)
- arxiv full_isoroots: nodes 0.884, acc - Full -0.01 +- 0.02 (n=40)
- arxiv selective_topup: nodes 0.840, acc - Full -0.23 +- 0.17 (n=40)
- arxiv selective_logged: nodes 0.640, acc - Full -0.35 +- 0.20 (n=40)
- arxiv random_region: nodes 0.645, acc - Full -0.32 +- 0.19 (n=40)
- arxiv selective_trunc: nodes 0.208, acc - Full -0.64 +- 0.32 (n=40)

## eval_ablation (b) rehearsal share rho and budget N: GIST - Full (pp) and nodes / Full, GAT + SAGE x 2 streams

- N=2048 rho0: nodes 0.387, GIST - Full -1.32 +- 0.33
- N=2048 rho0.25: nodes 0.646, GIST - Full -0.62 +- 0.22
- N=2048 rho0.5: nodes 0.818, GIST - Full -0.30 +- 0.15
- N=2048 diffuse (Arxiv): nodes 0.636, GIST - Full -0.84
- N=4096 rho0: nodes 0.334, GIST - Full -1.40 +- 0.41
- N=4096 rho0.25: nodes 0.654, GIST - Full -0.62 +- 0.20
- N=4096 rho0.5: nodes 0.826, GIST - Full -0.32 +- 0.12
- N=4096 diffuse (Arxiv): nodes 0.632, GIST - Full -0.83
- N=8192 rho0: nodes 0.282, GIST - Full -1.43 +- 0.37
- N=8192 rho0.25: nodes 0.671, GIST - Full -0.42 +- 0.13
- N=8192 rho0.5: nodes 0.845, GIST - Full -0.22 +- 0.11
- N=8192 diffuse (Arxiv): nodes 0.632, GIST - Full -0.61

## eval_ablation (c) repair rate on Arxiv-year: repairs per 1000 steps and rule - Frozen (pp), mean over backbones

- full $\kappa{=}100$: repairs 8.6, - Frozen +0.32 +- 0.15
- full $\tau_{1\%}$: repairs 7.0, - Frozen +0.33 +- 0.14
- full default: repairs 14.4, - Frozen +0.59 +- 0.17
- full $\tau_{10\%}$: repairs 16.8, - Frozen +0.66 +- 0.16
- full $\kappa{=}25$: repairs 21.8, - Frozen +0.82 +- 0.19
- selective_logged $\kappa{=}100$: repairs 8.6, - Frozen -0.15 +- 0.17
- selective_logged $\tau_{1\%}$: repairs 7.0, - Frozen -0.10 +- 0.18
- selective_logged default: repairs 14.5, - Frozen +0.12 +- 0.15
- selective_logged $\tau_{10\%}$: repairs 16.7, - Frozen +0.14 +- 0.14
- selective_logged $\kappa{=}25$: repairs 21.7, - Frozen +0.32 +- 0.18
- GIST on a timer (Full's schedule): repairs 14.4, - Frozen +0.12 +- 0.14

## eval_ablation (d) breaking exactness inside the stream: arm - exact GIST (pp), seeds paired

- gat arxiv_year L-1 hops: +0.06 +- 0.12 (n=10); nodes 0.374 of GIST
- gat arxiv L-1 hops: -0.10 +- 0.44 (n=10); nodes 0.321 of GIST
- sage arxiv_year L-1 hops: +0.04 +- 0.07 (n=10); nodes 0.381 of GIST
- sage arxiv L-1 hops: +0.07 +- 0.13 (n=10); nodes 0.327 of GIST
- gcn arxiv_year L-1 hops: -0.01 +- 0.17 (n=10); nodes 0.390 of GIST
- gcn arxiv_year no fix (whole-graph parent): -0.01 +- 0.01 (n=10)
- gcn arxiv L-1 hops: -0.07 +- 0.38 (n=10); nodes 0.319 of GIST
- gcn arxiv no fix (whole-graph parent): -0.03 +- 0.06 (n=10)
- gin arxiv_year L-1 hops: -0.91 +- 0.40 (n=10); nodes 0.416 of GIST
- gin arxiv L-1 hops: -1.08 +- 0.53 (n=10); nodes 0.331 of GIST

## eval_accuracy (a) Full / GIST - Frozen, pp, mean +- 95% CI over backbones x seeds (Elliptic: macro-F1, D8 R3)

- arxiv_year full: +0.59 +- 0.17 (n=40)
- arxiv_year selective: +0.12 +- 0.15 (n=40)
- elliptic full: +2.27 +- 0.71 (n=40)
- elliptic selective: +1.81 +- 0.67 (n=40)
- arxiv full: -0.62 +- 0.27 (n=40)
- arxiv selective: -0.97 +- 0.26 (n=40)
- cora full: -0.51 +- 0.30 (n=40)
- cora selective: -1.14 +- 0.61 (n=40)
- pubmed full: -0.20 +- 0.35 (n=40)
- pubmed selective: -0.23 +- 0.34 (n=40)
- products full: -0.35 +- 0.39 (n=30)
- products selective: -0.42 +- 0.39 (n=30)

## eval_accuracy (b) every rule - Frozen on the Arxiv streams, pp (pooled over backbones x seeds; OOM seeds excluded)

- arxiv_year full: +0.59 +- 0.17 (n=40)
- arxiv full: -0.62 +- 0.27 (n=40)
- arxiv_year selective: +0.12 +- 0.15 (n=40)
- arxiv selective: -0.97 +- 0.26 (n=40)
- arxiv_year global_trigger: -0.31 +- 0.49 (n=30)
- arxiv global_trigger: -1.02 +- 0.46 (n=40)
- arxiv_year roland: +0.71 +- 0.23 (n=29)
- arxiv roland: +0.22 +- 0.30 (n=40)
- arxiv_year ewc: +0.51 +- 0.15 (n=40)
- arxiv ewc: -0.42 +- 0.23 (n=40)
- arxiv_year twp: +0.45 +- 0.13 (n=40)
- arxiv twp: -0.40 +- 0.22 (n=40)
- arxiv_year ergnn: +0.05 +- 0.19 (n=40)
- arxiv ergnn: -1.05 +- 0.31 (n=40)

## eval_accuracy (c) drifted / clean holdout nodes: rule - Frozen, pp (pooled over backbones x seeds)

- arxiv_year full: drifted +0.67, clean +0.41
- arxiv_year selective: drifted +0.22, clean -0.08
- elliptic full: drifted +0.15, clean -0.35
- elliptic selective: drifted -0.13, clean -0.52
- arxiv full: drifted +1.86, clean -1.32
- arxiv selective: drifted +1.48, clean -1.70
- cora full: drifted +2.18, clean -2.35
- cora selective: drifted +1.33, clean -2.86
- pubmed full: drifted +1.50, clean -1.40
- pubmed selective: drifted +1.41, clean -1.34
- products full: drifted +1.98, clean -1.07
- products selective: drifted +1.86, clean -1.12

## eval_timeline (pp against Frozen, 50-step mean over backbones x seeds)

- arxiv_year roland: 29 pairs, whole-stream mean +0.71 pp; end (last 50) +2.17; first half +0.03, second half +1.39
- arxiv_year full: 40 pairs, whole-stream mean +0.59 pp; end (last 50) +1.84; first half -0.06, second half +1.23
- arxiv_year selective: 40 pairs, whole-stream mean +0.12 pp; end (last 50) +1.09; first half -0.36, second half +0.60
- arxiv roland: 40 pairs, whole-stream mean +0.22 pp; end (last 50) +0.71; first half -0.10, second half +0.54
- arxiv full: 40 pairs, whole-stream mean -0.62 pp; end (last 50) -1.08; first half -0.43, second half -0.81
- arxiv selective: 40 pairs, whole-stream mean -0.97 pp; end (last 50) -1.81; first half -0.67, second half -1.26

## eval_ablation (a) root rule: nodes / Full and accuracy - Full (pp), mean over backbones (seeds paired)

- arxiv_year full_isoroots: nodes 0.915, acc - Full -0.02 +- 0.03 (n=40)
- arxiv_year selective_topup: nodes 0.872, acc - Full -0.36 +- 0.09 (n=40)
- arxiv_year selective_logged: nodes 0.702, acc - Full -0.47 +- 0.09 (n=40)
- arxiv_year random_region: nodes 0.706, acc - Full -0.42 +- 0.09 (n=40)
- arxiv_year selective_trunc: nodes 0.274, acc - Full -0.67 +- 0.23 (n=40)
- arxiv full_isoroots: nodes 0.884, acc - Full -0.01 +- 0.02 (n=40)
- arxiv selective_topup: nodes 0.840, acc - Full -0.23 +- 0.17 (n=40)
- arxiv selective_logged: nodes 0.640, acc - Full -0.35 +- 0.20 (n=40)
- arxiv random_region: nodes 0.645, acc - Full -0.32 +- 0.19 (n=40)
- arxiv selective_trunc: nodes 0.208, acc - Full -0.64 +- 0.32 (n=40)

## eval_ablation (b) rehearsal share rho and budget N: GIST - Full (pp) and nodes / Full, GAT + SAGE x 2 streams

- N=2048 rho0: nodes 0.387, GIST - Full -1.32 +- 0.33
- N=2048 rho0.25: nodes 0.646, GIST - Full -0.62 +- 0.22
- N=2048 rho0.5: nodes 0.818, GIST - Full -0.30 +- 0.15
- N=2048 diffuse (Arxiv): nodes 0.636, GIST - Full -0.84
- N=4096 rho0: nodes 0.334, GIST - Full -1.40 +- 0.41
- N=4096 rho0.25: nodes 0.654, GIST - Full -0.62 +- 0.20
- N=4096 rho0.5: nodes 0.826, GIST - Full -0.32 +- 0.12
- N=4096 diffuse (Arxiv): nodes 0.632, GIST - Full -0.83
- N=8192 rho0: nodes 0.282, GIST - Full -1.43 +- 0.37
- N=8192 rho0.25: nodes 0.671, GIST - Full -0.42 +- 0.13
- N=8192 rho0.5: nodes 0.845, GIST - Full -0.22 +- 0.11
- N=8192 diffuse (Arxiv): nodes 0.632, GIST - Full -0.61

## eval_ablation (c) repair rate on Arxiv-year: repairs per 1000 steps and rule - Frozen (pp), mean over backbones

- full $\kappa{=}100$: repairs 8.6, - Frozen +0.32 +- 0.15
- full $\tau_{1\%}$: repairs 7.0, - Frozen +0.33 +- 0.14
- full default: repairs 14.4, - Frozen +0.59 +- 0.17
- full $\tau_{10\%}$: repairs 16.8, - Frozen +0.66 +- 0.16
- full $\kappa{=}25$: repairs 21.8, - Frozen +0.82 +- 0.19
- selective_logged $\kappa{=}100$: repairs 8.6, - Frozen -0.15 +- 0.17
- selective_logged $\tau_{1\%}$: repairs 7.0, - Frozen -0.10 +- 0.18
- selective_logged default: repairs 14.5, - Frozen +0.12 +- 0.15
- selective_logged $\tau_{10\%}$: repairs 16.7, - Frozen +0.14 +- 0.14
- selective_logged $\kappa{=}25$: repairs 21.7, - Frozen +0.32 +- 0.18
- GIST on a timer (Full's schedule): repairs 14.4, - Frozen +0.12 +- 0.14

## eval_ablation (d) breaking exactness inside the stream: arm - exact GIST (pp), seeds paired

- gat arxiv_year L-1 hops: +0.06 +- 0.12 (n=10); nodes 0.374 of GIST
- gat arxiv L-1 hops: -0.10 +- 0.44 (n=10); nodes 0.321 of GIST
- sage arxiv_year L-1 hops: +0.04 +- 0.07 (n=10); nodes 0.381 of GIST
- sage arxiv L-1 hops: +0.07 +- 0.13 (n=10); nodes 0.327 of GIST
- gcn arxiv_year L-1 hops: -0.01 +- 0.17 (n=10); nodes 0.390 of GIST
- gcn arxiv_year no fix (whole-graph parent): -0.01 +- 0.01 (n=10)
- gcn arxiv L-1 hops: -0.07 +- 0.38 (n=10); nodes 0.319 of GIST
- gcn arxiv no fix (whole-graph parent): -0.03 +- 0.06 (n=10)
- gin arxiv_year L-1 hops: -0.91 +- 0.40 (n=10); nodes 0.416 of GIST
- gin arxiv L-1 hops: -1.08 +- 0.53 (n=10); nodes 0.331 of GIST

## eval_families: reach = uniform / regional 2-hop share (E1, budget nearest the graph's); saved = 1 - GIST/Full nodes; G = Full - Frozen MRR x100 (D6)

- web web-google: reach 5.25x, saved 47.3%, G +1.12 +- 0.53, GIST - Full -0.75
- web web-berkstan: reach 7.46x, saved 42.6%, G +0.31 +- 0.24, GIST - Full -0.33
- web web-stanford: reach 5.54x, saved 38.4%, G +0.38 +- 0.31, GIST - Full -0.56
- citation patents: reach 4.01x, saved 34.4%, G +1.83 +- 0.80, GIST - Full -0.17
- citation hepph: reach 1.70x, saved 16.2%, G +0.74 +- 0.41, GIST - Full -0.27
- citation hepth: reach 1.41x, saved 14.1%, G +1.21 +- 0.43, GIST - Full -0.27
- social youtube: reach 1.72x, saved 26.8%, G +2.13 +- 0.46, GIST - Full -0.72
- social twitter: reach 1.35x, saved 21.2%, G +0.33 +- 0.11, GIST - Full -0.40
- social facebook: reach 2.43x, saved 19.7%, G +0.07 +- 0.04, GIST - Full -0.02
- p2p gnutella25: reach 1.30x, saved 6.8%, G -0.04 +- 0.05, GIST - Full +0.01
- p2p gnutella30: reach 1.39x, saved 2.4%, G -0.05 +- 0.07, GIST - Full -0.00
- p2p gnutella31: reach 1.29x, saved 1.1%, G -0.04 +- 0.05, GIST - Full +0.00
- road road-ca: reach 2.60x, saved 6.8%, G -0.85 +- 0.62, GIST - Full -0.27
- road road-pa: reach 4.28x, saved 5.4%, G -0.85 +- 0.45, GIST - Full -0.10
- road road-tx: reach 3.42x, saved 3.9%, G -0.84 +- 0.56, GIST - Full -0.14
- Spearman reach vs GIST/Full nodes without road (12): -0.88 (p=0.0002)
