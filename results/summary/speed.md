## Repair time (sync on, ms, mean of seed p50s; speedup mean of per-seed ratios +- sd)

- GAT: N=2048: 131->101 x1.31+-0.17 [1.16,1.59] net of community x1.18 nodes 65.0% (48.5K->31.5K) community 11.3 ms | N=4096: 203->140 x1.46+-0.18 [1.29,1.74] net of community x1.36 nodes 63.8% (69.4K->44.3K) community 10.5 ms | N=8192: 296->184 x1.62+-0.16 [1.40,1.83] net of community x1.53 nodes 63.8% (91.8K->58.6K) community 10.6 ms
- SAGE: N=2048: 83->59 x1.44+-0.27 [1.10,1.75] net of community x1.21 nodes 63.1% (48.4K->30.5K) community 11.0 ms | N=4096: 128->86 x1.55+-0.28 [1.14,1.84] net of community x1.36 nodes 63.7% (69.3K->44.2K) community 11.3 ms | N=8192: 188->115 x1.65+-0.17 [1.41,1.88] net of community x1.48 nodes 64.8% (91.8K->59.5K) community 13.2 ms
- GCN: N=2048: 85->68 x1.26+-0.15 [1.11,1.50] net of community x1.06 nodes 63.8% (48.4K->30.9K) community 12.3 ms | N=4096: 131->89 x1.49+-0.23 [1.24,1.75] net of community x1.32 nodes 63.3% (69.3K->43.8K) community 11.7 ms | N=8192: 188->119 x1.59+-0.12 [1.47,1.73] net of community x1.46 nodes 62.9% (91.7K->57.7K) community 10.2 ms
- GIN: N=2048: 83->50 x1.66+-0.15 [1.51,1.83] net of community x1.34 nodes 60.6% (48.4K->29.4K) community 11.8 ms | N=4096: 129->74 x1.74+-0.09 [1.68,1.90] net of community x1.51 nodes 62.0% (69.3K->43.0K) community 11.3 ms | N=8192: 185->105 x1.77+-0.08 [1.67,1.87] net of community x1.61 nodes 63.6% (91.8K->58.3K) community 10.3 ms

## Mean repair split (ms): sampling + model update

- GAT: N=2048: Full 73+58 GIST 53+48 | N=4096: Full 116+87 GIST 73+64 | N=8192: Full 171+125 GIST 98+85
- SAGE: N=2048: Full 73+10 GIST 46+14 | N=4096: Full 115+13 GIST 68+16 | N=8192: Full 171+17 GIST 95+20
- GCN: N=2048: Full 74+10 GIST 53+17 | N=4096: Full 117+14 GIST 74+20 | N=8192: Full 170+18 GIST 98+24
- GIN: N=2048: Full 72+11 GIST 40+13 | N=4096: Full 114+15 GIST 61+15 | N=8192: Full 167+18 GIST 88+19

## End to end (sync off, ms/step)

- GAT: N=2048: full 42.32 gist 41.98 periodic-no-detect 41.94 gist<full 4/5 gist<=periodic 3/5 | N=4096: full 43.20 gist 43.01 periodic-no-detect 43.22 gist<full 3/5 gist<=periodic 3/5 | N=8192: full 44.85 gist 43.38 periodic-no-detect 44.23 gist<full 5/5 gist<=periodic 4/5
- SAGE: N=2048: full 34.40 gist 34.47 periodic-no-detect 34.12 gist<full 3/5 gist<=periodic 0/5 | N=4096: full 35.21 gist 34.73 periodic-no-detect 34.89 gist<full 5/5 gist<=periodic 4/5 | N=8192: full 36.69 gist 35.18 periodic-no-detect 35.03 gist<full 5/5 gist<=periodic 2/5
- GCN: N=2048: full 35.02 gist 35.39 periodic-no-detect 35.02 gist<full 0/5 gist<=periodic 0/5 | N=4096: full 36.08 gist 36.00 periodic-no-detect 36.25 gist<full 4/5 gist<=periodic 4/5 | N=8192: full 37.00 gist 36.41 periodic-no-detect 36.55 gist<full 4/5 gist<=periodic 3/5
- GIN: N=2048: full 33.69 gist 33.35 periodic-no-detect 33.83 gist<full 5/5 gist<=periodic 4/5 | N=4096: full 34.63 gist 33.70 periodic-no-detect 34.08 gist<full 5/5 gist<=periodic 5/5 | N=8192: full 35.19 gist 34.18 periodic-no-detect 35.04 gist<full 5/5 gist<=periodic 5/5

## Overhead on the inference path (baseline arm)

- GAT: infer p50 7.76 detect p50 0.37 = 4.8% of infer, 0.95% of step; sampling p50 25.8; wall/1000 steps det on 40.17 s off 39.89 s; repeat-run gap 1.50 s
- SAGE: infer p50 0.79 detect p50 0.35 = 43.8% of infer, 1.05% of step; sampling p50 26.4; wall/1000 steps det on 33.40 s off 33.09 s; repeat-run gap 1.68 s
- GCN: infer p50 0.98 detect p50 0.37 = 37.4% of infer, 1.09% of step; sampling p50 26.6; wall/1000 steps det on 33.89 s off 33.69 s; repeat-run gap 1.55 s
- GIN: infer p50 0.93 detect p50 0.36 = 38.5% of infer, 1.11% of step; sampling p50 25.6; wall/1000 steps det on 32.54 s off 32.21 s; repeat-run gap 2.85 s

## E7 serving (seeds 0-4, 20 s window per condition)

- GAT idle: n=5 oom=0 p50 7.35 p99 7.94 (x1.00 idle) requests 2719 repair step p50 None nodes None
- GAT full: n=5 oom=0 p50 18.65 p99 19.42 (x2.45 idle) requests 1090 repair step p50 218.5 nodes 91891.8
- GAT selective: n=5 oom=0 p50 18.40 p99 19.36 (x2.44 idle) requests 1109 repair step p50 171.1 nodes 74558.4
- GAT global: n=5 oom=5 p50 7.36 p99 8.10 (x1.02 idle) requests 2710 repair step p50 None nodes 169343.0
- SAGE idle: n=5 oom=0 p50 0.50 p99 0.53 (x1.00 idle) requests 39867 repair step p50 None nodes None
- SAGE full: n=5 oom=0 p50 2.74 p99 2.81 (x5.27 idle) requests 8497 repair step p50 11.6 nodes 91891.8
- SAGE selective: n=5 oom=0 p50 2.71 p99 2.84 (x5.33 idle) requests 8571 repair step p50 8.8 nodes 74558.4
- SAGE global: n=5 oom=0 p50 2.75 p99 2.83 (x5.31 idle) requests 7372 repair step p50 43.4 nodes 169343.0
