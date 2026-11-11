"""Per-repair cost and time, Full vs GIST (selective) vs whole graph, median over seeds, from D1 raw (eval.tex tab:runtime).
Times are D1 in-stream records on shared GPUs: indicative only."""
import json, glob, numpy as np
for f in sorted(glob.glob("results/D1_main/raw/streams/D1/*.json")):
    d = json.load(open(f)); hp = d["hyperparameters"]
    if hp["variant"] not in ("arxiv", "arxiv_year", "elliptic", "products"): continue
    def g(a, k):
        v = [r[k] for r in d["rows"] if r["arm"] == a and not r.get("oom") and r.get(k)]
        return float(np.median(v)) if v else float("nan")
    print(f"{hp['variant']:10s} {hp['model']:5s} full {g('full','retrain_p50_ms'):7.1f} ms gist {g('selective','retrain_p50_ms'):7.1f} ms "
          f"x{g('full','retrain_p50_ms')/g('selective','retrain_p50_ms'):.2f} | whole {g('global_trigger','retrain_p50_ms'):7.1f} ms "
          f"| nodes {g('full','nodes'):8.0f}/{g('selective','nodes'):8.0f} | GFLOPs {g('full','gflops_per_event'):6.1f}/{g('selective','gflops_per_event'):6.1f} "
          f"| peakMB {g('full','retrain_peak_mb'):6.0f}/{g('selective','retrain_peak_mb'):6.0f}")
