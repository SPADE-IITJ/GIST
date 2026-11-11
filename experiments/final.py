"""Phase D entry point: run any experiments / linkpred / timing driver on the evaluation seeds, into results/<part>/raw/.

    python -m experiments.final D1_main streams --exp D1 --model gat --variant arxiv_year --arms ... --seeds 0 ... 9
    python -m experiments.final D6_graphs linkpred run --model gat --graph youtube --seeds 0 ... 9 --tag _final
    python -m experiments.final D3_speed gate_overhead --seeds 0 1 2 3 4 --budgets 8192 ... --out results/D3_speed/raw/gat.json
    python -m experiments.final D2_exact backbones --seeds 0 ... 9

Seeds 0-9 are allowed only through this entry point and ; every
other path still refuses them. Each part writes only inside results/<part>/raw/.
"""
from __future__ import annotations

import importlib
import importlib.util
import os
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
FINAL = ROOT / "results"
GIST_MODULES = ("streams", "e1_census", "e3a_cost", "e4_fidelity", "e5_static", "e7_interference", "limits", "e1_mix")


def main():
    if len(sys.argv) < 3:
        raise SystemExit(__doc__)
    part, mod, rest = sys.argv[1], sys.argv[2], sys.argv[3:]
    raw = FINAL / part / "raw"
    raw.mkdir(parents=True, exist_ok=True)
    os.environ["GIST_EVAL"] = "1"
    os.environ["GIST_RESULTS"] = str(raw)
    os.environ.pop("GIST_SMOKE", None) if not os.environ.get("FINAL_SMOKE") else None
    sys.path.insert(0, str(ROOT))
    if mod in GIST_MODULES:
        m = importlib.import_module(f"experiments.{mod}")
        sys.argv = [f"experiments.{mod}", *rest]
        m.main()
    elif mod == "linkpred":                       # linkpred/run.py: its own guard and results directory
        import linkpred.run as S
        from experiments import common as C
        S.RES = raw
        orig = S.guard
        S.guard = lambda seeds, reported=False: None if C.eval_seeds_allowed() and set(seeds) <= set(range(10)) else orig(seeds, reported)
        sys.argv = ["linkpred/run.py", *rest]
        S.main()
    elif mod == "gate_overhead":                # scripts/gate_overhead.py refuses 0-9 through EVALUATION_SEEDS
        from experiments import common as C
        if not C.eval_seeds_allowed():
            raise SystemExit("evaluation seeds not allowed")
        spec = importlib.util.spec_from_file_location("gate_overhead", ROOT / "scripts" / "gate_overhead.py")
        G = importlib.util.module_from_spec(spec)
        sys.argv = ["scripts/gate_overhead.py", *rest]
        spec.loader.exec_module(G)
        G.EVALUATION_SEEDS = frozenset()
        raise SystemExit(G.main())
    elif mod == "backbones":                    # PyG Temporal lives in env-backbones ($B)
        b = os.environ.get("B", "python")
        raise SystemExit(subprocess.call([b, "-m", "experiments.backbones", *rest], cwd=ROOT, env=os.environ))
    else:
        raise SystemExit(f"unknown module {mod!r}")


if __name__ == "__main__":
    main()
