"""Run several experiments jobs in order as one queue job, so a dependency needs no BARRIER (which drains every lane):

    python -m experiments.chain calibrate probe --model sage --variant products ;; calibrate threshold ... ;; final ...

Each step runs as `python -m experiments.<step>`. A step that exits 0 leaves a marker in runs/chain/, so a requeued
chain (OOM retry, restart) skips the steps it already finished; the steps themselves resume per seed. The first failing
step ends the chain with its exit code.
"""
from __future__ import annotations

import hashlib
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
MARKS = ROOT / "runs" / "chain"


def main():
    steps, cur = [], []
    for a in sys.argv[1:]:
        if a == ";;":
            steps.append(cur); cur = []
        else:
            cur.append(a)
    steps.append(cur)
    steps = [s for s in steps if s]
    if not steps:
        raise SystemExit(__doc__)
    MARKS.mkdir(parents=True, exist_ok=True)
    for s in steps:
        line = " ".join(s)
        mark = MARKS / (hashlib.sha1(line.encode()).hexdigest()[:16] + ".done")
        if mark.exists():
            print(f"chain: skip (done) {line}", flush=True)
            continue
        print(f"chain: run {line}", flush=True)
        rc = subprocess.call([sys.executable, "-m", f"experiments.{s[0]}", *s[1:]], cwd=ROOT)
        if rc != 0:
            print(f"chain: step failed rc={rc}: {line}", flush=True)
            raise SystemExit(rc)
        mark.write_text(line + "\n")


if __name__ == "__main__":
    main()
