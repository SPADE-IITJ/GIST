"""SNAP edge lists -> an undirected simple graph (``lo < hi`` pairs) with an optional arrival time per edge.

Time stamps need no extra download:
* arXiv citation graphs (HepTh, HepPh): node ids are arXiv numbers ``yymmnnn``, so the month is in the id. An edge
  takes the later of its two papers' months (the citing paper's).
* US patents: utility patent numbers are issued in grant order, so the larger number is the citing patent and its
  rank is the arrival time.
Every other graph has no time stamps; its edges arrive in a seeded random order (``stream.build_plan``).
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import torch

ROOT = Path(__file__).resolve().parent / "data"
RAW, CACHE = ROOT / "raw", ROOT / "cache"

GRAPHS = {
    "facebook": dict(file="facebook_combined.txt", family="social"),
    "twitter": dict(file="twitter_combined.txt", family="social"),
    "youtube": dict(file="com-youtube.ungraph.txt", family="social"),
    "web-stanford": dict(file="web-Stanford.txt", family="web"),
    "web-google": dict(file="web-Google.txt", family="web"),
    "web-berkstan": dict(file="web-BerkStan.txt", family="web"),
    "gnutella25": dict(file="p2p-Gnutella25.txt", family="p2p"),
    "gnutella30": dict(file="p2p-Gnutella30.txt", family="p2p"),
    "gnutella31": dict(file="p2p-Gnutella31.txt", family="p2p"),
    "road-pa": dict(file="roadNet-PA.txt", family="road"),
    "road-tx": dict(file="roadNet-TX.txt", family="road"),
    "road-ca": dict(file="roadNet-CA.txt", family="road"),
    "hepth": dict(file="Cit-HepTh.txt", family="citation", time="arxiv"),
    "hepph": dict(file="Cit-HepPh.txt", family="citation", time="arxiv"),
    "patents": dict(file="cit-Patents.txt", family="citation", time="patent"),
}


def arxiv_month(ids: np.ndarray) -> np.ndarray:
    """Months since year 0 from integer arXiv ids ``yymmnnn`` (leading zeros were lost when parsed as int)."""
    yy, mm = ids // 100000, (ids // 1000) % 100
    year = np.where(yy >= 90, 1900 + yy, 2000 + yy)
    return year * 12 + np.clip(mm, 1, 12) - 1


def _read(path: Path) -> tuple[np.ndarray, np.ndarray]:
    df = pd.read_csv(path, sep=r"\s+", comment="#", header=None, usecols=[0, 1], engine="c")
    df = df.apply(pd.to_numeric, errors="coerce").dropna().astype(np.int64)
    return df[0].to_numpy(), df[1].to_numpy()


def simple_graph(a: np.ndarray, b: np.ndarray, t: np.ndarray | None = None):
    """Relabel to 0..n-1, drop self-loops, keep each unordered pair once (its earliest time). Returns lo, hi, n, t, ids."""
    ids, inv = np.unique(np.concatenate([a, b]), return_inverse=True)
    a, b = inv[: len(a)], inv[len(a):]
    keep = a != b
    a, b = a[keep], b[keep]
    t = None if t is None else t[keep]
    lo, hi = np.minimum(a, b), np.maximum(a, b)
    n = len(ids)
    order = np.argsort(t, kind="stable") if t is not None else np.arange(len(lo))
    pair = lo[order] * n + hi[order]
    _, first = np.unique(pair, return_index=True)
    sel = order[first]
    return lo[sel], hi[sel], n, (None if t is None else t[sel]), ids


def load(name: str) -> dict:
    """``{edges: LongTensor[2, m], num_nodes, time: float64 array or None, family}``, cached under data/cache."""
    spec = GRAPHS[name]
    cache = CACHE / f"{name}.pt"
    if cache.exists():
        return torch.load(cache, weights_only=False)
    a, b = _read(RAW / spec["file"])
    kind = spec.get("time")
    if kind == "arxiv":
        t = np.maximum(arxiv_month(a), arxiv_month(b)).astype(np.float64)
    elif kind == "patent":
        t = np.maximum(a, b).astype(np.float64)
    else:
        t = None
    lo, hi, n, t, _ = simple_graph(a, b, t)
    out = dict(edges=torch.from_numpy(np.stack([lo, hi])).long(), num_nodes=int(n), time=t, family=spec["family"])
    CACHE.mkdir(parents=True, exist_ok=True)
    torch.save(out, cache)
    return out
