"""A replayable link stream, materialised once per seed and replayed by every arm.

Every edge gets an arrival **key**:
* ``BASE`` (-4): the base snapshot G0, visible from the start; the model is warmed up on it.
* ``VAL`` (-2): a slice of G0 held out of message passing during warmup (the warmup probe scores it); visible in
  the stream.
* slice ``s`` of the stream: half its edges are **context** (key ``2s-1``), revealed before step ``s`` is scored, and
  half are **queries** (key ``2s``), scored at step ``s`` before they are inserted.

At step ``s`` the visible graph is every edge with key < ``2s``. Nothing scored at ``s`` is visible at ``s``, and an
update at ``s`` (after insertion) sees keys < ``2s+1``, so a query edge is always scored before any model trains on it.

**Injected drift** (graphs without time stamps): a link-pattern shift. From its step on, each arriving edge with an
endpoint in a source community C is, with probability p, re-pointed to a node of a target community D (drawn
proportional to base degree). The model learnt that C links within C; now C links to D. Ground truth for detection
is the pair {C, D} and the event step.

**Natural drift** (arXiv and patent citations): edges arrive in time order; nothing is injected. Its null twin keeps
the same base and shuffles the stream (``shuffle_time``), which removes temporal drift.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import List

import numpy as np
import torch

from gist.drift.partitioners import metis_partition

BASE, VAL = -4, -2
NEVER = np.iinfo(np.int64).max // 4


def _metis(g0: torch.Tensor, n: int, k: int, tries: int = 3) -> np.ndarray:
    """METIS assignment, recomputed if it comes back out of range (seen once, 2026-10-19: web-berkstan seed 113 got
    0x0101... ids and did not reproduce). METIS is deterministic, so a valid retry is the same partition."""
    for _ in range(tries):
        cid = metis_partition(g0, n, num_parts=k).numpy().astype(np.int64)
        if len(cid) == n and cid.min() >= 0 and cid.max() < k:
            return cid
    raise RuntimeError(f"METIS gave out-of-range ids {tries} times (n={n}, k={k})")


@dataclass
class DriftEvent:
    step: int
    source: int
    target: int
    strength: float
    redirected: int = 0


@dataclass
class LinkPlan:
    num_nodes: int
    edges: torch.Tensor            # [2, m] lo < hi, after drift
    key: torch.Tensor              # [m]
    steps: int
    scored: List[torch.Tensor]     # per step: ids of query edges scored there (both endpoints present)
    cid: torch.Tensor              # [n] community, -1 if none
    num_comms: int
    events: List[DriftEvent]
    drift_step: np.ndarray         # [num_comms] first step the community drifted (source or target); NEVER if not
    redirected: torch.Tensor       # [m] bool
    first_key: torch.Tensor        # [n] smallest incident key; present at step t iff first_key < 2t
    stream_ids: torch.Tensor       # stream edge ids sorted by key (for update windows)
    seed: int
    temporal: bool
    null: bool = False
    meta: dict = field(default_factory=dict)

    def window(self, t: int, width: int) -> torch.Tensor:
        """Stream edges of slices t-width+1 .. t (all visible once slice t is inserted)."""
        k = self.key[self.stream_ids]
        lo = torch.searchsorted(k, torch.tensor(2 * (t - width + 1) - 1))
        hi = torch.searchsorted(k, torch.tensor(2 * t + 1))
        return self.stream_ids[lo:hi]

    def present_count(self, t: int) -> int:
        return int(torch.searchsorted(self.meta["fk_sorted"], torch.tensor(2 * t)))


def num_communities(n: int) -> int:
    return int(np.clip(round(n / 1000), 16, 256))


def build_plan(edges: torch.Tensor, num_nodes: int, *, seed: int, steps: int = 1000, time=None,
               shuffle_time: bool = False, inject: bool = True, base_fraction: float = 0.5,
               val_fraction: float = 0.05, scored_per_step: int = 512, min_slice: int = 32,
               events_per_1000: float = 8.0, num_comms: int | None = None) -> LinkPlan:
    lo, hi = edges[0].numpy().copy(), edges[1].numpy().copy()
    m, n = len(lo), int(num_nodes)
    r_order, r_val, r_split, r_score, r_drift = (np.random.default_rng([seed, s]) for s in (0xA1, 0xA2, 0xA3, 0xA4, 0xA5))

    nb = int(round(m * base_fraction))
    if time is not None:
        order = np.lexsort((r_order.random(m), np.asarray(time)))
        base, stream = order[:nb], order[nb:]
        if shuffle_time:
            stream = np.random.default_rng([seed, 0xA6]).permutation(stream)
    else:
        order = r_order.permutation(m)
        base, stream = order[:nb], order[nb:]

    key = np.empty(m, np.int64)
    key[base] = BASE
    key[r_val.choice(base, int(round(val_fraction * nb)), replace=False)] = VAL
    T = int(min(steps, len(stream) // min_slice))
    if T < 10:
        raise ValueError(f"stream too short: {len(stream)} edges")
    slices = np.array_split(stream, T)
    for s, sl in enumerate(slices):
        ctx = r_split.random(len(sl)) < 0.5
        key[sl[ctx]] = 2 * s - 1
        key[sl[~ctx]] = 2 * s

    # communities: METIS on G0; nodes isolated in G0 inherit from their earliest neighbour that has one
    k = num_comms or num_communities(n)
    b0 = key < 0
    g0 = torch.from_numpy(np.stack([np.r_[lo[b0], hi[b0]], np.r_[hi[b0], lo[b0]]]))
    cid = _metis(g0, n, k)
    deg0 = np.bincount(np.r_[lo[b0], hi[b0]], minlength=n)
    cid[deg0 == 0] = -1
    for _ in range(10):
        miss = cid < 0
        if not miss.any():
            break
        a = (cid[lo] >= 0) & miss[hi]
        b = (cid[hi] >= 0) & miss[lo]
        if not (a.any() or b.any()):
            break
        tgt, src, kk = np.r_[hi[a], lo[b]], np.r_[lo[a], hi[b]], np.r_[key[a], key[b]]
        o = np.lexsort((kk, tgt))
        tgt, src = tgt[o], src[o]
        first = np.r_[True, tgt[1:] != tgt[:-1]]
        cid[tgt[first]] = cid[src[first]]

    # injected link-pattern drift
    events: List[DriftEvent] = []
    drift_step = np.full(k, NEVER, np.int64)
    redirected = np.zeros(m, bool)
    if inject and time is None:
        E = max(3, int(round(events_per_1000 * T / 1000)))
        # one event per equal bin of [T/10, T - T/20), placed in the bin's middle half: spaced, never clustered
        lo_s, hi_s = max(2, T // 10), T - max(2, T // 20)
        edges_b = np.linspace(lo_s, hi_s, E + 1)
        at = [int(a + (b - a) * r_drift.uniform(0.25, 0.75)) for a, b in zip(edges_b[:-1], edges_b[1:])]
        sizes = np.bincount(cid[cid >= 0], minlength=k)
        cands = np.nonzero(sizes >= 0.5 * sizes[sizes > 0].mean())[0]
        sources = r_drift.choice(cands, E, replace=False)
        slice_of = np.where(key >= -1, (key + 1) // 2, -1)
        for s, c in zip(sorted(at), sources):
            d = int(r_drift.choice(cands[cands != c]))
            p = float(r_drift.uniform(0.5, 0.9))
            members = np.nonzero(cid == d)[0]
            w = deg0[members] + 1.0
            in_lo, in_hi = cid[lo] == c, cid[hi] == c
            mask = (slice_of >= s) & (in_lo | in_hi) & (r_drift.random(m) < p)
            new = r_drift.choice(members, int(mask.sum()), p=w / w.sum())
            keep_lo = in_lo[mask]                      # replace the endpoint outside C (hi if both are in C)
            nl = np.where(keep_lo, lo[mask], new)
            nh = np.where(keep_lo, new, hi[mask])
            ok = nl != nh
            idx = np.nonzero(mask)[0][ok]
            nl, nh = nl[ok], nh[ok]
            lo[idx], hi[idx] = np.minimum(nl, nh), np.maximum(nl, nh)
            redirected[idx] = True
            drift_step[c] = min(drift_step[c], s)
            drift_step[d] = min(drift_step[d], s)
            events.append(DriftEvent(int(s), int(c), d, p, int(len(idx))))

    fk = np.full(n, NEVER, np.int64)
    np.minimum.at(fk, lo, key)
    np.minimum.at(fk, hi, key)

    stream_mask = key >= -1
    stream_ids = np.nonzero(stream_mask)[0]
    stream_ids = stream_ids[np.argsort(key[stream_ids], kind="stable")]
    qk = key[stream_ids]
    scored = []
    for s in range(T):
        a, b = np.searchsorted(qk, 2 * s), np.searchsorted(qk, 2 * s + 1)
        q = stream_ids[a:b]
        q = q[(fk[lo[q]] < 2 * s) & (fk[hi[q]] < 2 * s)]
        if len(q) > scored_per_step:
            q = np.sort(r_score.choice(q, scored_per_step, replace=False))
        scored.append(torch.from_numpy(q).long())

    fk_t = torch.from_numpy(fk)
    plan = LinkPlan(num_nodes=n, edges=torch.from_numpy(np.stack([lo, hi])).long(), key=torch.from_numpy(key),
                    steps=T, scored=scored, cid=torch.from_numpy(cid), num_comms=k, events=events,
                    drift_step=drift_step, redirected=torch.from_numpy(redirected), first_key=fk_t,
                    stream_ids=torch.from_numpy(stream_ids).long(), seed=seed, temporal=time is not None,
                    null=(time is None and not inject) or (time is not None and shuffle_time))
    srt = torch.sort(fk_t)
    plan.meta["fk_sorted"], plan.meta["present_order"] = srt.values, srt.indices
    return plan
