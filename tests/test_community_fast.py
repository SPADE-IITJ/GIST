"""The vectorised community stage reproduces the per-node loops it replaced, bit for bit.

G1 found the community stage (~330 ms per event on Arxiv) was what made selective slower than full end to end:
a Python loop over every sampled node in ``observe`` and a synchronous METIS refresh on the event path. The loops
were vectorised and the refresh can run in the background (``drift.community_refresh_lag``). The old code is copied
below as the reference; every field is compared with ``==``, not a tolerance.
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Dict, List

import numpy as np
import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from gist.drift.global_communities import GlobalCommunityTracker
from gist.drift.partitioners import assignment_to_groups
from gist.drift.signals import RunningStandardiser


# --- the replaced implementations, verbatim apart from being free functions ----------------------------------

def loop_groups(assignment, min_size):
    groups = []
    valid = assignment[assignment >= 0]
    if valid.numel() == 0:
        return groups
    for cid in torch.unique(valid).tolist():
        members = (assignment == cid).nonzero(as_tuple=True)[0]
        if members.numel() >= min_size:
            groups.append(members.tolist())
    return groups


def loop_match(previous, groups, next_id):
    overlaps = []
    for group_idx, members in enumerate(groups):
        prior = previous[torch.as_tensor(members, dtype=torch.long)]
        prior = prior[prior >= 0]
        if prior.numel() == 0:
            continue
        ids, counts = torch.unique(prior, return_counts=True)
        for cid, count in zip(ids.tolist(), counts.tolist()):
            overlaps.append((int(count), group_idx, int(cid)))
    overlaps.sort(reverse=True)
    matched, claimed = {}, set()
    for _, group_idx, cid in overlaps:
        if group_idx in matched or cid in claimed:
            continue
        matched[group_idx] = cid
        claimed.add(cid)
    for group_idx in range(len(groups)):
        if group_idx not in matched:
            matched[group_idx] = next_id
            next_id += 1
    return matched, next_id


class LoopObserver:
    """GlobalCommunityTracker.observe before vectorisation."""

    def __init__(self, ema=0.1, warmup=20, threshold=2.11):
        self.ema, self.threshold = ema, threshold
        self._gate_norm = RunningStandardiser(ema=ema, warmup=warmup)
        self._reference: Dict[int, np.ndarray] = {}
        self.history: Dict[int, List[tuple]] = {}

    def observe(self, assignment, node_ids, embeddings, step):
        emb = embeddings.detach().cpu().numpy()
        totals, counts = {}, {}
        for local_idx, global_id in enumerate(node_ids.cpu().tolist()):
            cid = int(assignment[global_id])
            if cid < 0:
                continue
            current = emb[local_idx]
            previous = self._reference.get(global_id)
            if previous is None:
                self._reference[global_id] = current
                drift = 0.0
            else:
                drift = float(np.linalg.norm(current - previous))
                self._reference[global_id] = self.ema * current + (1 - self.ema) * previous
            totals[cid] = totals.get(cid, 0.0) + drift
            counts[cid] = counts.get(cid, 0) + 1
        scores = {cid: totals[cid] / counts[cid] for cid in totals}
        for cid, score in scores.items():
            self.history.setdefault(cid, []).append((step, score))
        max_score = max(scores.values()) if scores else 0.0
        max_z = self._gate_norm.update(max_score)
        flag = bool(self._gate_norm.ready and max_z > self.threshold)
        return scores, counts, max_score, max_z, flag


# --- fixtures ---------------------------------------------------------------------------------------------------

N = 600


def _graph(seed, edges=3000):
    g = torch.Generator().manual_seed(seed)
    # Planted blocks so Louvain finds stable communities, plus noise edges.
    block = torch.arange(N) // 50
    src = torch.randint(0, N, (edges,), generator=g)
    dst = (block[src] * 50 + torch.randint(0, 50, (edges,), generator=g)) % N
    noise = torch.randint(0, N, (2, edges // 10), generator=g)
    ei = torch.cat([torch.stack([src, dst]), noise], dim=1)
    ei = ei[:, ei[0] != ei[1]]
    return torch.cat([ei, ei.flip(0)], dim=1)


# --- tests ------------------------------------------------------------------------------------------------------

@pytest.mark.parametrize("seed", [0, 1, 2])
def test_groups_identical_to_loop(seed):
    g = torch.Generator().manual_seed(seed)
    assignment = torch.randint(-1, 40, (5000,), generator=g)
    assignment[assignment == 7] = 99   # a sparse id
    assert assignment_to_groups(assignment, 120) == loop_groups(assignment, 120)
    assert assignment_to_groups(torch.full((10,), -1), 1) == []


@pytest.mark.parametrize("partitioner", ["louvain", "metis"])
def test_refresh_identical_to_loop(partitioner):
    from gist.drift.partitioners import metis_available, partition_graph
    if partitioner == "metis" and not metis_available():
        pytest.skip("METIS unavailable")
    tracker = GlobalCommunityTracker(N, partitioner=partitioner, num_parts=12, min_size=15, seed=0)
    previous, next_id = torch.full((N,), -1, dtype=torch.long), 0
    for step, seed in enumerate([0, 1, 2]):
        ei = _graph(seed)
        part = tracker.refresh(ei, step)
        raw, _ = partition_graph(ei, N, backend=partitioner, seed=0, num_parts=12)
        found = loop_groups(raw, 15)
        found.sort(key=len, reverse=True)
        matched, next_id = loop_match(previous, found, next_id)
        expect = torch.full((N,), -1, dtype=torch.long)
        sizes = {}
        for gi, cid in matched.items():
            expect[torch.as_tensor(found[gi])] = cid
            sizes[cid] = len(found[gi])
        assert torch.equal(part.assignment, expect)
        assert part.sizes == sizes and list(part.sizes) == list(sizes)
        previous = expect
    assert part.refresh_count == 3 and tracker._next_id == next_id


def test_observe_identical_to_loop():
    tracker = GlobalCommunityTracker(N, partitioner="louvain", min_size=15, seed=0, warmup=3)
    tracker.refresh(_graph(0), 0)
    tracker.partition.assignment[::17] = -1          # unassigned nodes are skipped
    loop = LoopObserver(warmup=3)
    g = torch.Generator().manual_seed(5)
    for step in range(25):
        ids = torch.randperm(N, generator=g)[: 300 + step]
        emb = torch.randn(ids.numel(), 7, generator=g) * (1 + (step > 15))
        got = tracker.observe(ids, emb, step)
        scores, counts, max_score, max_z, flag = loop.observe(tracker.partition.assignment, ids, emb, step)
        assert got.scores == scores and list(got.scores) == list(scores)   # values and order
        assert got.counts == counts
        assert (got.max_score, got.max_z, got.flag) == (max_score, max_z, flag)
        assert tracker.drifting_communities(got) == tracker.drifting_communities(
            type(got)(scores=scores, counts=counts))
    assert tracker.history == loop.history
    for nid, ref in loop._reference.items():
        assert np.array_equal(tracker._reference[nid], ref)


def test_observe_rejects_duplicate_ids():
    tracker = GlobalCommunityTracker(N, partitioner="louvain", seed=0)
    tracker.refresh(_graph(0), 0)
    with pytest.raises(ValueError):
        tracker.observe(torch.tensor([3, 3]), torch.zeros(2, 4), 0)


def test_background_refresh_installs_on_schedule_and_matches_sync():
    lag, interval = 5, 20
    bg = GlobalCommunityTracker(N, partitioner="louvain", refresh_interval=interval, refresh_lag=lag, seed=0)
    sync = GlobalCommunityTracker(N, partitioner="louvain", refresh_interval=interval, seed=0)
    graphs = {s: _graph(s // interval) for s in range(0, 70)}
    installs = []
    for step in range(70):
        before = bg.partition.refresh_count
        bg.tick(graphs[step], step)
        if bg.partition.refresh_count != before:
            installs.append(step)
            sync.refresh(graphs[step - lag], step - lag)   # the snapshot the background run saw
            assert torch.equal(bg.partition.assignment, sync.partition.assignment)
            assert bg.partition.refreshed_at == step - lag
    bg.close()
    assert installs == [5, 25, 45, 65]


def test_background_refresh_waits_when_no_partition_exists():
    bg = GlobalCommunityTracker(N, partitioner="louvain", refresh_interval=20, refresh_lag=5, seed=0)
    bg.tick(_graph(0), 0)
    assert bg.partition.refreshed_at < 0
    assert bg.ensure(_graph(0), 1) is True            # an event at step 1 cannot run without one
    assert bg.partition.refreshed_at == 0
    assert bg.ensure(_graph(0), 2) is False           # and it does not refresh synchronously after that
    fresh = GlobalCommunityTracker(N, partitioner="louvain", refresh_interval=20, refresh_lag=5, seed=0)
    assert fresh.ensure(_graph(0), 3) is True and fresh.partition.refreshed_at == 3   # no tick ever ran
    bg.close(); fresh.close()


def test_sync_mode_is_unchanged():
    t = GlobalCommunityTracker(N, partitioner="louvain", refresh_interval=200, seed=0)
    t.tick(_graph(0), 0)                               # a no-op without a lag
    assert t.partition.refreshed_at < 0
    assert t.ensure(_graph(0), 0) is True and t.ensure(_graph(0), 199) is False and t.ensure(_graph(0), 200) is True


@pytest.mark.parametrize("lag", [-1, 200, 250])
def test_lag_must_be_shorter_than_interval(lag):
    with pytest.raises(ValueError):
        GlobalCommunityTracker(N, refresh_interval=200, refresh_lag=lag)
