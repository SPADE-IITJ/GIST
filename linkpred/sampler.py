"""Neighbour sampling on a growing graph without rebuilding it: a time-keyed CSR.

All edges of the stream (both directions) are stored once, each row sorted by arrival key. The neighbours of ``u``
visible below key ``kv`` are then a prefix of its row, found by one ``searchsorted`` on ``row * K + key``. Sampling
draws ``fanout[l]`` of them per node (all of them when there are no more), so an arm never rebuilds a graph.

``exclude`` drops given node pairs from the sampled computation graph: an update's own target edges must not carry
messages, or the model would learn to predict edges it can see.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Sequence

import torch
from torch import Tensor

from .stream import BASE, LinkPlan


@dataclass
class Sub:
    n_id: Tensor          # global ids; the first num_roots are the roots, in the order given
    edge_index: Tensor    # [2, E] local ids, message direction src -> dst
    num_roots: int


def pair_key(a: Tensor, b: Tensor, n: int) -> Tensor:
    return torch.minimum(a, b) * n + torch.maximum(a, b)


class TemporalSampler:
    def __init__(self, plan: LinkPlan, device) -> None:
        self.device = torch.device(device)
        self.n = plan.num_nodes
        lo, hi = plan.edges
        row = torch.cat([lo, hi])
        col = torch.cat([hi, lo])
        key = torch.cat([plan.key, plan.key])
        self.K = 2 * plan.steps + 8
        g = row * self.K + (key - BASE)
        o = torch.argsort(g)
        self.g = g[o].to(self.device)
        self.col = col[o].to(self.device)
        self.rowptr = torch.zeros(self.n + 1, dtype=torch.long)
        self.rowptr[1:] = torch.bincount(row, minlength=self.n).cumsum(0)
        self.rowptr = self.rowptr.to(self.device)
        self.loc = torch.full((self.n,), -1, dtype=torch.long, device=self.device)
        self._key_sorted = key[o].to(self.device)

    def visible_degree(self, nodes: Tensor, kv: int) -> Tensor:
        end = torch.searchsorted(self.g, nodes * self.K + (kv - BASE))
        return end - self.rowptr[nodes]

    def sample(self, roots: Tensor, kv: int, fanout: Sequence[int], gen: Optional[torch.Generator] = None,
               exclude: Optional[Tensor] = None) -> Sub:
        """Sampled L-hop computation graph of unique ``roots`` on the edges with key < ``kv``."""
        dev, n = self.device, self.n
        roots = roots.to(dev)
        ids = [roots]
        self.loc[roots] = torch.arange(len(roots), device=dev)
        count = len(roots)
        srcs, dsts = [], []
        front = roots
        excl = None if exclude is None else torch.sort(exclude.to(dev)).values
        for k in fanout:
            if len(front) == 0:
                break
            start = self.rowptr[front]
            cnt = self.visible_degree(front, kv)
            take = torch.clamp(cnt, max=int(k))
            total = int(take.sum())
            if total == 0:
                break
            rep = torch.repeat_interleave(torch.arange(len(front), device=dev), take)
            off = torch.cumsum(take, 0) - take
            within = torch.arange(total, device=dev) - off[rep]
            c = cnt[rep]
            rnd = (torch.rand(total, device=dev, generator=gen) * c).long().clamp_(max=c.max() - 1)
            pos = torch.where(c <= k, within, torch.minimum(rnd, c - 1))
            nbr = self.col[start[rep] + pos]
            dst = front[rep]
            pk = dst * n + nbr
            if excl is not None and len(excl):
                q = pair_key(dst, nbr, n)
                i = torch.searchsorted(excl, q).clamp_(max=len(excl) - 1)
                keep = excl[i] != q
                nbr, dst, pk = nbr[keep], dst[keep], pk[keep]
            pk, first = _unique_first(pk)
            nbr, dst = nbr[first], dst[first]
            srcs.append(nbr)
            dsts.append(dst)
            new = torch.unique(nbr[self.loc[nbr] < 0])
            self.loc[new] = torch.arange(count, count + len(new), device=dev)
            count += len(new)
            ids.append(new)
            front = new
        n_id = torch.cat(ids)
        if srcs:
            src, dst = torch.cat(srcs), torch.cat(dsts)
            ei = torch.stack([self.loc[src], self.loc[dst]])
        else:
            ei = torch.empty(2, 0, dtype=torch.long, device=dev)
        self.loc[n_id] = -1
        return Sub(n_id, ei, len(roots))

    def visible_edge_index(self, kv: int, exclude: Optional[Tensor] = None) -> Tensor:
        """Every visible edge (both directions), global ids, as ``[2, E]`` (src -> dst); for the exactness audit."""
        keep = self._key_sorted < kv
        row = torch.repeat_interleave(torch.arange(self.n, device=self.device), self.rowptr[1:] - self.rowptr[:-1])
        r, c = row[keep], self.col[keep]
        if exclude is not None and len(exclude):
            q = pair_key(r, c, self.n)
            ex = torch.sort(exclude.to(self.device)).values
            i = torch.searchsorted(ex, q).clamp_(max=len(ex) - 1)
            m = ex[i] != q
            r, c = r[m], c[m]
        return torch.stack([c, r])


def _unique_first(x: Tensor):
    """Unique values of ``x`` and the index of one occurrence of each."""
    u, inv = torch.unique(x, return_inverse=True)
    first = torch.full((len(u),), len(x), dtype=torch.long, device=x.device)
    first.scatter_reduce_(0, inv, torch.arange(len(x), device=x.device), reduce="amin")
    return u, first
