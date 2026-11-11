"""Region adapters: a per-community correction to the backbone's output, owned by the runtime.

Why: one gradient step on the shared weights barely
repairs a drifted region, and fitting that region harder makes the model forget the others, because the same weights
serve every region. A label remap or a local covariate shift belongs to one region, so each community gets parameters
of its own:

    logits'(v) = logits(v) + W_c logits(v) + b_c,   c = community of v   (unchanged for nodes with no community)

* **Zero-initialised**: a warm model's predictions are identical until an update touches a community.
* **Per node, after message passing**: the L-hop dependence, and with it exactness of the subgraph update (C4), is
  unchanged; ``audit_exactness`` composes the adapter when one is given.
* **Updated row by row**: only the communities of the current update's targets move. Rows without a gradient get no
  momentum and no decay, so an update in one region leaves every other community's predictions exactly unchanged.
* **Runtime-owned**: keyed by the community tracker's stable ids; the backbone is not modified (C1).
"""
from __future__ import annotations

from typing import Callable, Optional

import torch
import torch.nn.functional as F
from torch import Tensor


class RegionAdapter:
    def __init__(self, num_classes: int, assignment: Callable[[], Optional[Tensor]], lr: float = 1e-2,
                 l2: float = 1e-3, steps: int = 5, capacity: int = 64, betas=(0.9, 0.999), eps: float = 1e-8) -> None:
        self.num_classes = int(num_classes)
        self.assignment = assignment          # () -> [N] long tensor of community ids (-1: none), CPU
        self.lr, self.l2, self.steps = float(lr), float(l2), int(steps)
        self.betas, self.eps = betas, float(eps)
        self.device = torch.device("cpu")
        self._alloc(int(capacity))

    # -- storage --------------------------------------------------------------------------------------------------
    def _alloc(self, capacity: int, old: Optional[dict] = None) -> None:
        c = self.num_classes
        fresh = dict(W=torch.zeros(capacity, c, c), b=torch.zeros(capacity, c), mW=torch.zeros(capacity, c, c),
                     vW=torch.zeros(capacity, c, c), mb=torch.zeros(capacity, c), vb=torch.zeros(capacity, c),
                     t=torch.zeros(capacity))
        fresh = {k: v.to(self.device) for k, v in fresh.items()}
        if old is not None:
            n = old["t"].numel()
            for k in fresh:
                fresh[k][:n] = old[k]
        self.W = fresh["W"].requires_grad_(True)
        self.b = fresh["b"].requires_grad_(True)
        self.mW, self.vW, self.mb, self.vb, self.t = fresh["mW"], fresh["vW"], fresh["mb"], fresh["vb"], fresh["t"]

    def _state(self) -> dict:
        return dict(W=self.W.detach(), b=self.b.detach(), mW=self.mW, vW=self.vW, mb=self.mb, vb=self.vb, t=self.t)

    def to(self, device) -> "RegionAdapter":
        self.device = torch.device(device)
        state = {k: v.to(self.device) for k, v in self._state().items()}
        self._alloc(state["t"].numel(), state)
        return self

    @property
    def capacity(self) -> int:
        return int(self.t.numel())

    def community_ids(self, node_ids: Tensor) -> Tensor:
        """Community id of each node (-1 where none), on the adapter's device; grows storage for new ids."""
        assign = self.assignment()
        if assign is None or assign.numel() == 0:
            return torch.full((node_ids.numel(),), -1, dtype=torch.long, device=self.device)
        cids = assign[node_ids.detach().cpu().long()]
        top = int(cids.max()) if cids.numel() else -1
        if top >= self.capacity:
            self._alloc(max(2 * self.capacity, top + 1), self._state())
        return cids.to(self.device)

    # -- forward --------------------------------------------------------------------------------------------------
    def apply(self, logits: Tensor, node_ids: Tensor, W: Optional[Tensor] = None, b: Optional[Tensor] = None) -> Tensor:
        """Adapted logits for the nodes ``node_ids`` (global ids, one per row of ``logits``)."""
        cids = self.community_ids(node_ids).to(logits.device)
        W = self.W if W is None else W
        b = self.b if b is None else b
        has = (cids >= 0).to(logits.dtype).unsqueeze(1)
        idx = cids.clamp(min=0)
        delta = torch.bmm(W[idx].to(logits.dtype), logits.unsqueeze(2)).squeeze(2) + b[idx].to(logits.dtype)
        return logits + has * delta

    # -- update ---------------------------------------------------------------------------------------------------
    @torch.no_grad()
    def step(self) -> int:
        """Row-wise Adam on the rows that received a gradient, with L2 toward zero; returns how many moved."""
        gW, gb = self.W.grad, self.b.grad
        if gW is None and gb is None:
            return 0
        gW = torch.zeros_like(self.W) if gW is None else gW
        gb = torch.zeros_like(self.b) if gb is None else gb
        rows = (gW.flatten(1).abs().sum(1) + gb.abs().sum(1)) > 0
        if bool(rows.any()):
            b1, b2 = self.betas
            self.t[rows] += 1
            t = self.t[rows]
            for p, g, m, v, shape in ((self.W, gW, self.mW, self.vW, (-1, 1, 1)), (self.b, gb, self.mb, self.vb, (-1, 1))):
                grad = g[rows] + self.l2 * p[rows]
                m[rows] = b1 * m[rows] + (1 - b1) * grad
                v[rows] = b2 * v[rows] + (1 - b2) * grad * grad
                mhat = m[rows] / (1 - b1 ** t).view(shape)
                vhat = v[rows] / (1 - b2 ** t).view(shape)
                p[rows] -= self.lr * mhat / (vhat.sqrt() + self.eps)
        self.W.grad = None
        self.b.grad = None
        return int(rows.sum())

    def refit(self, logits: Tensor, labels: Tensor, node_ids: Tensor) -> None:
        """``steps`` adapter-only steps on fixed backbone logits of the update's targets (no backbone pass)."""
        logits = logits.detach()
        for _ in range(self.steps):
            loss = F.cross_entropy(self.apply(logits, node_ids), labels)
            loss.backward()
            self.step()

    def update(self, base_logits: Tensor, labels: Tensor, node_ids: Tensor) -> None:
        """After the backbone step: apply the adapter gradient it produced, then refit on the same targets."""
        self.step()
        self.refit(base_logits, labels, node_ids)
