"""Adaptation policies.

Three arms:

``none``       never adapts during the stream (baseline).
``full``       one gradient step over every trainable root in the batch.
``periodic``   the same, on a fixed schedule, ignoring drift (control arm).
``selective``  one gradient step over the drifted region of the graph.

``selective`` supports three modes. ``subgraph`` is the Phase-2 contribution:
it extracts the affected communities' induced subgraph and runs forward and
backward over that alone, so cost scales with the drifted region rather than
with the batch. ``reweight`` and ``mask`` reproduce Phase 1 / DASS-GNN, where
the backward pass still spanned the whole batch and localization bought no
compute saving at all.
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import List, Optional

import torch
import torch.nn.functional as F
from torch import Tensor

from .subgraph import audit_exactness, full_batch_stats, subgraph_gradient_step


@dataclass
class RetrainContext:
    """Everything a policy may need to decide and perform an update."""

    step: int
    batch: object  # the current inference batch
    train_roots: Tensor  # trainable roots within that batch
    community: object = None
    backend: object = None
    sampler: object = None  # samples a fresh batch around given seed nodes
    train_mask: Tensor = None  # graph-wide trainable node mask
    seed: int = 0
    oracle_nodes: Tensor = None  # adaptation.oracle_where: nodes hit by injected events since the last update
    adapter: object = None  # adaptation.region_adapter: RegionAdapter applied to the logits (None: off)


@dataclass
class RetrainInfo:
    step: int
    loss: float
    nodes_targeted: int  # nodes contributing to the loss (the "localized" set)
    nodes_updated: int  # nodes with any non-zero loss weight
    nodes_in_backward: int  # nodes materialised in the forward/backward pass
    edges_in_backward: int  # edges message-passed over
    communities_selected: int
    communities_total: int
    # Seconds spent sampling the region batch, for the selective arm only.
    # This is the overhead localization pays to buy a smaller backward pass:
    # `full` reuses the inference batch already sampled for the step and
    # pays nothing here. Separating it from the gradient step is what makes
    # the wall-clock comparison explicable rather than just observed.
    region_sample_s: float = 0.0
    # Float64 exactness audit of this event's update (selective, first k events).
    audit: Optional[dict] = None
    # Uniform roots the iso_full arm drew to match its node target.
    roots_drawn: int = 0


def _weighted_gradient_step(
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    batch,
    weights: Tensor,
    adapter=None,
) -> Optional[tuple[float, int]]:
    """One optimizer step on a weighted mean of the per-root losses."""
    if float(weights.sum()) <= 0:
        return None

    model.train()
    optimizer.zero_grad()
    out, _, _ = model(batch.x, batch.edge_index)

    root_out = out[: batch.batch_size]
    root_y = batch.y[: batch.batch_size]
    if root_y.numel() == 0:
        return None
    base = root_out
    if adapter is not None:
        root_ids = batch.n_id[: batch.batch_size]
        root_out = adapter.apply(root_out, root_ids)

    # Weights are built from CPU-resident masks (train_mask indexed by root
    # ids); the per-node loss lives on the model's device. This path is only
    # reached with supervise_halo off, so it went unexercised on GPU.
    weights = weights.to(root_out.device)

    per_node = F.cross_entropy(root_out, root_y, reduction="none")
    # Proper weighted mean. The prototype used ``(loss * w).mean()``, which
    # additionally scales the gradient by the mean weight and so silently
    # shrinks the effective learning rate as more nodes are down-weighted.
    loss = (per_node * weights).sum() / weights.sum()

    loss.backward()
    optimizer.step()
    if adapter is not None:
        keep = weights > 0
        adapter.update(base[keep].detach(), root_y[keep], root_ids.to(keep.device)[keep])
    return float(loss.item()), int((weights > 0).sum())


def _full_batch_gradient_step(
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    batch,
    train_mask: Tensor,
    adapter=None,
) -> Optional[tuple[float, int]]:
    """Gradient step supervising every trainable node in the batch.

    The forward pass materialises activations for all sampled nodes anyway, so
    supervising the non-root trainable ones is free - the same accounting that
    motivates halo rehearsal in the selective arm.
    """
    model.train()
    optimizer.zero_grad()
    out, _, _ = model(batch.x, batch.edge_index)

    # train_mask is CPU-resident; the loss indexes a device tensor.
    trainable = train_mask[batch.n_id.cpu()].to(out.device)
    if not bool(trainable.any()):
        return None

    if adapter is None:
        loss = F.cross_entropy(out[trainable], batch.y[trainable])
    else:
        ids = batch.n_id.to(trainable.device)[trainable]
        loss = F.cross_entropy(adapter.apply(out[trainable], ids), batch.y[trainable])
    loss.backward()
    optimizer.step()
    if adapter is not None:
        adapter.update(out[trainable].detach(), batch.y[trainable], ids)
    return float(loss.item()), int(trainable.sum())


class Retrainer:
    name = "none"
    requires_community = False

    def should_fire(self, step: int, drift_signal: bool, controller) -> bool:
        """Whether to attempt a retraining step now."""
        return controller.should_retrain(drift_signal, step)

    def apply(self, model, optimizer, ctx: RetrainContext) -> Optional[RetrainInfo]:
        return None


class NoRetrainer(Retrainer):
    name = "none"

    def should_fire(self, step: int, drift_signal: bool, controller) -> bool:
        return False


class FullRetrainer(Retrainer):
    """Retrain over the whole sampled batch.

    ``supervise_halo`` gives this arm the *same* free supervision the selective
    arm gets: loss over every trainable node the forward pass already
    materialised, not just the batch roots. Without it the comparison is
    unfair in selective's favour - selective would be supervising ~4x more
    nodes, and any gain could be attributed to extra supervision rather than
    to locality. With it, the only difference between the arms is *where* the
    materialised nodes are and therefore how many there are.
    """

    name = "full"
    requires_community = False

    def __init__(self, supervise_halo: bool = True, max_nodes: int = 0, target_nodes: int = 0) -> None:
        self.supervise_halo = supervise_halo
        # iso_full: choose the number of uniform roots per event so the sampled
        # batch has about this many nodes - selective's backward size - instead
        # of a fixed root count. Matched cost, not matched roots.
        self.target_nodes = int(target_nodes)
        self._last_roots = 0
        # When set, retrain on a freshly sampled batch of this many roots
        # drawn uniformly from the trainable pool, instead of on the step's
        # inference batch.
        #
        # This exists so the budget can be matched. `selective` draws
        # `max_retrain_nodes` roots from the drifted communities, while this
        # arm otherwise inherits whatever `stream.batch_size` happens to be,
        # so raising the budget would enlarge only selective's update. With
        # both drawing the same number of roots, the arms differ in *where*
        # the nodes sit and in nothing else - which is the whole claim.
        self.max_nodes = int(max_nodes)
        self._last_region_sample_s = 0.0

    def _global_batch(self, ctx: RetrainContext):
        """A fresh uniform sample of `max_nodes` trainable roots."""
        if self.max_nodes <= 0 or ctx.sampler is None or ctx.train_mask is None:
            return None
        pool = ctx.train_mask.nonzero(as_tuple=True)[0]
        if pool.numel() == 0:
            return None
        generator = torch.Generator().manual_seed(ctx.seed + ctx.step)
        if self.target_nodes > 0:
            return self._iso_batch(ctx, pool[torch.randperm(pool.numel(), generator=generator)])
        pick = torch.randperm(pool.numel(), generator=generator)[: self.max_nodes]
        nodes = pool[pick]
        self._last_roots = int(nodes.numel())
        start = time.perf_counter()
        batch = ctx.sampler.sample(nodes, ctx.seed + ctx.step, ctx.step)
        self._last_region_sample_s = time.perf_counter() - start
        return batch

    def _iso_batch(self, ctx: RetrainContext, order: Tensor):
        """Bisect the number of leading roots of ``order`` whose sampled batch is closest to ``target_nodes``."""
        sample = lambda r: ctx.sampler.sample(order[:r], ctx.seed + ctx.step, ctx.step)
        lo, hi = 1, int(min(order.numel(), max(self.max_nodes, 1) * 4))
        best = None
        start = time.perf_counter()
        for _ in range(12):
            if lo > hi:
                break
            mid = (lo + hi) // 2
            b = sample(mid)
            gap = abs(int(b.num_nodes) - self.target_nodes)
            if best is None or gap < best[0]:
                best = (gap, mid, b)
            if int(b.num_nodes) < self.target_nodes:
                lo = mid + 1
            else:
                hi = mid - 1
        self._last_region_sample_s = time.perf_counter() - start
        self._last_roots = best[1]
        return best[2]

    def apply(self, model, optimizer, ctx: RetrainContext) -> Optional[RetrainInfo]:
        batch, step = ctx.batch, ctx.step

        self._last_region_sample_s = 0.0
        if self.max_nodes > 0:
            sampled = self._global_batch(ctx)
            if sampled is not None:
                # Sampled inside the retrainer, so it never passed through
                # the main loop's batch.to(device).
                batch = sampled.to(next(model.parameters()).device)

        if self.supervise_halo and ctx.train_mask is not None and hasattr(batch, "n_id"):
            result = _full_batch_gradient_step(model, optimizer, batch, ctx.train_mask, adapter=ctx.adapter)
        else:
            roots = ctx.train_roots
            if batch is not ctx.batch:
                # Resampled batch: every root was drawn from the trainable
                # pool by construction, so ctx.train_roots no longer applies.
                roots = torch.ones(int(batch.batch_size), dtype=torch.bool)
            result = _weighted_gradient_step(model, optimizer, batch, roots.float(), adapter=ctx.adapter)
        if result is None:
            return None
        loss, updated = result
        # Charge every sampled node, not just the roots: the forward pass
        # materialises activations for the whole subgraph. Phase 1 counted
        # only roots, which understated full retraining's cost.
        stats = full_batch_stats(batch)
        return RetrainInfo(
            step=step,
            loss=loss,
            nodes_targeted=updated,
            nodes_updated=updated,
            nodes_in_backward=stats.nodes_in_backward,
            edges_in_backward=stats.edges_in_backward,
            communities_selected=0,
            communities_total=0,
            region_sample_s=self._last_region_sample_s,
            roots_drawn=self._last_roots,
        )


class PeriodicRetrainer(FullRetrainer):
    """Retrain on a fixed schedule, ignoring the drift signal entirely.

    This is the control that makes the paper's central claim falsifiable.
    "Drift-aware adaptation beats no adaptation" is a weak claim - so does
    retraining blindly on a timer. The claim that matters is that *triggering
    on drift* beats spending the same retraining budget on a fixed schedule.
    Without this arm, a saturated drift gate is indistinguishable from a
    working one, which is exactly the failure that the first Phase-1 run hit.
    """

    name = "periodic"
    requires_community = False

    def __init__(self, period: int, supervise_halo: bool = True,
                 max_nodes: int = 0) -> None:
        super().__init__(supervise_halo=supervise_halo, max_nodes=max_nodes)
        self.period = max(1, int(period))

    def should_fire(self, step: int, drift_signal: bool, controller) -> bool:
        return step > 0 and step % self.period == 0


class SelectiveRetrainer(Retrainer):
    """Community-localized retraining.

    ``subgraph`` mode is the Phase-2 contribution: the update is restricted to
    the induced subgraph the target nodes actually depend on, so backward-pass
    cost scales with the drifted region rather than with the batch. It is
    exact - see gist/adaptation/subgraph.py.

    ``reweight`` and ``mask`` are retained to reproduce Phase 1 / DASS-GNN
    behaviour, where the backward pass still spanned the whole batch.
    """

    name = "selective"
    requires_community = True

    def __init__(
        self,
        backend,
        mode: str = "subgraph",
        background_weight: float = 0.1,
        num_hops: int = 2,
        max_nodes: int = 128,
        anchor_fraction: float = 0.25,
        halo_rehearsal: bool = True,
        random_region: bool = False,
        audit_events: int = 0,
    ) -> None:
        self.backend = backend
        # random_region: the C3 control. Same detector, budget, anchor and
        # mechanics, but the communities are drawn at random (as many as the
        # drift score would have picked) instead of by drift score.
        self.random_region = bool(random_region)
        self.audit_left = int(audit_events)
        self.mode = mode
        self.background_weight = background_weight
        self.num_hops = num_hops
        self.max_nodes = max_nodes
        self._last_region_sample_s = 0.0
        self.anchor_fraction = float(anchor_fraction)
        self.halo_rehearsal = bool(halo_rehearsal)

    def _region_batch(self, ctx: RetrainContext, drifting: List[int]):
        """Sample the drifted region of the *graph*, plus a rehearsal anchor.

        Two failures found by measurement, in order:

        1. Restricting the update to drifting nodes that happen to sit in the
           current inference batch yields ~2-3 nodes per event. An exact
           gradient on 2-3 nodes wrecks the model (-5.4pp vs no adaptation).
           The affected region is a property of the graph, not of the batch,
           so targets are drawn from it directly - which is also what the
           proposal's "extract only the affected subgraphs" describes.

        2. Drawing *only* from the drifted communities still lost 3.8pp,
           because the same ~110 nodes are revisited every event while `full`
           sees 128 fresh random roots. The model overfits the drifted region
           and forgets the rest - catastrophic forgetting, the problem ER-GNN
           addresses with memory replay.

        So a fraction of each retraining set is drawn uniformly from the whole
        trainable pool as rehearsal. DASS-GNN's loss reweighting was an
        accidental version of this: holding every train root at weight 0.1
        preserved global coverage. Making it explicit keeps that protection
        while retaining the cost saving, since the anchor is a minority of a
        set that is itself far smaller than a full batch.
        """
        if ctx.sampler is None or ctx.train_mask is None:
            return None

        generator = torch.Generator().manual_seed(ctx.seed + ctx.step)
        n_anchor = int(round(self.max_nodes * self.anchor_fraction))
        n_target = max(1, self.max_nodes - n_anchor)

        if ctx.oracle_nodes is not None and not self.random_region:
            # Diagnostic ceiling (adaptation.oracle_where): the truly drifted trainable nodes stand in for the region.
            pool = ctx.oracle_nodes[ctx.train_mask[ctx.oracle_nodes]]
            if pool.numel() > n_target:
                pool = pool[torch.randperm(pool.numel(), generator=generator)[:n_target]]
            nodes = pool
        else:
            nodes = self.backend.community_nodes(
                drifting, restrict_mask=ctx.train_mask, limit=n_target, generator=generator
            )
        if nodes is None:
            return None

        if n_anchor > 0:
            pool = ctx.train_mask.nonzero(as_tuple=True)[0]
            if pool.numel() > 0:
                pick = torch.randperm(pool.numel(), generator=generator)[:n_anchor]
                nodes = torch.unique(torch.cat([nodes, pool[pick]]))

        if nodes.numel() == 0:
            return None
        # Timed separately: this sample is the price of localization. It is
        # CPU-bound and does not shrink as the model grows, so it sets the
        # floor the GNN saving has to clear before selective can win.
        start = time.perf_counter()
        region = ctx.sampler.sample(nodes, ctx.seed + ctx.step, ctx.step)
        self._last_region_sample_s = time.perf_counter() - start
        return region

    def apply(self, model, optimizer, ctx: RetrainContext) -> Optional[RetrainInfo]:
        batch, step, community = ctx.batch, ctx.step, ctx.community
        train_roots = ctx.train_roots
        if community is None:
            return None

        drifting = self.backend.drifting(community)
        communities_total = getattr(community, "num_communities", 0)
        if self.random_region and drifting:
            ids = getattr(getattr(self.backend, "tracker", None), "partition", None)
            ids = sorted(ids.sizes) if ids is not None else []
            if ids:
                g = torch.Generator().manual_seed(ctx.seed * 1_000_003 + ctx.step + 0xA5)
                pick = torch.randperm(len(ids), generator=g)[: len(drifting)].tolist()
                drifting = [ids[i] for i in pick]

        if self.mode == "subgraph":
            self._last_region_sample_s = 0.0
            audit = None
            region = self._region_batch(ctx, drifting)
            if region is None:
                # No graph-wide membership available (batch backend): fall
                # back to the drifting roots of the current batch.
                fallback = self.backend.root_mask(community, drifting, batch) & train_roots
                if not bool(fallback.any()):
                    return None
                outcome = subgraph_gradient_step(
                    model, optimizer, batch, fallback, self.num_hops,
                    halo_train_mask=ctx.train_mask if self.halo_rehearsal else None, adapter=ctx.adapter,
                )
            else:
                # Every root of the region batch is a target by construction,
                # so its 2-hop expansion *is* the induced subgraph we need.
                targets = torch.ones(region.batch_size, dtype=torch.bool)
                if self.audit_left > 0 and not self.halo_rehearsal:
                    # Parent = the whole current graph the sampler draws from (features and
                    # labels as drifted so far), not the region batch: that batch already is
                    # the targets' sampled neighbourhood, so comparing against it is vacuous.
                    self.audit_left -= 1
                    g = ctx.sampler.data
                    audit = audit_exactness(model, g.x, g.y, g.edge_index, int(g.num_nodes),
                                            region.n_id[: region.batch_size], self.num_hops, adapter=ctx.adapter)
                outcome = subgraph_gradient_step(
                    model, optimizer, region, targets, self.num_hops,
                    halo_train_mask=ctx.train_mask if self.halo_rehearsal else None, adapter=ctx.adapter,
                )

            if outcome is None:
                return None
            loss, stats = outcome
            return RetrainInfo(
                step=step,
                loss=loss,
                nodes_targeted=stats.nodes_targeted,
                nodes_updated=stats.nodes_targeted,
                nodes_in_backward=stats.nodes_in_backward,
                edges_in_backward=stats.edges_in_backward,
                communities_selected=len(drifting),
                communities_total=communities_total,
                region_sample_s=self._last_region_sample_s,
                audit=audit,
            )

        targets = self.backend.root_mask(community, drifting, batch) & train_roots
        if not bool(targets.any()):
            return None

        if self.mode == "mask":
            weights = targets.float()
        else:  # "reweight"
            weights = train_roots.float() * self.background_weight
            weights[targets] = 1.0

        result = _weighted_gradient_step(model, optimizer, batch, weights, adapter=ctx.adapter)
        if result is None:
            return None
        loss, updated = result
        full = full_batch_stats(batch)
        return RetrainInfo(
            step=step,
            loss=loss,
            nodes_targeted=int(targets.sum()),
            nodes_updated=updated,
            # Unchanged from `full`: the backward pass still spans the batch.
            nodes_in_backward=full.nodes_in_backward,
            edges_in_backward=full.edges_in_backward,
            communities_selected=len(drifting),
            communities_total=communities_total,
        )


def build_retrainer(arm: str, community_backend, adaptation_cfg, num_hops: int = 2) -> Retrainer:
    if arm == "baseline":
        return NoRetrainer()
    if arm == "full":
        return FullRetrainer(
            supervise_halo=adaptation_cfg.supervise_halo,
            max_nodes=adaptation_cfg.full_retrain_nodes,
        )
    if arm == "periodic":
        return PeriodicRetrainer(
            period=adaptation_cfg.periodic_period,
            supervise_halo=adaptation_cfg.supervise_halo,
            max_nodes=adaptation_cfg.full_retrain_nodes,
        )
    if arm == "iso_full":
        return FullRetrainer(
            supervise_halo=adaptation_cfg.supervise_halo,
            max_nodes=adaptation_cfg.full_retrain_nodes,
            target_nodes=adaptation_cfg.iso_target_nodes,
        )
    if arm in ("selective", "random_region"):
        return SelectiveRetrainer(
            backend=community_backend,
            mode=adaptation_cfg.selective_mode,
            background_weight=adaptation_cfg.background_weight,
            num_hops=num_hops,
            max_nodes=adaptation_cfg.max_retrain_nodes,
            anchor_fraction=adaptation_cfg.anchor_fraction,
            halo_rehearsal=adaptation_cfg.halo_rehearsal,
            random_region=arm == "random_region",
            audit_events=adaptation_cfg.exactness_audit if arm == "selective" else 0,
        )
    raise ValueError(f"Unknown arm {arm!r}")
