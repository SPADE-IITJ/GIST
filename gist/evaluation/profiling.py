"""Phase 3: HPC instrumentation - latency, throughput, memory, FLOPs.

The proposal's §5 promises "batch processing latency, event streaming
throughput, and recovery adaptivity" and claims bounded backward-pass memory
and fewer full-graph kernel invocations. None of that was measurable before:
the prototype shipped an empty ``evaluation/metrics.py`` and contained no
timing code at all.

Two kinds of measurement live here, and they are not interchangeable:

* **Measured** - wall-clock and allocator peaks. Ground truth, but at Cora's
  scale (~700-node batches) framework overhead dominates real compute, so
  wall-clock differences between arms are mostly noise. Honest reporting of
  these numbers requires saying so.
* **Analytic** - FLOPs and activation bytes derived from the subgraph the
  arm actually materialised. Hardware-independent, noise-free, and the
  quantity the scalability claim is really about. These are the numbers that
  transfer to a larger graph.

Both are reported. The analytic ones carry the argument; the measured ones
keep it honest.
"""

from __future__ import annotations

import time
import tracemalloc
from contextlib import contextmanager
from dataclasses import dataclass, field
from typing import Dict, List, Optional

import numpy as np
import torch


def resolve_device(preference: str = "auto") -> torch.device:
    if preference == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(preference)


# --- analytic cost model ------------------------------------------------


def gat_layer_flops(nodes: int, edges: int, in_dim: int, out_dim: int, heads: int) -> int:
    """Forward FLOPs for one GATConv layer.

    Three terms: the per-node linear projection, the per-edge attention
    coefficients, and the per-edge weighted aggregation. Multiply-accumulate
    counted as 2 FLOPs.
    """
    projection = 2 * nodes * in_dim * out_dim * heads
    attention = 2 * edges * out_dim * heads
    aggregation = 2 * edges * out_dim * heads
    return projection + attention + aggregation


def model_flops(
    nodes: int,
    edges: int,
    in_dim: int,
    hidden: int,
    out_dim: int,
    heads: int,
    *,
    backward: bool = True,
) -> int:
    """Forward (and optionally backward) FLOPs for the two-layer GAT.

    The backward pass is charged at 2x forward, the standard approximation:
    one pass for input gradients and one for parameter gradients.
    """
    total = gat_layer_flops(nodes, edges, in_dim, hidden, heads)
    total += gat_layer_flops(nodes, edges, hidden * heads, out_dim, 1)
    return total * 3 if backward else total


def activation_bytes(
    nodes: int, hidden: int, out_dim: int, heads: int, dtype_bytes: int = 4
) -> int:
    """Activation memory the backward pass must retain.

    This is the quantity the proposal's "bounds the memory footprint required
    for the backward pass" refers to, and it scales with the *materialised*
    subgraph - which is exactly what localized retraining shrinks.
    """
    layer1 = nodes * hidden * heads
    layer2 = nodes * out_dim
    return (layer1 + layer2) * dtype_bytes


# --- measured -----------------------------------------------------------


class MemoryProbe:
    """Peak memory during a region, on whichever device is in use.

    CUDA reports allocator peaks, which are precise. On CPU there is no
    equivalent, so this falls back to ``tracemalloc``, which only sees Python
    allocations and misses most of what ATen does - it is a weak proxy and is
    labelled as such rather than quietly reported as if it were comparable.
    """

    def __init__(self, device: torch.device) -> None:
        self.device = device
        self.cuda = device.type == "cuda"
        self.peaks: List[int] = []

    @contextmanager
    def measure(self):
        if self.cuda:
            torch.cuda.reset_peak_memory_stats(self.device)
            torch.cuda.synchronize(self.device)
            try:
                yield
            finally:
                torch.cuda.synchronize(self.device)
                self.peaks.append(int(torch.cuda.max_memory_allocated(self.device)))
        else:
            started = not tracemalloc.is_tracing()
            if started:
                tracemalloc.start()
            before = tracemalloc.get_traced_memory()[1]
            try:
                yield
            finally:
                peak = tracemalloc.get_traced_memory()[1]
                self.peaks.append(int(max(peak - before, 0)))
                if started:
                    tracemalloc.stop()

    def summary(self) -> Dict[str, float]:
        if not self.peaks:
            return {"peak_bytes_mean": 0.0, "peak_bytes_max": 0.0, "measured": self.cuda}
        arr = np.asarray(self.peaks, dtype=float)
        return {
            "peak_bytes_mean": float(arr.mean()),
            "peak_bytes_max": float(arr.max()),
            # False means the numbers above are a weak CPU proxy, not
            # allocator ground truth.
            "measured": self.cuda,
        }


@dataclass
class LatencyStats:
    """Per-stage latency with tails, not just means.

    A streaming system is judged on its tail: a p99 spike is a stall. Means
    hide exactly the retraining spikes this work is about.
    """

    samples: Dict[str, List[float]] = field(default_factory=dict)

    def record(self, stage: str, seconds: float) -> None:
        self.samples.setdefault(stage, []).append(seconds)

    def summary(self) -> Dict[str, Dict[str, float]]:
        out: Dict[str, Dict[str, float]] = {}
        for stage, values in sorted(self.samples.items()):
            arr = np.asarray(values, dtype=float) * 1000.0
            if arr.size == 0:
                continue
            out[stage] = {
                "calls": int(arr.size),
                "total_s": float(arr.sum() / 1000.0),
                "mean_ms": float(arr.mean()),
                "p50_ms": float(np.percentile(arr, 50)),
                "p95_ms": float(np.percentile(arr, 95)),
                "p99_ms": float(np.percentile(arr, 99)),
                "max_ms": float(arr.max()),
            }
        return out


class StreamProfiler:
    """Stage latency, end-to-end throughput, memory, and analytic cost."""

    def __init__(self, device: torch.device, sync_stages: bool = True) -> None:
        self.device = device
        self.latency = LatencyStats()
        self.memory = MemoryProbe(device)
        self.flops_total = 0
        self.activation_bytes_peak = 0
        self.events = 0
        self._wall_start: Optional[float] = None
        self._wall_total = 0.0
        # CUDA kernels are asynchronous: without a synchronise, a stage timer
        # measures how long it took to *launch* the work, not to do it. The
        # GPU stages (infer, retrain) would read near-zero and the cost would
        # be misattributed to whichever later stage happened to block.
        #
        # Synchronising also serialises CPU and GPU, removing the cross-step
        # overlap a real streaming system enjoys, so it inflates end-to-end
        # wall-clock. Stage attribution and end-to-end throughput therefore
        # cannot be measured in the same run; `sync_stages` picks which one
        # this run is measuring. The end-to-end timer in `run()` always
        # synchronises at its boundaries, so `wall_s` is correct either way.
        self.sync_stages = sync_stages
        self._cuda = device.type == "cuda"
        self._sync = self._cuda and sync_stages

    @contextmanager
    def stage(self, name: str):
        if self._sync:
            torch.cuda.synchronize(self.device)
        start = time.perf_counter()
        try:
            yield
        finally:
            if self._sync:
                torch.cuda.synchronize(self.device)
            self.latency.record(name, time.perf_counter() - start)

    @contextmanager
    def run(self):
        if self._cuda:
            torch.cuda.synchronize(self.device)
        self._wall_start = time.perf_counter()
        try:
            yield
        finally:
            if self._cuda:
                torch.cuda.synchronize(self.device)
            self._wall_total = time.perf_counter() - self._wall_start

    def record_event(self) -> None:
        self.events += 1

    def record_cost(self, flops: int, act_bytes: int) -> None:
        self.flops_total += int(flops)
        self.activation_bytes_peak = max(self.activation_bytes_peak, int(act_bytes))

    def summary(self) -> Dict[str, object]:
        throughput = self.events / self._wall_total if self._wall_total > 0 else 0.0
        return {
            "device": str(self.device),
            # False means the per-stage breakdown misattributes async GPU
            # work and only `wall_s` / `events_per_s` should be quoted.
            "stage_times_synced": bool(self._sync or not self._cuda),
            "latency": self.latency.summary(),
            "memory": self.memory.summary(),
            "wall_s": self._wall_total,
            "events": self.events,
            "events_per_s": throughput,
            "retrain_gflops": self.flops_total / 1e9,
            "peak_activation_bytes": self.activation_bytes_peak,
        }
