"""Individual drift signals and the running standardiser that makes them comparable."""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import Tensor


class RunningStandardiser:
    """Streaming z-score using EMA estimates of mean and variance.

    The three drift signals live on incomparable scales, so a weighted sum of
    the *raw* values is dominated by whichever signal happens to be largest.
    Standardising each signal first is what makes the proposal's
    ``D_total = a*D_emb + b*D_pred + c*D_att`` a meaningful combination.
    """

    def __init__(self, ema: float = 0.1, warmup: int = 20, eps: float = 1e-8) -> None:
        self.ema = ema
        self.warmup = warmup
        self.eps = eps
        self.mean = 0.0
        self.var = 1.0
        self.count = 0

    @property
    def ready(self) -> bool:
        return self.count >= self.warmup

    def update(self, value: float) -> float:
        """Record ``value`` and return its standardised score."""
        self.count += 1
        if self.count == 1:
            self.mean = value
            self.var = 0.0
            return 0.0

        delta = value - self.mean
        z = delta / (self.var**0.5 + self.eps) if self.ready else 0.0

        self.mean += self.ema * delta
        self.var = (1 - self.ema) * (self.var + self.ema * delta * delta)
        return float(z)


@dataclass
class SignalReading:
    raw: float
    score: float  # standardised


class EmbeddingDrift:
    """D_emb(t) = | E[||h(t)||] - E[||h(t-d)||] | via an EMA reference."""

    def __init__(self, ema: float = 0.1, warmup: int = 20) -> None:
        self.reference: float | None = None
        self.ema = ema
        self.norm = RunningStandardiser(ema=ema, warmup=warmup)

    @staticmethod
    def statistic(embeddings: Tensor) -> Tensor:
        """The per-step statistic, left on the device."""
        return embeddings.norm(dim=1).mean()

    def update(self, embeddings: Tensor) -> SignalReading:
        return self.update_stat(float(self.statistic(embeddings)))

    def update_stat(self, stat: float) -> SignalReading:
        if self.reference is None:
            self.reference = stat
            return SignalReading(0.0, 0.0)
        raw = abs(stat - self.reference)
        self.reference = self.ema * stat + (1 - self.ema) * self.reference
        return SignalReading(raw, self.norm.update(raw))


class PredictionDrift:
    """D_pred(t) = || p_t - p_{t-d} ||_2 over the class histogram."""

    def __init__(self, num_classes: int, ema: float = 0.1, warmup: int = 20) -> None:
        self.num_classes = num_classes
        self.reference: Tensor | None = None
        self.ema = ema
        self.norm = RunningStandardiser(ema=ema, warmup=warmup)

    def update(self, predictions: Tensor) -> SignalReading:
        raw = self.raw_on_device(predictions)
        return SignalReading(0.0, 0.0) if raw is None else self.update_raw(float(raw))

    def raw_on_device(self, predictions: Tensor) -> Tensor | None:
        """Advance the reference histogram and return the raw distance, without a host sync.

        ``None`` on the first call, which only sets the reference. The empty-batch
        case is a ``where`` rather than a Python ``if`` on a device scalar, which
        forced a synchronisation every step; the values are identical.
        """
        counts = torch.bincount(predictions, minlength=self.num_classes).float()
        total = counts.sum()
        probs = torch.where(total > 0, counts / total, counts)
        if self.reference is None:
            self.reference = probs
            return None
        raw = torch.norm(probs - self.reference)
        self.reference = self.ema * probs + (1 - self.ema) * self.reference
        return raw

    def update_raw(self, raw: float) -> SignalReading:
        return SignalReading(raw, self.norm.update(raw))


class AttentionDrift:
    """D_att(t) = | E[alpha(t)] - E[alpha(t-d)] | over GAT coefficients."""

    def __init__(self, ema: float = 0.1, warmup: int = 20) -> None:
        self.reference: float | None = None
        self.ema = ema
        self.norm = RunningStandardiser(ema=ema, warmup=warmup)

    @staticmethod
    def statistic(attention: tuple[Tensor, Tensor]) -> Tensor:
        """The per-step statistic, left on the device."""
        return attention[1].mean()

    def update(self, attention: tuple[Tensor, Tensor]) -> SignalReading:
        return self.update_stat(float(self.statistic(attention)))

    def update_stat(self, stat: float) -> SignalReading:
        if self.reference is None:
            self.reference = stat
            return SignalReading(0.0, 0.0)
        raw = abs(stat - self.reference)
        self.reference = self.ema * stat + (1 - self.ema) * self.reference
        return SignalReading(raw, self.norm.update(raw))
