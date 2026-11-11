"""Composite multi-signal drift detector."""

from __future__ import annotations

from dataclasses import dataclass, asdict
from typing import Dict, Tuple

import torch
from torch import Tensor

from ..config import DriftConfig
from .signals import AttentionDrift, EmbeddingDrift, PredictionDrift, SignalReading


@dataclass
class DriftReading:
    """One step's drift state, with every term kept for ablation."""

    embedding_raw: float
    prediction_raw: float
    attention_raw: float
    embedding_z: float
    prediction_z: float
    attention_z: float
    total: float
    flag: bool

    def as_dict(self) -> Dict[str, float | bool]:
        return asdict(self)


_ZERO = SignalReading(0.0, 0.0)


class CompositeDriftDetector:
    """Implements D_total = a*z(D_emb) + b*z(D_pred) + c*z(D_att).

    The DASS-GNN prototype scored drift as ``(emb_raw + attn_raw) / 2`` and
    dropped the prediction signal from the score entirely, keeping it only as
    a boolean OR term. This restores all three signals to the composite and
    makes the weights explicit and configurable.
    """

    def __init__(self, num_classes: int, cfg: DriftConfig, use_attention: bool = True) -> None:
        self.cfg = cfg
        self.embedding = EmbeddingDrift(ema=cfg.ema, warmup=cfg.warmup)
        self.prediction = PredictionDrift(num_classes, ema=cfg.ema, warmup=cfg.warmup)
        self.attention = AttentionDrift(ema=cfg.ema, warmup=cfg.warmup)
        # Models without attention (SAGE, GCN, GIN) give the detector two
        # signals; the attention weight is dropped and the other two are
        # renormalised, so `total_threshold` still reads as sigmas. Its null
        # distribution changes, so it is recalibrated per backbone.
        self.use_attention = bool(use_attention)
        w_att = cfg.weight_attention if self.use_attention else 0.0

        total = cfg.weight_embedding + cfg.weight_prediction + w_att
        self._w_emb = cfg.weight_embedding / total
        self._w_pred = cfg.weight_prediction / total
        self._w_att = w_att / total

        # A weighted sum of unit-variance signals has std ||w||_2 (for roughly
        # uncorrelated signals), which is 0.577 for equal weights and 1.0 when
        # a single signal is used. Dividing by ||w||_2 restores unit variance,
        # so `total_threshold` reads as "sigmas" and stays comparable across
        # weight ablations instead of silently becoming stricter as weights
        # are spread over more signals.
        self._scale = (self._w_emb**2 + self._w_pred**2 + self._w_att**2) ** 0.5 or 1.0
        self.steps = 0

    @property
    def ready(self) -> bool:
        return self.steps > self.cfg.warmup

    def update(
        self,
        embeddings: Tensor,
        predictions: Tensor,
        attention: Tuple[Tensor, Tensor] | None,
    ) -> DriftReading:
        self.steps += 1
        # All three statistics stay on the device and cross to the host in one
        # transfer. Reading each with float() cost ~6 host synchronisations per
        # step (G1, M5); the arithmetic is unchanged.
        attend = self.use_attention and attention is not None
        stats = [self.embedding.statistic(embeddings)]
        if attend:
            stats.append(self.attention.statistic(attention))
        pred_raw = self.prediction.raw_on_device(predictions)
        if pred_raw is not None:
            stats.append(pred_raw)
        host = torch.stack(stats).tolist()
        emb = self.embedding.update_stat(host[0])
        att = self.attention.update_stat(host[1]) if attend else _ZERO
        pred = self.prediction.update_raw(host[-1]) if pred_raw is not None else _ZERO

        weighted = self._w_emb * emb.score + self._w_pred * pred.score + self._w_att * att.score
        total = weighted / self._scale
        flag = bool(self.ready and total > self.cfg.total_threshold)

        return DriftReading(
            embedding_raw=emb.raw,
            prediction_raw=pred.raw,
            attention_raw=att.raw,
            embedding_z=emb.score,
            prediction_z=pred.score,
            attention_z=att.score,
            total=float(total),
            flag=flag,
        )
