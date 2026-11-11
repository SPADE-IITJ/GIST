"""Deterministic seeding utilities.

Phase-1 requirement: every stochastic decision in an experiment must be
reproducible *and* identical across adaptation arms, so that any measured
difference is attributable to the adaptation policy alone.
"""

from __future__ import annotations

import os
import random

import numpy as np
import torch


def seed_everything(seed: int) -> None:
    """Seed every RNG this project draws from."""
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed % (2**32))
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def derive(seed: int, *parts: int) -> int:
    """Derive a stable child seed from a parent seed and integer coordinates.

    Used to give each stream step its own sampling seed without coupling steps
    to one another through a shared advancing RNG stream.
    """
    h = int(seed) & 0xFFFFFFFF
    for part in parts:
        h = (h * 1000003 + int(part)) & 0xFFFFFFFF
    return h


def numpy_rng(seed: int) -> np.random.Generator:
    """An isolated numpy Generator (never the global legacy RandomState)."""
    return np.random.default_rng(seed)
