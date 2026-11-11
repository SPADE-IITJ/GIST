"""Retraining gate: drift signal plus a cooldown."""

from __future__ import annotations


class RetrainingController:
    """Allows a retraining event when drift fires and the cooldown has elapsed.

    The prototype's controller counted its own internal invocations rather than
    stream steps, which made the cooldown drift out of sync with the stream
    whenever the controller was not consulted on every step. This version is
    driven by the actual step index.
    """

    def __init__(self, cooldown: int = 50) -> None:
        self.cooldown = cooldown
        self.last_retrain: int | None = None

    def should_retrain(self, drift: bool, step: int) -> bool:
        """Whether a retraining attempt is permitted at this step."""
        if not drift:
            return False
        if self.last_retrain is not None and (step - self.last_retrain) <= self.cooldown:
            return False
        return True

    def commit(self, step: int) -> None:
        """Start the cooldown. Call only when a gradient step actually happened.

        Keeping this separate from ``should_retrain`` matters: a selective
        policy can be permitted to retrain and then find no drifting community
        to act on. Arming the cooldown on that no-op would suppress the *next*
        genuine drift event for a full cooldown window.
        """
        self.last_retrain = step
