# -*- coding: utf-8 -*-
from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Callable, List

from fedclypse.contribution import Contribution


class Synchronizer(ABC):
    """A synchronization policy: the aggregation *trigger*. It decides when a
    round fires (``ready``) and how a contribution is weighted by its staleness
    (``weight``). Orthogonal to topology and to the aggregation algorithm."""

    @abstractmethod
    def ready(self, collected: List[Contribution], cohort: List[str]) -> bool: ...

    def weight(self, contribution: Contribution, current_round: int) -> float:
        return 1.0


class Synchronous(Synchronizer):
    """Barrier: fire once every selected member of the cohort has reported."""

    def ready(self, collected: List[Contribution], cohort: List[str]) -> bool:
        return len(collected) >= len(cohort)


class BufferedAsync(Synchronizer):
    """Fire every time ``k`` contributions accumulate, regardless of sender.

    Staleness weighting is intentionally out of scope for this policy: it inherits
    the flat ``weight() -> 1.0``. A staleness-weighted buffered-async variant
    (e.g. FedBuff-style) is a future extension.
    """

    def __init__(self, k: int) -> None:
        if k < 1:
            raise ValueError("BufferedAsync requires k >= 1")
        self.k = k

    def ready(self, collected: List[Contribution], cohort: List[str]) -> bool:
        return len(collected) >= self.k


def inverse_staleness(staleness: int) -> float:
    """Down-weight by ``1 / (1 + staleness)``; staleness is clamped at 0."""
    return 1.0 / (1.0 + max(0, staleness))


class Asynchronous(Synchronizer):
    """Fire on every single arrival; weight by a staleness function of
    ``current_round - contribution.version``."""

    def __init__(
        self, staleness_fn: Callable[[int], float] = inverse_staleness
    ) -> None:
        self.staleness_fn = staleness_fn

    def ready(self, collected: List[Contribution], cohort: List[str]) -> bool:
        return len(collected) >= 1

    def weight(self, contribution: Contribution, current_round: int) -> float:
        return self.staleness_fn(current_round - contribution.version)
