"""Synchronization policies: the aggregation trigger.

A ``Synchronizer`` decides *when* a round fires (``ready``) and *how* a
contribution is weighted by its staleness (``staleness_weight``). It is
orthogonal to both the network topology and the aggregation rule (e.g.
FedAvg): the same synchronizer can be paired with any topology or algorithm.
This module provides three reference policies -- ``Synchronous`` (barrier),
``BufferedAsync`` (fire on ``k`` arrivals), and ``Asynchronous`` (fire on
every arrival, staleness-weighted) -- plus the default staleness function
``inverse_staleness``.
"""

from __future__ import annotations

from abc import (
    ABC,
    abstractmethod,
)
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from collections.abc import Callable

    from fedclypse.core.contribution import Contribution

__all__ = [
    "Asynchronous",
    "BufferedAsync",
    "Synchronizer",
    "Synchronous",
    "inverse_staleness",
]


class Synchronizer(ABC):
    """A synchronization policy: the aggregation trigger.

    Decides when a round fires (``ready``) and how a contribution is weighted
    by its staleness (``staleness_weight``). Orthogonal to both the network
    topology and the aggregation algorithm.
    """

    @abstractmethod
    def ready(self, collected: list[Contribution], cohort: list[str]) -> bool:
        """Decide whether the collected contributions are enough to aggregate.

        Args:
            collected (List[Contribution]): The contributions received so far
                in the current round.
            cohort (List[str]): The ids of the neighbours selected to
                participate in the current round.

        Returns:
            bool: ``True`` if a round should fire now, ``False`` otherwise.
        """
        ...

    def staleness_weight(
        self,
        contribution: Contribution,  # noqa: ARG002
        current_round: int,  # noqa: ARG002
    ) -> float:
        """Compute the staleness multiplier applied to a contribution.

        The base implementation applies no staleness discount.

        Args:
            contribution (Contribution): The contribution being weighted.
            current_round (int): The round the aggregation is being computed
                for.

        Returns:
            float: The staleness multiplier; always ``1.0`` for the base
            ``Synchronizer``.
        """
        return 1.0


class Synchronous(Synchronizer):
    """Barrier: fire once every selected member of the cohort has reported."""

    def ready(self, collected: list[Contribution], cohort: list[str]) -> bool:
        """Check whether every cohort member has reported.

        Args:
            collected (List[Contribution]): The contributions received so far
                in the current round.
            cohort (List[str]): The ids of the neighbours selected to
                participate in the current round.

        Returns:
            bool: ``True`` once ``collected`` has at least as many entries as
            ``cohort``.
        """
        return len(collected) >= len(cohort)


class BufferedAsync(Synchronizer):
    """Fire every time ``k`` contributions accumulate, regardless of sender.

    Staleness weighting is intentionally out of scope for this policy: it
    inherits the flat ``staleness_weight() -> 1.0``. A staleness-weighted
    buffered-async variant (e.g. FedBuff-style) is a future extension.
    """

    def __init__(self, k: int) -> None:
        """Initialize the buffer threshold.

        Args:
            k (int): The number of contributions that must accumulate before
                a round fires. Must be at least 1.

        Raises:
            ValueError: If ``k`` is less than 1.
        """
        if k < 1:
            raise ValueError("BufferedAsync requires k >= 1")
        self.k = k

    def ready(
        self,
        collected: list[Contribution],
        cohort: list[str],  # noqa: ARG002
    ) -> bool:
        """Check whether the buffer threshold has been reached.

        Args:
            collected (List[Contribution]): The contributions received so far
                in the current round.
            cohort (List[str]): The ids of the neighbours selected to
                participate in the current round.

        Returns:
            bool: ``True`` once ``collected`` has at least ``self.k`` entries.
        """
        return len(collected) >= self.k


def inverse_staleness(staleness: int) -> float:
    """Down-weight a contribution inversely to its staleness.

    Args:
        staleness (int): The number of rounds the contribution lags the
            current round by. Clamped at 0 (never negative).

    Returns:
        float: ``1 / (1 + staleness)``, so a fresh contribution (staleness 0)
        weighs 1.0 and older contributions weigh proportionally less.
    """
    return 1.0 / (1.0 + max(0, staleness))


class Asynchronous(Synchronizer):
    """Fire on every single arrival, weighting each by its staleness."""

    def __init__(
        self, staleness_fn: Callable[[int], float] = inverse_staleness
    ) -> None:
        """Initialize the staleness function.

        Args:
            staleness_fn (Callable[[int], float]): A function mapping a
                staleness value (``current_round - contribution.version``) to
                a weight multiplier. Defaults to ``inverse_staleness``.
        """
        self.staleness_fn = staleness_fn

    def ready(
        self,
        collected: list[Contribution],
        cohort: list[str],  # noqa: ARG002
    ) -> bool:
        """Check whether at least one contribution has arrived.

        Args:
            collected (List[Contribution]): The contributions received so far
                in the current round.
            cohort (List[str]): The ids of the neighbours selected to
                participate in the current round.

        Returns:
            bool: ``True`` once ``collected`` has at least one entry.
        """
        return len(collected) >= 1

    def staleness_weight(self, contribution: Contribution, current_round: int) -> float:
        """Compute the staleness multiplier via ``self.staleness_fn``.

        Args:
            contribution (Contribution): The contribution being weighted.
            current_round (int): The round the aggregation is being computed
                for.

        Returns:
            float: ``self.staleness_fn(current_round - contribution.version)``.
        """
        return self.staleness_fn(current_round - contribution.version)
