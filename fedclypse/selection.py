# -*- coding: utf-8 -*-
"""Cohort-selection policies: which neighbours participate in a round.

A selection policy is a callable ``List[str] -> List[str]`` that, given the
current list of eligible neighbour ids, returns the subset ("cohort") invited
to participate in a round. ``select_all`` is a plain function; ``uniform`` and
``at_most`` are factories that build a stateful, seeded policy closure so
successive calls draw a fresh, reproducible sequence of cohorts.
"""
from __future__ import annotations

import random
from typing import Callable, List

__all__ = ["select_all", "uniform", "at_most"]


def select_all(neighbours: List[str]) -> List[str]:
    """Participation policy that selects every neighbour.

    Args:
        neighbours (List[str]): The ids of the currently eligible neighbours.

    Returns:
        List[str]: A new list containing every id in ``neighbours``.
    """
    return list(neighbours)


def uniform(fraction: float, seed: int = 0) -> Callable[[List[str]], List[str]]:
    """Build a policy that uniformly samples a fraction of the neighbours.

    Args:
        fraction (float): Fraction of neighbours to select each round; at
            least one is chosen when any exist.
        seed (int): Seed for the policy's random number generator. Defaults
            to 0.

    Returns:
        Callable[[List[str]], List[str]]: A selection policy. Each call
        advances the RNG, so successive rounds draw fresh cohorts and the
        seed reproduces the whole sequence.
    """
    rng = random.Random(seed)

    def select(neighbours: List[str]) -> List[str]:
        population = list(neighbours)
        k = min(len(population), max(1, round(fraction * len(population))))
        return rng.sample(population, k)

    return select


def at_most(k: int, seed: int = 0) -> Callable[[List[str]], List[str]]:
    """Build a policy that samples at most ``k`` neighbours per round.

    Args:
        k (int): The maximum number of neighbours to select each round; fewer
            are returned if the neighbour pool is smaller than ``k``.
        seed (int): Seed for the policy's random number generator. Defaults
            to 0.

    Returns:
        Callable[[List[str]], List[str]]: A selection policy. Each call
        advances the RNG, so successive rounds draw fresh cohorts and the
        seed reproduces the whole sequence.
    """
    rng = random.Random(seed)

    def select(neighbours: List[str]) -> List[str]:
        population = list(neighbours)
        return rng.sample(population, min(k, len(population)))

    return select
