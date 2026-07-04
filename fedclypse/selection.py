# -*- coding: utf-8 -*-
from __future__ import annotations

import random
from typing import Callable, List


def select_all(neighbours: List[str]) -> List[str]:
    """Participation policy that selects every neighbour."""
    return list(neighbours)


def uniform(fraction: float, seed: int = 0) -> Callable[[List[str]], List[str]]:
    """Return a policy that uniformly samples ``fraction`` of the neighbours
    (at least one when any exist), deterministically for the given ``seed``."""
    rng = random.Random(seed)

    def select(neighbours: List[str]) -> List[str]:
        population = list(neighbours)
        k = min(len(population), max(1, round(fraction * len(population))))
        return rng.sample(population, k)

    return select


def at_most(k: int, seed: int = 0) -> Callable[[List[str]], List[str]]:
    """Return a policy that samples at most ``k`` neighbours, deterministically
    for the given ``seed``."""
    rng = random.Random(seed)

    def select(neighbours: List[str]) -> List[str]:
        population = list(neighbours)
        return rng.sample(population, min(k, len(population)))

    return select
