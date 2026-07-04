# -*- coding: utf-8 -*-
from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Dict, List

import numpy as np


class Partitioner(ABC):
    """Splits a dataset (given its per-sample ``labels``) across ``num_clients``,
    deterministically for a ``seed``. Returns an index-map ``{client_id: [indices]}``
    that is a partition of ``range(len(labels))``."""

    @abstractmethod
    def partition(
        self, labels: np.ndarray, num_clients: int, seed: int
    ) -> Dict[str, List[int]]: ...


class IID(Partitioner):
    """Shuffle all indices and split into ``num_clients`` near-equal parts."""

    def partition(
        self, labels: np.ndarray, num_clients: int, seed: int
    ) -> Dict[str, List[int]]:
        rng = np.random.default_rng(seed)
        order = rng.permutation(len(labels))
        parts = np.array_split(order, num_clients)
        return {f"client_{i}": part.tolist() for i, part in enumerate(parts)}


class Dirichlet(Partitioner):
    """Label-distribution skew: for each class, draw client proportions
    ``~ Dir(alpha)`` and hand that class's samples out accordingly. Small ``alpha``
    => strong skew; large ``alpha`` => near-IID."""

    def __init__(self, alpha: float) -> None:
        if alpha <= 0:
            raise ValueError("Dirichlet alpha must be > 0")
        self.alpha = alpha

    def partition(
        self, labels: np.ndarray, num_clients: int, seed: int
    ) -> Dict[str, List[int]]:
        rng = np.random.default_rng(seed)
        labels = np.asarray(labels)
        result: Dict[str, List[int]] = {f"client_{i}": [] for i in range(num_clients)}
        for cls in np.unique(labels):
            class_idx = np.where(labels == cls)[0]
            rng.shuffle(class_idx)
            proportions = rng.dirichlet([self.alpha] * num_clients)
            cuts = (np.cumsum(proportions) * len(class_idx)).astype(int)[:-1]
            for i, chunk in enumerate(np.split(class_idx, cuts)):
                result[f"client_{i}"].extend(chunk.tolist())
        return result
