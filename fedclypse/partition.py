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


class Pathological(Partitioner):
    """Sort by label, cut into ``num_clients * classes_per_client`` label-contiguous
    shards, and assign ``classes_per_client`` shards to each client (McMahan-style
    pathological non-IID: each client sees only a few classes)."""

    def __init__(self, classes_per_client: int) -> None:
        if classes_per_client < 1:
            raise ValueError("classes_per_client must be >= 1")
        self.classes_per_client = classes_per_client

    def partition(
        self, labels: np.ndarray, num_clients: int, seed: int
    ) -> Dict[str, List[int]]:
        rng = np.random.default_rng(seed)
        labels = np.asarray(labels)
        order = np.argsort(labels, kind="stable")
        n_shards = num_clients * self.classes_per_client
        shards = np.array_split(order, n_shards)
        shard_order = rng.permutation(n_shards)
        result: Dict[str, List[int]] = {}
        for i in range(num_clients):
            assigned = shard_order[
                i * self.classes_per_client : (i + 1) * self.classes_per_client
            ]
            result[f"client_{i}"] = np.concatenate(
                [shards[s] for s in assigned]
            ).tolist()
        return result


class QuantitySkew(Partitioner):
    """Clients get unequal *amounts* of data: sizes are drawn ``~ Dir(beta)`` over
    clients (labels do not affect the split). Small ``beta`` => more imbalance."""

    def __init__(self, beta: float) -> None:
        if beta <= 0:
            raise ValueError("QuantitySkew beta must be > 0")
        self.beta = beta

    def partition(
        self, labels: np.ndarray, num_clients: int, seed: int
    ) -> Dict[str, List[int]]:
        rng = np.random.default_rng(seed)
        n = len(labels)
        order = rng.permutation(n)
        proportions = rng.dirichlet([self.beta] * num_clients)
        cuts = (np.cumsum(proportions) * n).astype(int)[:-1]
        return {
            f"client_{i}": chunk.tolist()
            for i, chunk in enumerate(np.split(order, cuts))
        }


class NaturalId(Partitioner):
    """Partition by a natural grouping column (e.g. writer / user / speaker id):
    one client per unique id, aligned with the samples. ``num_clients`` and ``seed``
    are accepted for interface uniformity but ignored — the data fixes the split."""

    def __init__(self, ids) -> None:
        self._ids = list(ids)

    def partition(
        self, labels: np.ndarray, num_clients: int = 0, seed: int = 0
    ) -> Dict[str, List[int]]:
        result: Dict[str, List[int]] = {}
        for i, gid in enumerate(self._ids):
            result.setdefault(f"client_{gid}", []).append(int(i))
        return result
