# -*- coding: utf-8 -*-
from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any, Dict, List, Sequence

import numpy as np


class DataSource(ABC):
    """A re-openable, picklable locator for a dataset.

    ``open()`` must work on any worker — it (re)loads the data locally rather than
    holding a driver-bound object — so that only integer indices need to travel with
    a client. ``labels()`` is called once on the driver to compute the partition.
    """

    @abstractmethod
    def open(self) -> Any:
        """Return an indexable dataset (supports ``len`` and ``[]``)."""

    @abstractmethod
    def labels(self) -> np.ndarray:
        """Return the per-sample labels, aligned with ``open()``'s indices."""


class Subset:
    """A lazy view of an indexable dataset restricted to ``indices``."""

    def __init__(self, dataset: Any, indices: Sequence[int]) -> None:
        self.dataset = dataset
        self.indices = list(indices)

    def __len__(self) -> int:
        return len(self.indices)

    def __getitem__(self, i: int) -> Any:
        return self.dataset[self.indices[i]]


@dataclass
class ClientData:
    """A client's share of a dataset: a re-openable source + its index list.

    Light and picklable — only the indices cross the wire. ``materialize()`` runs on
    the worker, re-opening the source and viewing just this client's indices.
    """

    source: DataSource
    indices: List[int]

    def __post_init__(self) -> None:
        self.indices = list(self.indices)

    def materialize(self) -> Subset:
        return Subset(self.source.open(), self.indices)


class InMemorySource(DataSource):
    """A ``DataSource`` backed by an in-memory indexable dataset and a labels array.

    Useful for tests and for small, self-contained datasets (the materialized-shard
    case) where re-opening is just returning the held object.
    """

    def __init__(self, data: Any, labels: Sequence[int]) -> None:
        self._data = data
        self._labels = np.asarray(labels)

    def open(self) -> Any:
        return self._data

    def labels(self) -> np.ndarray:
        return self._labels


def split(
    source: DataSource, partitioner: Any, num_clients: int, seed: int = 0
) -> Dict[str, ClientData]:
    """Partition ``source`` across clients and return ``{client_id: ClientData}``.

    The partition is computed on the driver from ``source.labels()``; each returned
    ``ClientData`` carries only the source and its indices.
    """
    index_map = partitioner.partition(source.labels(), num_clients, seed)
    return {cid: ClientData(source, indices) for cid, indices in index_map.items()}
