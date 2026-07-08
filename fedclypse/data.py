# -*- coding: utf-8 -*-
"""Cluster-safe dataset access: re-openable sources, client shards, and splitting.

The data layer is designed so that only integer indices ever need to cross the
wire to a worker: a ``DataSource`` is re-openable (``open()`` (re)loads the data
locally rather than holding a driver-bound object), so a ``ClientData`` — a
source plus its index list — is light and picklable. ``labels()`` is called
once on the driver to compute the partition; ``ClientData.materialize()`` runs
on the worker to lazily view just that client's shard. ``split()`` ties a
``DataSource`` and a ``partition.Partitioner`` together to build the
per-client ``ClientData`` mapping.
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any, Dict, List, Sequence

import numpy as np

__all__ = ["DataSource", "Subset", "ClientData", "InMemorySource", "split"]


class DataSource(ABC):
    """A re-openable, picklable locator for a dataset.

    ``open()`` must work on any worker — it (re)loads the data locally rather than
    holding a driver-bound object — so that only integer indices need to travel with
    a client. ``labels()`` is called once on the driver to compute the partition.
    """

    @abstractmethod
    def open(self) -> Any:
        """Return an indexable dataset (supports ``len`` and ``[]``).

        Returns:
            Any: An object supporting ``len()`` and integer ``__getitem__``,
            freshly (re)loaded on whichever process calls this method.
        """

    @abstractmethod
    def labels(self) -> np.ndarray:
        """Return the per-sample labels, aligned with ``open()``'s indices.

        Returns:
            np.ndarray: The per-sample labels, in the same order as the
            dataset returned by ``open()``.
        """


class Subset:
    """A lazy view of an indexable dataset restricted to ``indices``."""

    def __init__(self, dataset: Any, indices: Sequence[int]) -> None:
        """Store the dataset and the indices this view exposes.

        Args:
            dataset (Any): The underlying indexable dataset (supports ``len``
                and ``[]``) to view a subset of.
            indices (Sequence[int]): The positions into ``dataset`` that make
                up this subset, in the order they should be exposed.
        """
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

    Attributes:
        source (DataSource): The re-openable dataset this shard is drawn from.
        indices (List[int]): The positions into ``source``'s data that belong
            to this client.
    """

    source: DataSource
    indices: List[int]

    def __post_init__(self) -> None:
        self.indices = list(self.indices)

    def materialize(self) -> Subset:
        """Build this client's dataset view on the worker.

        Re-opens ``source`` locally and wraps it in a ``Subset`` restricted to
        ``indices``, so the underlying data never needs to be pickled.

        Returns:
            Subset: A lazy view of ``source.open()`` restricted to ``indices``.
        """
        return Subset(self.source.open(), self.indices)


class InMemorySource(DataSource):
    """A ``DataSource`` backed by an in-memory indexable dataset and a labels array.

    Useful for tests and for small, self-contained datasets (the materialized-shard
    case) where re-opening is just returning the held object.
    """

    def __init__(self, data: Any, labels: Sequence[int]) -> None:
        """Store the dataset and its labels.

        Args:
            data (Any): The indexable dataset (supports ``len`` and ``[]``)
                to serve from ``open()``.
            labels (Sequence[int]): The per-sample labels, aligned with
                ``data``.
        """
        self._data = data
        self._labels = np.asarray(labels)

    def open(self) -> Any:
        """Return the held in-memory dataset.

        Returns:
            Any: The dataset passed to the constructor.
        """
        return self._data

    def labels(self) -> np.ndarray:
        """Return the held labels array.

        Returns:
            np.ndarray: The labels passed to the constructor.
        """
        return self._labels


def split(
    source: DataSource, partitioner: Any, num_clients: int, seed: int = 0
) -> Dict[str, ClientData]:
    """Partition ``source`` across clients and return ``{client_id: ClientData}``.

    The partition is computed on the driver from ``source.labels()``; each returned
    ``ClientData`` carries only the source and its indices.

    Args:
        source (DataSource): The dataset to split. Its ``labels()`` are used
            to compute the partition; its ``open()`` is deferred to each
            client's ``materialize()`` call.
        partitioner (Any): A ``partition.Partitioner``-like object exposing
            ``partition(labels, num_clients, seed) -> Dict[str, List[int]]``.
        num_clients (int): Number of clients to split across.
        seed (int): Seed for any randomness in the partition. Defaults to 0.

    Returns:
        Dict[str, ClientData]: A mapping from client id to that client's
        ``ClientData`` shard of ``source``.
    """
    index_map = partitioner.partition(source.labels(), num_clients, seed)
    return {cid: ClientData(source, indices) for cid, indices in index_map.items()}
