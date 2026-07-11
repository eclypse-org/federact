"""Sample-index partitioners: turn per-sample labels into per-client shards.

A ``Partitioner`` maps a dataset's ``labels`` array to an index-map
``{client_id: [indices]}`` that is a true partition of ``range(len(labels))``
(every index appears in exactly one client, none dropped or duplicated). This
module provides five reference strategies: ``IID`` (uniform random split),
``Dirichlet`` (label-distribution skew), ``Pathological`` (shard-based,
McMahan-style non-IID), ``QuantitySkew`` (unequal shard sizes, labels
unaffected), and ``NaturalId`` (client boundaries fixed by a pre-existing id
column, e.g. writer/speaker/user id). ``data.split()`` drives a ``Partitioner``
against a ``data.DataSource`` to build ``ClientData`` shards.
"""

from __future__ import annotations

from abc import (
    ABC,
    abstractmethod,
)

import numpy as np

__all__ = [
    "IID",
    "Dirichlet",
    "NaturalId",
    "Partitioner",
    "Pathological",
    "QuantitySkew",
]


class Partitioner(ABC):
    """Splits a dataset across clients, deterministically for a seed.

    Given a dataset's per-sample ``labels``, splits it across ``num_clients``,
    deterministically for a ``seed``. Returns an index-map
    ``{client_id: [indices]}`` that is a partition of ``range(len(labels))``.
    """

    @abstractmethod
    def partition(
        self, labels: np.ndarray, num_clients: int, seed: int
    ) -> dict[str, list[int]]:
        """Partition sample indices across clients.

        Args:
            labels (np.ndarray): Per-sample labels for the full dataset.
            num_clients (int): Number of clients to split across.
            seed (int): Seed for any randomness in the partition.

        Returns:
            Dict[str, List[int]]: A mapping from client id to the list of sample indices
            assigned to that client.
        """
        ...


class IID(Partitioner):
    """Shuffle all indices and split into ``num_clients`` near-equal parts."""

    def partition(
        self, labels: np.ndarray, num_clients: int, seed: int
    ) -> dict[str, list[int]]:
        """Randomly shuffle and evenly split all sample indices.

        Args:
            labels (np.ndarray): Per-sample labels for the full dataset (only
                its length is used; label values do not affect the split).
            num_clients (int): Number of clients to split across.
            seed (int): Seed for the shuffle.

        Returns:
            Dict[str, List[int]]: A mapping from client id to a near-equal
            share of the shuffled indices.
        """
        rng = np.random.default_rng(seed)
        order = rng.permutation(len(labels))
        parts = np.array_split(order, num_clients)
        return {f"client_{i}": part.tolist() for i, part in enumerate(parts)}


class Dirichlet(Partitioner):
    """Label-distribution skew via a per-class Dirichlet draw.

    For each class, draws client proportions ``~ Dir(alpha)`` and hands that
    class's samples out accordingly. Small ``alpha`` => strong skew; large
    ``alpha`` => near-IID.
    """

    def __init__(self, alpha: float) -> None:
        """Set the Dirichlet concentration parameter.

        Args:
            alpha (float): Concentration of the per-class ``Dir(alpha)`` draw.
                Must be strictly positive. Small values yield strong
                label-distribution skew; large values approach IID.

        Raises:
            ValueError: If ``alpha`` is not strictly positive.
        """
        if alpha <= 0:
            raise ValueError("Dirichlet alpha must be > 0")
        self.alpha = alpha

    def partition(
        self, labels: np.ndarray, num_clients: int, seed: int
    ) -> dict[str, list[int]]:
        """Split each class's samples across clients via a Dirichlet draw.

        Args:
            labels (np.ndarray): Per-sample labels for the full dataset.
            num_clients (int): Number of clients to split across.
            seed (int): Seed for the per-class shuffles and Dirichlet draws.

        Returns:
            Dict[str, List[int]]: A mapping from client id to the list of
            sample indices assigned to that client, covering every index
            exactly once.
        """
        rng = np.random.default_rng(seed)
        labels = np.asarray(labels)
        result: dict[str, list[int]] = {f"client_{i}": [] for i in range(num_clients)}
        for cls in np.unique(labels):
            class_idx = np.where(labels == cls)[0]
            rng.shuffle(class_idx)
            proportions = rng.dirichlet([self.alpha] * num_clients)
            cuts = (np.cumsum(proportions) * len(class_idx)).astype(int)[:-1]
            for i, chunk in enumerate(np.split(class_idx, cuts)):
                result[f"client_{i}"].extend(chunk.tolist())
        return result


class Pathological(Partitioner):
    """Shard-based pathological non-IID split (McMahan-style).

    Sorts by label, cuts into ``num_clients * classes_per_client``
    label-contiguous shards, and assigns ``classes_per_client`` shards to each
    client, so each client sees only a few classes.
    """

    def __init__(self, classes_per_client: int) -> None:
        """Set how many label-contiguous shards each client receives.

        Args:
            classes_per_client (int): Number of shards assigned to each
                client. Must be at least 1.

        Raises:
            ValueError: If ``classes_per_client`` is less than 1.
        """
        if classes_per_client < 1:
            raise ValueError("classes_per_client must be >= 1")
        self.classes_per_client = classes_per_client

    def partition(
        self, labels: np.ndarray, num_clients: int, seed: int
    ) -> dict[str, list[int]]:
        """Cut label-sorted indices into shards and deal them out to clients.

        Args:
            labels (np.ndarray): Per-sample labels for the full dataset.
            num_clients (int): Number of clients to split across.
            seed (int): Seed for the shard-to-client assignment shuffle.

        Returns:
            Dict[str, List[int]]: A mapping from client id to the list of
            sample indices assigned to that client, covering every index
            exactly once.
        """
        rng = np.random.default_rng(seed)
        labels = np.asarray(labels)
        order = np.argsort(labels, kind="stable")
        n_shards = num_clients * self.classes_per_client
        shards = np.array_split(order, n_shards)
        shard_order = rng.permutation(n_shards)
        result: dict[str, list[int]] = {}
        for i in range(num_clients):
            assigned = shard_order[
                i * self.classes_per_client : (i + 1) * self.classes_per_client
            ]
            result[f"client_{i}"] = np.concatenate(
                [shards[s] for s in assigned]
            ).tolist()
        return result


class QuantitySkew(Partitioner):
    """Unequal shard sizes via a Dirichlet draw over clients.

    Clients get unequal *amounts* of data: sizes are drawn ``~ Dir(beta)``
    over clients (labels do not affect the split). Small ``beta`` => more
    imbalance.
    """

    def __init__(self, beta: float) -> None:
        """Set the Dirichlet concentration parameter over client shard sizes.

        Args:
            beta (float): Concentration of the ``Dir(beta)`` draw over
                per-client shard sizes. Must be strictly positive. Small
                values yield strongly unequal shard sizes; large values
                approach equal-sized shards.

        Raises:
            ValueError: If ``beta`` is not strictly positive.
        """
        if beta <= 0:
            raise ValueError("QuantitySkew beta must be > 0")
        self.beta = beta

    def partition(
        self, labels: np.ndarray, num_clients: int, seed: int
    ) -> dict[str, list[int]]:
        """Split shuffled indices into unequal-sized, label-agnostic shards.

        Args:
            labels (np.ndarray): Per-sample labels for the full dataset (only
                its length is used; label values do not affect the split).
            num_clients (int): Number of clients to split across.
            seed (int): Seed for the shuffle and the Dirichlet draw over
                shard sizes.

        Returns:
            Dict[str, List[int]]: A mapping from client id to the list of
            sample indices assigned to that client, covering every index
            exactly once.
        """
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
    """Partition by a pre-existing natural grouping column.

    One client per unique id (e.g. writer / user / speaker id), aligned with
    the samples. ``num_clients`` and ``seed`` are accepted for interface
    uniformity but ignored — the data fixes the split.
    """

    def __init__(self, ids) -> None:
        """Store the per-sample natural group id.

        Args:
            ids: A sequence of per-sample group ids (e.g. writer/user/speaker
                id), aligned index-for-index with the dataset to be
                partitioned.
        """
        self._ids = list(ids)

    def partition(
        self,
        labels: np.ndarray,  # noqa: ARG002
        num_clients: int = 0,  # noqa: ARG002
        seed: int = 0,  # noqa: ARG002
    ) -> dict[str, list[int]]:
        """Group sample indices by their pre-existing natural id.

        Args:
            labels (np.ndarray): Per-sample labels for the full dataset;
                unused, since the split is fixed by ``self._ids``.
            num_clients (int): Accepted for interface uniformity with other
                ``Partitioner`` subclasses but ignored — the number of
                clients is determined by the number of unique ids. Defaults
                to 0.
            seed (int): Accepted for interface uniformity but ignored, since
                this partitioner has no randomness. Defaults to 0.

        Returns:
            Dict[str, List[int]]: A mapping from ``"client_{id}"`` (for each
            unique id in ``self._ids``) to the list of sample indices sharing
            that id, covering every index exactly once.
        """
        result: dict[str, list[int]] = {}
        for i, gid in enumerate(self._ids):
            result.setdefault(f"client_{gid}", []).append(int(i))
        return result
