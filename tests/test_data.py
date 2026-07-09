# -*- coding: utf-8 -*-
import numpy as np

from fedclypse.data import ClientData, InMemorySource, Subset, split


def test_subset_is_a_lazy_view_by_index():
    sub = Subset(["a", "b", "c", "d"], [3, 1])
    assert len(sub) == 2
    assert sub[0] == "d"
    assert sub[1] == "b"


def test_in_memory_source_exposes_data_and_labels():
    src = InMemorySource([10, 11, 12], labels=[0, 1, 0])
    assert list(src.open()) == [10, 11, 12]
    assert np.array_equal(src.labels(), np.array([0, 1, 0]))


def test_client_data_materializes_its_shard_from_the_source():
    src = InMemorySource([10, 11, 12, 13], labels=[0, 0, 1, 1])
    shard = ClientData(src, [0, 2]).materialize()
    assert len(shard) == 2
    assert shard[0] == 10
    assert shard[1] == 12


def test_split_maps_partitioner_output_to_client_data():
    class _FakePartitioner:
        def partition(self, labels, num_clients, seed):
            return {"client_0": [0, 1], "client_1": [2]}

    src = InMemorySource([10, 11, 12], labels=[0, 1, 0])
    clients = split(src, _FakePartitioner(), num_clients=2, seed=0)

    assert set(clients) == {"client_0", "client_1"}
    assert isinstance(clients["client_0"], ClientData)
    assert clients["client_0"].indices == [0, 1]
    assert [clients["client_1"].materialize()[0]] == [12]


def test_split_with_dirichlet_produces_disjoint_materializable_shards():
    from fedclypse.data import Dirichlet

    data = list(range(20))
    labels = [i % 4 for i in range(20)]
    src = InMemorySource(data, labels)

    clients = split(src, Dirichlet(0.5), num_clients=3, seed=0)

    seen = []
    for cd in clients.values():
        shard = cd.materialize()
        seen.extend(shard[j] for j in range(len(shard)))
    assert sorted(seen) == data  # every sample delivered exactly once, values intact


def test_client_data_copies_indices_to_avoid_aliasing():
    idx = [0, 1]
    cd = ClientData(InMemorySource([10, 11, 12], [0, 0, 1]), idx)
    idx.append(2)
    assert cd.indices == [0, 1]


def test_subset_supports_empty_indices():
    sub = Subset(["a", "b"], [])
    assert len(sub) == 0
