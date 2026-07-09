# -*- coding: utf-8 -*-
import numpy as np
import pytest

from fedclypse.data import Dirichlet, IID, NaturalId, Pathological, QuantitySkew


def _covers_all(index_map, n):
    flat = sorted(i for indices in index_map.values() for i in indices)
    return flat == list(range(n))


def _all_python_ints(index_map):
    return all(type(i) is int for indices in index_map.values() for i in indices)


LABELS = np.array([0, 0, 0, 1, 1, 1, 2, 2, 2, 3])


def test_iid_partitions_all_indices_into_num_clients():
    m = IID().partition(LABELS, num_clients=3, seed=0)
    assert set(m) == {"client_0", "client_1", "client_2"}
    assert _covers_all(m, len(LABELS))
    assert _all_python_ints(m)


def test_iid_is_deterministic_for_a_seed():
    assert IID().partition(LABELS, 3, 0) == IID().partition(LABELS, 3, 0)


def test_iid_parts_are_near_equal():
    sizes = sorted(len(v) for v in IID().partition(LABELS, 3, 0).values())
    assert max(sizes) - min(sizes) <= 1


def test_dirichlet_covers_all_indices_exactly_once():
    m = Dirichlet(0.5).partition(LABELS, num_clients=3, seed=1)
    assert _covers_all(m, len(LABELS))
    assert _all_python_ints(m)


def test_dirichlet_is_deterministic_for_a_seed():
    assert Dirichlet(0.3).partition(LABELS, 4, 7) == Dirichlet(0.3).partition(
        LABELS, 4, 7
    )


def test_dirichlet_rejects_non_positive_alpha():
    with pytest.raises(ValueError):
        Dirichlet(0.0)


def test_pathological_covers_all_and_gives_each_client_its_shards():
    m = Pathological(2).partition(LABELS, num_clients=2, seed=0)
    assert set(m) == {"client_0", "client_1"}
    assert _covers_all(m, len(LABELS))
    assert _all_python_ints(m)


def test_pathological_is_deterministic_for_a_seed():
    assert Pathological(2).partition(LABELS, 2, 3) == Pathological(2).partition(
        LABELS, 2, 3
    )


def test_pathological_rejects_non_positive_classes_per_client():
    with pytest.raises(ValueError):
        Pathological(0)


def test_quantity_skew_covers_all_indices_exactly_once():
    m = QuantitySkew(0.5).partition(LABELS, num_clients=3, seed=2)
    assert _covers_all(m, len(LABELS))
    assert _all_python_ints(m)


def test_quantity_skew_is_deterministic_for_a_seed():
    assert QuantitySkew(0.5).partition(LABELS, 3, 2) == QuantitySkew(0.5).partition(
        LABELS, 3, 2
    )


def test_quantity_skew_rejects_non_positive_beta():
    with pytest.raises(ValueError):
        QuantitySkew(0.0)


def test_natural_id_groups_indices_by_their_id():
    m = NaturalId([7, 7, 9, 9, 9, 3]).partition(labels=None, num_clients=99, seed=0)
    assert m == {"client_7": [0, 1], "client_9": [2, 3, 4], "client_3": [5]}
    assert _all_python_ints(m)


def test_pathological_gives_empty_clients_when_more_clients_than_samples():
    m = Pathological(1).partition(np.array([0, 1]), num_clients=4, seed=0)
    assert set(m) == {"client_0", "client_1", "client_2", "client_3"}
    assert _covers_all(m, 2)
    assert sum(len(v) == 0 for v in m.values()) >= 2
