# -*- coding: utf-8 -*-
import pytest

from fedclypse.contribution import Contribution
from fedclypse.synchronization import (
    Asynchronous,
    BufferedAsync,
    Synchronous,
    inverse_staleness,
)

COHORT = ["a", "b", "c"]


def _contribs(n, version=0):
    return [Contribution(payload=None, version=version) for _ in range(n)]


def test_synchronous_ready_only_when_whole_cohort_reported():
    s = Synchronous()
    assert not s.ready(_contribs(2), COHORT)
    assert s.ready(_contribs(3), COHORT)


def test_synchronous_default_weight_is_one():
    s = Synchronous()
    assert (
        s.staleness_weight(Contribution(payload=None, version=0), current_round=5)
        == 1.0
    )


def test_buffered_async_ready_at_k():
    b = BufferedAsync(2)
    assert not b.ready(_contribs(1), COHORT)
    assert b.ready(_contribs(2), COHORT)


def test_buffered_async_rejects_k_below_one():
    with pytest.raises(ValueError):
        BufferedAsync(0)


def test_asynchronous_ready_on_first_arrival():
    a = Asynchronous()
    assert not a.ready(_contribs(0), COHORT)
    assert a.ready(_contribs(1), COHORT)


def test_asynchronous_weight_uses_staleness():
    a = Asynchronous()
    c = Contribution(payload=None, version=2)
    assert a.staleness_weight(c, current_round=5) == 0.25  # staleness 3 -> 1/(1+3)


def test_inverse_staleness_clamps_negative_to_zero():
    assert inverse_staleness(0) == 1.0
    assert inverse_staleness(-4) == 1.0
    assert inverse_staleness(3) == 0.25
