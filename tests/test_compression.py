# -*- coding: utf-8 -*-
import numpy as np
import pytest

from fedclypse.compression import identity, topk
from fedclypse.parameters import Parameters


def _p(*arrays):
    return Parameters([np.array(a, dtype=float) for a in arrays])


def test_identity_returns_same_object():
    p = _p([1.0, 2.0])
    assert identity(p) is p


def test_topk_keeps_largest_magnitude_zeroes_rest():
    result = topk(0.5)(_p([1.0, -4.0, 2.0, -3.0]))
    assert np.allclose(result.tensors[0], [0.0, -4.0, 0.0, -3.0])


def test_topk_full_fraction_preserves_values():
    result = topk(1.0)(_p([1.0, -4.0, 2.0]))
    assert np.allclose(result.tensors[0], [1.0, -4.0, 2.0])


def test_topk_keeps_at_least_one_entry():
    result = topk(0.01)(_p([1.0, 5.0, 2.0, 3.0]))
    assert np.allclose(result.tensors[0], [0.0, 5.0, 0.0, 0.0])


def test_topk_handles_multiple_tensors():
    result = topk(0.5)(_p([1.0, -4.0, 2.0, -3.0], [10.0, -1.0]))
    assert np.allclose(result.tensors[0], [0.0, -4.0, 0.0, -3.0])
    assert np.allclose(result.tensors[1], [10.0, 0.0])


def test_topk_rejects_out_of_range_fraction():
    with pytest.raises(ValueError):
        topk(0.0)
    with pytest.raises(ValueError):
        topk(1.5)
