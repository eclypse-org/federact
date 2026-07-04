# -*- coding: utf-8 -*-
import numpy as np
import pytest

from fedclypse.parameters import Parameters


def _p(*arrays):
    return Parameters([np.array(a, dtype=float) for a in arrays])


def test_shapes_reports_tensor_shapes():
    assert _p([1.0, 2.0], [[3.0, 4.0]]).shapes == [(2,), (1, 2)]


def test_is_compatible_true_for_same_shapes():
    assert _p([1.0, 2.0]).is_compatible(_p([5.0, 6.0]))


def test_is_compatible_false_for_different_shapes():
    assert not _p([1.0, 2.0]).is_compatible(_p([1.0, 2.0, 3.0]))


def test_zeros_like_returns_zero_tensors_same_shape():
    z = _p([1.0, 2.0], [[3.0, 4.0]]).zeros_like()
    assert z.shapes == [(2,), (1, 2)]
    assert all(np.all(t == 0) for t in z.tensors)


def test_add_sums_elementwise():
    r = _p([1.0, 2.0]).add(_p([3.0, 4.0]))
    assert np.allclose(r.tensors[0], [4.0, 6.0])


def test_add_rejects_mismatched_shapes():
    with pytest.raises(ValueError):
        _p([1.0, 2.0]).add(_p([1.0, 2.0, 3.0]))


def test_scale_multiplies_all_tensors():
    r = _p([1.0, 2.0], [[3.0]]).scale(2.0)
    assert np.allclose(r.tensors[0], [2.0, 4.0])
    assert np.allclose(r.tensors[1], [[6.0]])
