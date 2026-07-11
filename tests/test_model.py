import numpy as np
import pytest

from fedclypse.core import ArrayModel, Model, Parameters


def _p(*arrays):
    return Parameters([np.array(a, dtype=float) for a in arrays])


def test_model_is_abstract():
    with pytest.raises(TypeError):
        Model()


def test_array_model_get_returns_held_parameters():
    p = _p([1.0, 2.0])
    assert ArrayModel(p).get_parameters() is p


def test_array_model_set_replaces_parameters():
    m = ArrayModel(_p([1.0, 2.0]))
    m.set_parameters(_p([3.0, 4.0]))
    assert np.allclose(m.get_parameters().tensors[0], [3.0, 4.0])


def test_array_model_set_copies_to_avoid_aliasing():
    src = _p([1.0, 2.0])
    m = ArrayModel(_p([0.0, 0.0]))
    m.set_parameters(src)
    src.tensors[0][0] = 99.0
    assert m.get_parameters().tensors[0][0] == 1.0


def test_array_model_default_descriptor():
    assert ArrayModel(_p([1.0])).descriptor == "array"


def test_array_model_custom_descriptor():
    assert ArrayModel(_p([1.0]), descriptor="cnn").descriptor == "cnn"


def test_array_model_set_copies_all_tensors_and_preserves_type():
    src = Parameters([np.array([1.0, 2.0]), np.array([[3.0], [4.0]])], "custom")
    m = ArrayModel(_p([0.0]))
    m.set_parameters(src)
    src.tensors[1][0, 0] = 99.0
    got = m.get_parameters()
    assert got.tensor_type == "custom"
    assert np.allclose(got.tensors[0], [1.0, 2.0])
    assert got.tensors[1][0, 0] == 3.0
