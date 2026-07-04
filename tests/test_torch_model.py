# -*- coding: utf-8 -*-
import numpy as np
import pytest

torch = pytest.importorskip("torch")
from torch import nn  # noqa: E402

from fedclypse.parameters import Parameters  # noqa: E402
from fedclypse.torch import TorchModel  # noqa: E402


def test_torch_model_get_parameters_matches_state_dict_shapes():
    params = TorchModel(nn.Linear(3, 2)).get_parameters()
    # Linear(3, 2): weight (2, 3) + bias (2,)
    assert params.shapes == [(2, 3), (2,)]
    assert params.tensor_type == "torch"


def test_torch_model_set_then_get_roundtrips_to_zeros():
    m = TorchModel(nn.Linear(3, 2))
    zeros = Parameters([np.zeros_like(t) for t in m.get_parameters().tensors], "torch")
    m.set_parameters(zeros)
    after = m.get_parameters()
    assert all(np.allclose(t, 0.0) for t in after.tensors)


def test_torch_model_descriptor_defaults_to_module_class_name():
    assert TorchModel(nn.Linear(2, 1)).descriptor == "Linear"


def test_torch_model_custom_descriptor():
    assert TorchModel(nn.Linear(2, 1), descriptor="mlp").descriptor == "mlp"
