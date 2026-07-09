# -*- coding: utf-8 -*-
import numpy as np
import pytest

torch = pytest.importorskip("torch")
from torch import nn  # noqa: E402

from fedclypse.core import Parameters  # noqa: E402
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


def test_torch_model_roundtrip_preserves_distinct_per_tensor_values():
    m = TorchModel(nn.Linear(3, 2))
    target = [
        np.arange(t.size, dtype=np.float32).reshape(t.shape) + 1.0
        for t in m.get_parameters().tensors
    ]
    m.set_parameters(Parameters(target, "torch"))
    after = m.get_parameters().tensors
    assert all(np.allclose(a, b) for a, b in zip(after, target))
