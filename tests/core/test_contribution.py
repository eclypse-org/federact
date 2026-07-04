# -*- coding: utf-8 -*-
import numpy as np
import pytest

from fedclypse.core.contribution import Contribution, parameters, pseudogradient
from fedclypse.core.parameters import Parameters


def _p(*arrays):
    return Parameters([np.array(a, dtype=float) for a in arrays])


def test_defaults_are_neutral_provenance():
    c = Contribution(payload=_p([1.0]))
    assert c.source == "" and c.version == 0
    assert c.model_descriptor is None and c.weight == 1.0


def test_params_returns_payload_when_parameters():
    p = _p([1.0, 2.0])
    assert Contribution(payload=p).params is p


def test_params_raises_when_payload_not_parameters():
    with pytest.raises(TypeError):
        Contribution(payload={"logits": [1, 2, 3]}).params


def test_parameters_constructor_carries_provenance():
    c = parameters(_p([1.0]), source="client_3", version=5,
                   model_descriptor="cnn", weight=42.0)
    assert c.source == "client_3" and c.version == 5
    assert c.model_descriptor == "cnn" and c.weight == 42.0
    assert isinstance(c.payload, Parameters)


def test_pseudogradient_payload_is_after_minus_before():
    before = _p([1.0, 2.0])
    after = _p([4.0, 6.0])
    c = pseudogradient(before, after, source="c", weight=3.0)
    assert np.allclose(c.params.tensors[0], [3.0, 4.0])
    assert c.source == "c" and c.weight == 3.0
