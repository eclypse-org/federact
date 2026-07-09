# -*- coding: utf-8 -*-
import asyncio

import numpy as np
import pytest

from fedclypse.core import ArrayModel, Entity, Parameters
from fedclypse.data import ClientData, InMemorySource


def _p(*arrays):
    return Parameters([np.array(a, dtype=float) for a in arrays])


def _client_data():
    return ClientData(InMemorySource([10, 11, 12], labels=[0, 1, 0]), [0, 2])


def test_entity_constructs_with_neutral_state():
    e = Entity(
        "client_0", model_factory=lambda: ArrayModel(_p([1.0])), data=_client_data()
    )
    assert e.id == "client_0"
    assert e.model is None
    assert e.dataset is None
    assert e.round == 0


def test_on_deploy_builds_model_and_materializes_data():
    e = Entity(
        "c",
        model_factory=lambda: ArrayModel(_p([5.0]), descriptor="m"),
        data=_client_data(),
    )
    e.on_deploy()
    assert isinstance(e.model, ArrayModel)
    assert e.model.descriptor == "m"
    assert len(e.dataset) == 2
    assert e.dataset[0] == 10 and e.dataset[1] == 12


def test_on_deploy_is_safe_without_model_or_data():
    e = Entity("c")
    e.on_deploy()
    assert e.model is None and e.dataset is None


def test_default_step_requires_override():
    e = Entity("c")
    with pytest.raises(NotImplementedError):
        asyncio.run(e.step())
