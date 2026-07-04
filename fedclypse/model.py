# -*- coding: utf-8 -*-
from __future__ import annotations

from abc import ABC, abstractmethod

from fedclypse.parameters import Parameters


class Model(ABC):
    """The entity's framework-coupled *state* face, exposed to the federation only
    through agnostic ``Parameters``.

    ``descriptor`` names the architecture; a learner stamps it onto a
    ``Contribution.model_descriptor`` so a homogeneous aggregator can reject mixing
    incompatible models (see ``fedclypse.aggregation``).
    """

    @abstractmethod
    def get_parameters(self) -> Parameters: ...

    @abstractmethod
    def set_parameters(self, params: Parameters) -> None: ...

    @property
    @abstractmethod
    def descriptor(self) -> str: ...


class ArrayModel(Model):
    """A framework-agnostic model that simply holds a ``Parameters`` value — for tests
    and for pure-simulation runs without a DL framework."""

    def __init__(self, parameters: Parameters, descriptor: str = "array") -> None:
        self._parameters = parameters
        self._descriptor = descriptor

    def get_parameters(self) -> Parameters:
        return self._parameters

    def set_parameters(self, params: Parameters) -> None:
        self._parameters = Parameters(
            [t.copy() for t in params.tensors], params.tensor_type
        )

    @property
    def descriptor(self) -> str:
        return self._descriptor
