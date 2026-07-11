"""The Model abstraction: an entity's framework-coupled state.

``Model`` exposes an entity's state to the federation only through the
agnostic ``Parameters`` container, keeping every framework-specific detail
(torch modules, numpy arrays, ...) behind ``get_parameters``/``set_parameters``.
This module also provides ``ArrayModel``, a framework-agnostic reference
implementation used in tests and for pure-simulation runs without a deep
learning framework.
"""

from __future__ import annotations

from abc import (
    ABC,
    abstractmethod,
)

from fedclypse.core.parameters import Parameters

__all__ = ["ArrayModel", "Model"]


class Model(ABC):
    """The entity's framework-coupled *state* face.

    Exposed to the federation only through agnostic ``Parameters``.
    ``descriptor`` names the architecture; a learner stamps it onto a
    ``Contribution.model_descriptor`` so a homogeneous aggregator can reject
    mixing incompatible models (see ``fedclypse.aggregation``).
    """

    @abstractmethod
    def get_parameters(self) -> Parameters:
        """Project the model's current state into an agnostic bundle.

        Returns:
            Parameters: The model's tensors, framework-agnostic.
        """
        ...

    @abstractmethod
    def set_parameters(self, params: Parameters) -> None:
        """Load an agnostic parameter bundle back into the model's state.

        Args:
            params (Parameters): The parameters to load. Must be
                shape-compatible with the model's current architecture.
        """
        ...

    @property
    @abstractmethod
    def descriptor(self) -> str:
        """str: A tag naming the model's architecture.

        Stamped onto a ``Contribution.model_descriptor`` so a homogeneous
        aggregator can reject mixing incompatible models.
        """
        ...


class ArrayModel(Model):
    """A framework-agnostic model that simply holds a ``Parameters`` value.

    Used for tests and for pure-simulation runs without a DL framework.
    """

    def __init__(self, parameters: Parameters, descriptor: str = "array") -> None:
        """Initialize the model with its held parameters.

        Args:
            parameters (Parameters): The parameter bundle held by this model;
                aliased, not copied.
            descriptor (str): A tag naming the model's architecture. Defaults
                to ``"array"``.
        """
        self._parameters = parameters
        self._descriptor = descriptor

    def get_parameters(self) -> Parameters:
        """Return the held parameters.

        Returns:
            Parameters: The parameter bundle held by this model, aliased
            (not copied).
        """
        return self._parameters

    def set_parameters(self, params: Parameters) -> None:
        """Replace the held parameters with a defensive copy of ``params``.

        Args:
            params (Parameters): The parameters to load. Every tensor is
                copied, so later in-place mutation of ``params`` does not
                alias this model's state.
        """
        self._parameters = Parameters(
            [t.copy() for t in params.tensors], params.tensor_type
        )

    @property
    def descriptor(self) -> str:
        """str: The tag naming this model's architecture."""
        return self._descriptor
