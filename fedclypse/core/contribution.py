"""The Contribution type: an opaque payload plus the provenance an aggregator needs.

``Contribution`` is the currency of exchange between federation nodes and
aggregation rules. This module also provides two convenience constructors,
``parameters`` and ``pseudogradient``, for the two most common ``Parameters``
based payloads.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from fedclypse.core.parameters import Parameters

__all__ = ["Contribution", "parameters", "pseudogradient"]


@dataclass
class Contribution:
    """The currency of exchange between federation nodes and aggregators.

    An opaque, algorithm-defined ``payload`` (the "knowledge") plus a
    transparent provenance envelope an aggregator needs to combine it
    responsibly. The payload is interpreted only by the aggregation tool;
    ``Parameters`` is its default realization, but it may equally be a
    pseudogradient, logits, prototypes, or representations.

    Attributes:
        payload (Any): The algorithm-defined knowledge being contributed (e.g.
            a ``Parameters`` bundle, a pseudogradient, logits, prototypes, or
            other representations). Its type is opaque to ``Contribution``
            itself and is interpreted only by the aggregator that consumes it.
        source (str): An identifier for the contribution's origin (e.g. the
            client/node id). Defaults to ``""``.
        version (int): The model version the contribution was computed
            against. Defaults to ``0``.
        model_descriptor (Optional[str]): An opaque tag identifying the model
            architecture/shape the payload is compatible with, used by
            heterogeneity-aware aggregators to group compatible contributions.
            Defaults to ``None``.
        weight (float): The relative importance of this contribution in an
            aggregation (e.g. the client's number of local examples for
            FedAvg). Defaults to ``1.0``.
    """

    payload: Any
    source: str = ""
    version: int = 0
    model_descriptor: str | None = None
    weight: float = 1.0

    @property
    def params(self) -> Parameters:
        """Parameters: The payload cast to ``Parameters``.

        Raises:
            TypeError: If ``payload`` is not a ``Parameters`` instance (e.g.
                it holds a pseudogradient represented differently, logits, or
                another algorithm-specific payload type).
        """
        if not isinstance(self.payload, Parameters):
            raise TypeError(
                f"Contribution payload is {type(self.payload).__name__}, not Parameters"
            )
        return self.payload


def parameters(
    params: Parameters,
    *,
    source: str = "",
    version: int = 0,
    model_descriptor: str | None = None,
    weight: float = 1.0,
) -> Contribution:
    """Wrap a model's parameters as a Contribution (the homogeneous default).

    Args:
        params (Parameters): The model parameters to carry as the
            contribution's payload.
        source (str): An identifier for the contribution's origin. Defaults
            to ``""``.
        version (int): The model version ``params`` was computed against.
            Defaults to ``0``.
        model_descriptor (Optional[str]): An opaque tag identifying the model
            architecture/shape, used by heterogeneity-aware aggregators.
            Defaults to ``None``.
        weight (float): The relative importance of this contribution in an
            aggregation. Defaults to ``1.0``.

    Returns:
        Contribution: A contribution whose payload is ``params``.
    """
    return Contribution(
        payload=params,
        source=source,
        version=version,
        model_descriptor=model_descriptor,
        weight=weight,
    )


def pseudogradient(
    before: Parameters,
    after: Parameters,
    *,
    source: str = "",
    version: int = 0,
    model_descriptor: str | None = None,
    weight: float = 1.0,
) -> Contribution:
    """Wrap the local model change ``after - before`` as a Contribution.

    Args:
        before (Parameters): The model parameters before local training (the
            starting point the change is measured from).
        after (Parameters): The model parameters after local training.
        source (str): An identifier for the contribution's origin. Defaults
            to ``""``.
        version (int): The model version ``before`` was computed against.
            Defaults to ``0``.
        model_descriptor (Optional[str]): An opaque tag identifying the model
            architecture/shape, used by heterogeneity-aware aggregators.
            Defaults to ``None``.
        weight (float): The relative importance of this contribution in an
            aggregation. Defaults to ``1.0``.

    Returns:
        Contribution: A contribution whose payload is the element-wise
        difference ``after - before``.
    """
    delta = after.add(before.scale(-1.0))
    return Contribution(
        payload=delta,
        source=source,
        version=version,
        model_descriptor=model_descriptor,
        weight=weight,
    )
