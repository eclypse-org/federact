# -*- coding: utf-8 -*-
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Optional

from fedclypse.core.parameters import Parameters


@dataclass
class Contribution:
    """The currency of exchange: an opaque, algorithm-defined ``payload`` (the
    "knowledge") plus a transparent provenance envelope an aggregator needs to
    combine responsibly.

    The payload is interpreted only by the aggregation tool; ``Parameters`` is its
    default realization, but it may equally be a pseudogradient, logits, prototypes,
    or representations.
    """

    payload: Any
    source: str = ""
    version: int = 0
    model_descriptor: Optional[str] = None
    weight: float = 1.0

    @property
    def params(self) -> Parameters:
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
    model_descriptor: Optional[str] = None,
    weight: float = 1.0,
) -> Contribution:
    """Wrap a model's parameters as a Contribution (the homogeneous default)."""
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
    model_descriptor: Optional[str] = None,
    weight: float = 1.0,
) -> Contribution:
    """Wrap the local model change ``after - before`` as a Contribution."""
    delta = after.add(before.scale(-1.0))
    return Contribution(
        payload=delta,
        source=source,
        version=version,
        model_descriptor=model_descriptor,
        weight=weight,
    )
