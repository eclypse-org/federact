# -*- coding: utf-8 -*-
from __future__ import annotations

from typing import List

from fedclypse.core.contribution import Contribution
from fedclypse.core.parameters import Parameters


class IncompatibleContributionsError(ValueError):
    """Raised when contributions cannot be combined by a homogeneous aggregator."""


def _check(contribs: List[Contribution]) -> None:
    if not contribs:
        raise ValueError("Cannot aggregate an empty list of contributions")

    payloads = [c.payload for c in contribs]
    if not all(isinstance(p, Parameters) for p in payloads):
        raise IncompatibleContributionsError(
            "All contribution payloads must be Parameters for this aggregator"
        )

    first = payloads[0]
    if not all(first.is_compatible(p) for p in payloads[1:]):
        raise IncompatibleContributionsError(
            "Contribution parameter shapes are incompatible"
        )

    descriptors = {
        c.model_descriptor for c in contribs if c.model_descriptor is not None
    }
    if len(descriptors) > 1:
        raise IncompatibleContributionsError(
            f"Heterogeneous model descriptors {descriptors} require a "
            f"heterogeneity-aware aggregator, not a homogeneous one"
        )


def weighted_sum(contribs: List[Contribution]) -> Parameters:
    """Σ wᵢ · payloadᵢ."""
    _check(contribs)
    acc = contribs[0].payload.zeros_like()
    for c in contribs:
        acc = acc.add(c.payload.scale(c.weight))
    return acc


def mean(contribs: List[Contribution]) -> Parameters:
    """Uniform average of the payloads (ignores weights)."""
    _check(contribs)
    acc = contribs[0].payload.zeros_like()
    for c in contribs:
        acc = acc.add(c.payload)
    return acc.scale(1.0 / len(contribs))


def fedavg(contribs: List[Contribution]) -> Parameters:
    """Σ (wᵢ / Σw) · payloadᵢ — FedAvg's weight-by-num-examples aggregation."""
    _check(contribs)
    total = sum(c.weight for c in contribs)
    if total == 0:
        raise ValueError("Total contribution weight is zero")
    acc = contribs[0].payload.zeros_like()
    for c in contribs:
        acc = acc.add(c.payload.scale(c.weight / total))
    return acc
