# -*- coding: utf-8 -*-
"""Homogeneous aggregation rules that combine contributions into new parameters."""
from __future__ import annotations

from typing import List

from fedclypse.contribution import Contribution
from fedclypse.parameters import Parameters

__all__ = ["IncompatibleContributionsError", "weighted_sum", "mean", "fedavg"]


class IncompatibleContributionsError(ValueError):
    """Raised when contributions cannot be combined by a homogeneous aggregator.

    This covers payloads that are not ``Parameters``, ``Parameters`` with
    mismatched tensor shapes, and contributions carrying more than one
    distinct ``model_descriptor``.
    """


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
    """Aggregate contributions by their raw, unnormalized weight.

    Computes ``Σ wᵢ · payloadᵢ`` over the contributions, where ``wᵢ`` is each
    ``Contribution.weight``. Unlike ``fedavg``, weights are used as-is and are
    not divided by their sum, so the result's magnitude scales with the total
    weight rather than being a true average.

    Args:
        contribs (List[Contribution]): The per-client contributions to
            combine. Payloads must be ``Parameters`` of matching shapes and
            compatible model descriptors.

    Returns:
        Parameters: The weighted-sum parameters.

    Raises:
        IncompatibleContributionsError: If payloads are not ``Parameters``,
            have mismatched shapes, or carry heterogeneous model descriptors.
        ValueError: If ``contribs`` is empty.
    """
    _check(contribs)
    acc = contribs[0].payload.zeros_like()
    for c in contribs:
        acc = acc.add(c.payload.scale(c.weight))
    return acc


def mean(contribs: List[Contribution]) -> Parameters:
    """Aggregate contributions by a uniform, unweighted average.

    Computes the arithmetic mean of the payloads, ignoring each
    ``Contribution.weight`` entirely — every contribution counts equally.

    Args:
        contribs (List[Contribution]): The per-client contributions to
            combine. Payloads must be ``Parameters`` of matching shapes and
            compatible model descriptors.

    Returns:
        Parameters: The uniformly-averaged parameters.

    Raises:
        IncompatibleContributionsError: If payloads are not ``Parameters``,
            have mismatched shapes, or carry heterogeneous model descriptors.
        ValueError: If ``contribs`` is empty.
    """
    _check(contribs)
    acc = contribs[0].payload.zeros_like()
    for c in contribs:
        acc = acc.add(c.payload)
    return acc.scale(1.0 / len(contribs))


def fedavg(contribs: List[Contribution]) -> Parameters:
    """Aggregate contributions by FedAvg's num-examples-weighted mean.

    Computes ``Σ (wᵢ / Σw) · payloadᵢ`` over the contributions, where ``wᵢ`` is
    each ``Contribution.weight`` (the number of local examples). This is the
    homogeneous aggregation rule assumed by standard FedAvg.

    Args:
        contribs (List[Contribution]): The per-client contributions to
            combine. Payloads must be ``Parameters`` of matching shapes and
            compatible model descriptors.

    Returns:
        Parameters: The weighted-mean parameters.

    Raises:
        IncompatibleContributionsError: If payloads are not ``Parameters``,
            have mismatched shapes, or carry heterogeneous model descriptors.
        ValueError: If ``contribs`` is empty or the total weight is zero.
    """
    _check(contribs)
    total = sum(c.weight for c in contribs)
    if total == 0:
        raise ValueError("Total contribution weight is zero")
    acc = contribs[0].payload.zeros_like()
    for c in contribs:
        acc = acc.add(c.payload.scale(c.weight / total))
    return acc
