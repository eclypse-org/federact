# -*- coding: utf-8 -*-
"""Compression policies: reduce the size of a Contribution's payload.

A compression policy is a callable ``Parameters -> Parameters`` applied to a
contribution's payload before it crosses the wire (or before it is
aggregated). ``identity`` is a no-op baseline; ``topk`` is a factory that
builds a magnitude-based sparsification closure.
"""
from __future__ import annotations

from typing import Callable

import numpy as np

from fedclypse.parameters import Parameters

__all__ = ["identity", "topk"]


def identity(params: Parameters) -> Parameters:
    """No-op compression.

    Args:
        params (Parameters): The parameters to pass through unchanged.

    Returns:
        Parameters: The same ``params`` object, unmodified.
    """
    return params


def topk(fraction: float) -> Callable[[Parameters], Parameters]:
    """Build a sparsification policy that keeps only the largest entries.

    Args:
        fraction (float): Fraction of entries to keep per tensor, by
            magnitude. Must be in ``(0, 1]``.

    Returns:
        Callable[[Parameters], Parameters]: A compression policy that, per
        tensor, keeps the ``fraction`` of entries with the largest magnitude
        and zeroes the rest.

    Raises:
        ValueError: If ``fraction`` is not in ``(0, 1]``.
    """
    if not 0.0 < fraction <= 1.0:
        raise ValueError("topk fraction must be in (0, 1]")

    def compress(params: Parameters) -> Parameters:
        out = []
        for t in params.tensors:
            flat = t.ravel()
            k = max(1, int(round(fraction * flat.size)))
            if k >= flat.size:
                out.append(t.copy())
                continue
            keep = np.argpartition(np.abs(flat), flat.size - k)[flat.size - k :]
            sparse = np.zeros_like(flat)
            sparse[keep] = flat[keep]
            out.append(sparse.reshape(t.shape))
        return Parameters(out, params.tensor_type)

    return compress
