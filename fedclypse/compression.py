# -*- coding: utf-8 -*-
from __future__ import annotations

from typing import Callable

import numpy as np

from fedclypse.parameters import Parameters


def identity(params: Parameters) -> Parameters:
    """No-op compression."""
    return params


def topk(fraction: float) -> Callable[[Parameters], Parameters]:
    """Return a sparsification policy that keeps, per tensor, the ``fraction`` of
    entries with the largest magnitude and zeroes the rest."""
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
