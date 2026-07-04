# -*- coding: utf-8 -*-
from __future__ import annotations

from dataclasses import dataclass
from typing import List, Tuple

import numpy as np


@dataclass
class Parameters:
    """A framework-agnostic bundle of model tensors.

    This is the default realization of a Contribution's payload and the unit of
    exchange that crosses the wire. It carries no framework object — only numpy
    arrays plus a ``tensor_type`` tag naming the framework they were projected from.
    """

    tensors: List[np.ndarray]
    tensor_type: str = "numpy"

    @property
    def shapes(self) -> List[Tuple[int, ...]]:
        return [t.shape for t in self.tensors]

    def is_compatible(self, other: "Parameters") -> bool:
        return self.shapes == other.shapes

    def zeros_like(self) -> "Parameters":
        return Parameters([np.zeros_like(t) for t in self.tensors], self.tensor_type)

    def add(self, other: "Parameters") -> "Parameters":
        if not self.is_compatible(other):
            raise ValueError(
                f"Cannot add Parameters with mismatched shapes: "
                f"{self.shapes} vs {other.shapes}"
            )
        return Parameters(
            [a + b for a, b in zip(self.tensors, other.tensors)], self.tensor_type
        )

    def scale(self, factor: float) -> "Parameters":
        return Parameters([t * factor for t in self.tensors], self.tensor_type)
