"""Framework-agnostic container for the model tensors carried by a Contribution.

``Parameters`` is the default payload realization used across fedclypse: it
holds a model's tensors as plain numpy arrays so aggregation rules can combine
them without depending on any particular deep-learning framework.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

__all__ = ["Parameters"]


@dataclass
class Parameters:
    """A framework-agnostic bundle of model tensors.

    This is the default realization of a Contribution's payload and the unit of
    exchange that crosses the wire. It carries no framework object — only numpy
    arrays plus a ``tensor_type`` tag naming the framework they were projected
    from.

    Construction does not copy: a ``Parameters`` aliases the arrays it is given,
    and the ``parameters()`` / ``pseudogradient()`` constructors likewise do not
    copy. Callers needing isolation from later in-place mutation must copy the
    arrays themselves. The non-mutating helpers (``zeros_like``, ``add``,
    ``scale``) always return new objects.

    Attributes:
        tensors (List[np.ndarray]): The ordered list of tensors making up the
            model's parameters (or a pseudogradient), aliased rather than
            copied at construction time.
        tensor_type (str): A tag naming the framework the tensors were
            projected from (e.g. ``"numpy"``, ``"torch"``). Defaults to
            ``"numpy"``.
    """

    tensors: list[np.ndarray]
    tensor_type: str = "numpy"

    @property
    def shapes(self) -> list[tuple[int, ...]]:
        """List[Tuple[int, ...]]: The shape of each tensor, in order."""
        return [t.shape for t in self.tensors]

    def is_compatible(self, other: Parameters) -> bool:
        """Check whether two parameter bundles have matching tensor shapes.

        Args:
            other (Parameters): The parameters to compare against.

        Returns:
            bool: ``True`` if ``other`` has the same number of tensors and each
            tensor has the same shape, in order, as this bundle's; ``False``
            otherwise.
        """
        return self.shapes == other.shapes

    def zeros_like(self) -> Parameters:
        """Build a zero-filled bundle with the same shapes and tensor type.

        Returns:
            Parameters: A new ``Parameters`` whose tensors are zero arrays with
            the same shapes as this bundle's, tagged with the same
            ``tensor_type``.
        """
        return Parameters([np.zeros_like(t) for t in self.tensors], self.tensor_type)

    def add(self, other: Parameters) -> Parameters:
        """Add two parameter bundles element-wise.

        Args:
            other (Parameters): The parameters to add to this bundle. Must be
                shape-compatible with this bundle (see ``is_compatible``).

        Returns:
            Parameters: A new ``Parameters`` whose tensors are the element-wise
            sum of this bundle's and ``other``'s.

        Raises:
            ValueError: If ``other``'s tensor shapes do not match this
                bundle's.
        """
        if not self.is_compatible(other):
            raise ValueError(
                f"Cannot add Parameters with mismatched shapes: "
                f"{self.shapes} vs {other.shapes}"
            )
        return Parameters(
            [a + b for a, b in zip(self.tensors, other.tensors, strict=False)],
            self.tensor_type,
        )

    def scale(self, factor: float) -> Parameters:
        """Scale every tensor by a scalar factor.

        Args:
            factor (float): The multiplier applied element-wise to each tensor.

        Returns:
            Parameters: A new ``Parameters`` with each tensor multiplied by ``factor``.
        """
        return Parameters([t * factor for t in self.tensors], self.tensor_type)
