# -*- coding: utf-8 -*-
"""PyTorch adapter for the agnostic ``Model`` interface: ``TorchModel``.

Bridges a ``torch.nn.Module``'s ``state_dict`` and fedclypse's
framework-agnostic ``Parameters``/``Model`` abstractions, so torch modules
can be trained and aggregated with the rest of fedclypse without further
glue code.
"""
from __future__ import annotations

from typing import Optional

import torch

from fedclypse.model import Model
from fedclypse.parameters import Parameters

__all__ = ["TorchModel"]


class TorchModel(Model):
    """Adapts a ``torch.nn.Module`` to the agnostic ``Model`` interface.

    ``Parameters`` are the module's ``state_dict`` values as numpy arrays, in
    ``state_dict`` order; ``set_parameters`` loads them back matching keys and dtypes.
    """

    def __init__(
        self, module: "torch.nn.Module", descriptor: Optional[str] = None
    ) -> None:
        """Wrap an existing torch module.

        Args:
            module (torch.nn.Module): The torch module whose ``state_dict``
                this model exposes and updates. Held by reference, not
                copied.
            descriptor (Optional[str]): A tag naming the model's
                architecture, stamped onto contributions for
                heterogeneity-aware aggregation. Defaults to ``None``, which
                uses ``type(module).__name__``.
        """
        self.module = module
        self._descriptor = descriptor or type(module).__name__

    def get_parameters(self) -> Parameters:
        """Project the module's current state into an agnostic bundle.

        Returns:
            Parameters: The module's ``state_dict`` values, in order, as
            detached numpy arrays (moved to CPU), tagged with
            ``tensor_type="torch"``.
        """
        tensors = [v.detach().cpu().numpy() for v in self.module.state_dict().values()]
        return Parameters(tensors, tensor_type="torch")

    def set_parameters(self, params: Parameters) -> None:
        """Load an agnostic parameter bundle back into the module's state.

        Matches ``params.tensors`` to the module's ``state_dict`` entries by
        position (i.e. ``state_dict`` iteration order) and casts each array
        to the corresponding existing parameter's dtype before loading.

        Args:
            params (Parameters): The parameters to load; must supply exactly
                one tensor per entry in the module's ``state_dict``, in the
                same order and with compatible shapes.
        """
        state_dict = self.module.state_dict()
        new_state = {
            key: torch.as_tensor(array, dtype=current.dtype)
            for (key, current), array in zip(state_dict.items(), params.tensors)
        }
        self.module.load_state_dict(new_state)

    @property
    def descriptor(self) -> str:
        """str: The tag naming this model's architecture."""
        return self._descriptor
