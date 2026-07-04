# -*- coding: utf-8 -*-
from __future__ import annotations

from typing import Optional

import torch

from fedclypse.model import Model
from fedclypse.parameters import Parameters


class TorchModel(Model):
    """Adapts a ``torch.nn.Module`` to the agnostic ``Model`` interface.

    ``Parameters`` are the module's ``state_dict`` values as numpy arrays, in
    ``state_dict`` order; ``set_parameters`` loads them back matching keys and dtypes.
    """

    def __init__(
        self, module: "torch.nn.Module", descriptor: Optional[str] = None
    ) -> None:
        self.module = module
        self._descriptor = descriptor or type(module).__name__

    def get_parameters(self) -> Parameters:
        tensors = [v.detach().cpu().numpy() for v in self.module.state_dict().values()]
        return Parameters(tensors, tensor_type="torch")

    def set_parameters(self, params: Parameters) -> None:
        state_dict = self.module.state_dict()
        new_state = {
            key: torch.as_tensor(array, dtype=current.dtype)
            for (key, current), array in zip(state_dict.items(), params.tensors)
        }
        self.module.load_state_dict(new_state)

    @property
    def descriptor(self) -> str:
        return self._descriptor
