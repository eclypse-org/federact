# -*- coding: utf-8 -*-
from __future__ import annotations

from typing import Any, Callable, List, Optional, Union

from eclypse.remote.service import Service


class Entity(Service):
    """A situated federated-learning participant: an eclypse ``Service`` whose
    situatedness (node, neighbourhood, data) and state (model) are first-class, and
    whose behaviour is written in ``run()`` (or ``step()``) using fedclypse tools.

    It is a thin shell: ``on_deploy`` builds the model and materializes the local data
    **on the worker** (only the picklable ``model_factory`` and the ``ClientData``
    indices travel), and the ``mpi`` helpers wrap eclypse's message passing.
    """

    def __init__(
        self,
        entity_id: str,
        *,
        model_factory: Optional[Callable[[], Any]] = None,
        data: Optional[Any] = None,
        **service_kwargs: Any,
    ) -> None:
        super().__init__(entity_id, **service_kwargs)
        self._model_factory = model_factory
        self._client_data = data
        self.model: Any = None
        self.dataset: Any = None
        self.round: int = 0

    # ---- lifecycle (runs on the worker) ----
    def on_deploy(self) -> None:
        if self._model_factory is not None:
            self.model = self._model_factory()
        if self._client_data is not None:
            self.dataset = self._client_data.materialize()
        self.round = 0

    # ---- situatedness / communication (thin wrappers over eclypse mpi) ----
    async def neighbours(self) -> List[str]:
        return await self.mpi.get_neighbors()

    async def send(self, recipient_ids: Union[str, List[str]], **body: Any) -> None:
        await self.mpi.send(recipient_ids, body)

    async def broadcast(self, **body: Any) -> None:
        await self.mpi.bcast(body)

    async def receive(self) -> dict:
        # eclypse injects ``sender_id`` into the returned body dict.
        return await self.mpi.recv()

    # ---- behaviour seam ----
    async def step(self) -> None:
        raise NotImplementedError(
            f"{type(self).__name__} must override run() or step()"
        )
