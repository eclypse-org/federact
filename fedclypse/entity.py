# -*- coding: utf-8 -*-
"""The Entity base class: a situated eclypse Service for federated learning.

``Entity`` subclasses eclypse's ``Service`` and makes an FL participant's
situatedness (node, neighbourhood, data) and state (model) first-class,
leaving behaviour to be written in ``run()``/``step()`` using the ``mpi``
helpers this class wraps around eclypse's message passing.
"""
from __future__ import annotations

from typing import Any, Callable, List, Optional, Union

from eclypse.remote.service import Service

__all__ = ["Entity"]


class Entity(Service):
    """A situated federated-learning participant.

    An eclypse ``Service`` whose situatedness (node, neighbourhood, data) and
    state (model) are first-class, and whose behaviour is written in
    ``run()`` (or ``step()``) using fedclypse tools. It is a thin shell:
    ``on_deploy`` builds the model and materializes the local data **on the
    worker** (only the picklable ``model_factory`` and the ``ClientData``
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
        """Initialize the entity in a neutral, undeployed state.

        No model is built and no data is materialized here — that is
        deferred to ``on_deploy``, which runs on the worker once the entity
        is placed, so only the picklable ``model_factory``/``data``
        references need to cross the wire.

        Args:
            entity_id (str): The entity's unique id within the application.
            model_factory (Optional[Callable[[], Any]]): A zero-argument,
                picklable callable that builds this entity's model; invoked
                from ``on_deploy``. Defaults to ``None`` (no model).
            data (Optional[Any]): A ``ClientData`` (or compatible) descriptor
                whose ``materialize()`` is called from ``on_deploy`` to build
                this entity's local dataset. Defaults to ``None`` (no data).
            **service_kwargs (Any): Forwarded to
                ``eclypse.remote.service.Service.__init__``.
        """
        super().__init__(entity_id, **service_kwargs)
        self._model_factory = model_factory
        self._client_data = data
        self.model: Any = None
        self.dataset: Any = None
        self.round: int = 0

    # ---- lifecycle (runs on the worker) ----
    def on_deploy(self) -> None:
        """Build the model and materialize the local data, on the worker.

        Called by eclypse once the entity has been placed on a node. Builds
        ``self.model`` from ``model_factory`` and ``self.dataset`` from
        ``data.materialize()`` when they were provided, and resets
        ``self.round`` to ``0``. Safe to call when neither was provided:
        ``self.model``/``self.dataset`` are simply left ``None``.
        """
        if self._model_factory is not None:
            self.model = self._model_factory()
        if self._client_data is not None:
            self.dataset = self._client_data.materialize()
        self.round = 0

    # ---- situatedness / communication (thin wrappers over eclypse mpi) ----
    async def neighbours(self) -> List[str]:
        """List the ids of this entity's topology neighbours.

        Returns:
            List[str]: The neighbour entity ids, as reported by eclypse's
            infrastructure/topology manager.
        """
        return await self.mpi.get_neighbors()

    async def send(self, recipient_ids: Union[str, List[str]], **body: Any) -> None:
        """Send a message body to one or more neighbour entities.

        Args:
            recipient_ids (Union[str, List[str]]): The neighbour id or ids to route to.
            **body (Any): Arbitrary picklable key/value pairs forming the message body;
                eclypse injects the sender id under ``sender_id`` on receipt.
        """
        await self.mpi.send(recipient_ids, body)

    async def broadcast(self, **body: Any) -> None:
        """Send a message body to every neighbour entity.

        Args:
            **body (Any): Arbitrary picklable key/value pairs forming the message body;
                eclypse injects the sender id under ``sender_id`` on receipt.
        """
        await self.mpi.bcast(body)

    async def receive(self) -> dict:
        """Await the next incoming message body.

        Returns:
            dict: The sender's message body, with the sender's id injected by
            eclypse under the ``sender_id`` key.
        """
        # eclypse injects ``sender_id`` into the returned body dict.
        return await self.mpi.recv()

    # ---- behaviour seam ----
    async def step(self) -> None:
        """The entity's default per-step behaviour (unimplemented).

        Subclasses must override ``run()`` or ``step()`` with their FL logic;
        the base ``Entity`` provides neither.

        Raises:
            NotImplementedError: Always, unless overridden by a subclass.
        """
        raise NotImplementedError(
            f"{type(self).__name__} must override run() or step()"
        )
