# -*- coding: utf-8 -*-
"""Reference FedAvg behaviours: a synchronous server and a reactive client.

``FedAvgServer`` and ``FedAvgClient`` are the canonical ``Entity`` subclasses
demonstrating the star-topology FedAvg pattern: each round the server selects
a cohort of client neighbours, broadcasts the current global parameters,
collects one ``Contribution`` per cohort member, and replaces the global
model with their FedAvg (num-examples-weighted mean, see
``fedclypse.aggregation.fedavg``); the client is a reactive learner that
answers whichever round it is invited to.
"""
from __future__ import annotations

import asyncio
from typing import Any, Callable, List, Optional

from fedclypse.aggregation import fedavg
from fedclypse.contribution import Contribution
from fedclypse.entity import Entity
from fedclypse.parameters import Parameters
from fedclypse.selection import select_all

__all__ = ["FedAvgServer", "FedAvgClient"]


class FedAvgServer(Entity):
    """Reference synchronous FedAvg aggregator.

    Each round: select a cohort of client neighbours, send them the current global
    parameters, wait for exactly one ``Contribution`` per cohort member, and replace
    the global model with their FedAvg (num-examples-weighted mean). After ``rounds``
    rounds it PARKS (loops on ``self.running``) so the driver's ``sim.stop()`` cancels
    it cleanly (returning early would print a noisy "Event loop is closed" traceback).
    """

    def __init__(
        self,
        entity_id: str,
        *,
        model_factory: Optional[Callable[[], Any]] = None,
        data: Any = None,
        rounds: int,
        selection: Callable[[List[str]], List[str]] = select_all,
        **service_kwargs: Any,
    ) -> None:
        """Initialize the server with its round budget and cohort policy.

        Args:
            entity_id (str): The entity's unique id within the application.
            model_factory (Optional[Callable[[], Any]]): A zero-argument,
                picklable callable that builds the global model; forwarded to
                ``Entity``. Defaults to ``None`` (no model).
            data (Any): Forwarded to ``Entity`` as its ``data`` argument. The
                reference server has no use for local data, so this is
                normally left ``None``.
            rounds (int): The number of FedAvg rounds to run before parking.
            selection (Callable[[List[str]], List[str]]): The cohort-selection
                policy invoked each round with the current neighbour ids; see
                ``fedclypse.selection``. Defaults to ``select_all``.
            **service_kwargs (Any): Forwarded to ``Entity.__init__``.
        """
        super().__init__(
            entity_id, model_factory=model_factory, data=data, **service_kwargs
        )
        self.rounds = rounds
        self.selection = selection

    def aggregate(self, contributions: List[Contribution]) -> Parameters:
        """Combine a round's collected contributions into new global parameters.

        Args:
            contributions (List[Contribution]): One contribution per cohort
                member, as collected by ``run``.

        Returns:
            Parameters: The FedAvg (num-examples-weighted mean) of the
            contributions' payloads; see ``fedclypse.aggregation.fedavg``.

        Raises:
            IncompatibleContributionsError: If a payload is not
                ``Parameters``, shapes mismatch, or model descriptors are
                heterogeneous. See ``fedclypse.aggregation.fedavg``.
            ValueError: If ``contributions`` is empty or the total weight is
                zero.
        """
        return fedavg(contributions)

    async def run(self) -> None:
        """Drive ``rounds`` synchronous FedAvg rounds, then park.

        Each round selects a cohort via ``self.selection``, broadcasts the
        current global parameters to it, blocks until exactly one message
        has been received from every cohort member, aggregates the results
        with ``aggregate``, and advances ``self.round``. A round whose cohort
        is empty still advances ``self.round`` but skips the
        send/receive/aggregate cycle. Once ``rounds`` rounds have run, loops
        on ``self.running`` instead of returning, so the driver's
        ``sim.stop()`` cancels the entity cleanly (via task cancellation)
        rather than the coroutine returning while the event loop is being
        torn down.
        """
        clients = await self.neighbours()
        for r in range(self.rounds):
            cohort = self.selection(clients)
            if not cohort:
                self.round = r + 1
                continue
            params = self.model.get_parameters()
            await self.send(
                cohort,
                round=r,
                tensors=[t.copy() for t in params.tensors],
                model_descriptor=self.model.descriptor,
            )
            collected: List[Contribution] = []
            for _ in range(len(cohort)):
                msg = await self.receive()
                collected.append(
                    Contribution(
                        payload=Parameters(msg["tensors"]),
                        source=msg["sender_id"],
                        version=msg["round"],
                        model_descriptor=msg.get("model_descriptor"),
                        weight=msg["weight"],
                    )
                )
            self.model.set_parameters(self.aggregate(collected))
            self.round = r + 1
        while self.running:
            await asyncio.sleep(0.05)


class FedAvgClient(Entity):
    """Reference FedAvg client: a reactive learner.

    Loops: receive the server's parameters, run ``local_update`` (the training seam —
    identity by default; override to train), adopt the result, and send it back as a
    ``Contribution`` weighted by its number of local examples. Being reactive (no fixed
    round count) it composes with server-side selection: an unselected round leaves it
    blocked in ``receive()`` until it is picked again, and shutdown cancels that wait.
    """

    def local_update(self, params: Parameters) -> Parameters:
        """Run the local training step.

        The reference implementation does nothing (returns the received
        parameters unchanged); override to run real local optimization.

        Args:
            params (Parameters): The global parameters received from the
                server for this round.

        Returns:
            Parameters: The updated (locally trained) parameters to adopt
            and send back. Defaults to returning ``params`` unchanged.
        """
        return params

    @property
    def num_examples(self) -> float:
        """float: This client's FedAvg weight, its number of local examples.

        Reads ``len(self.dataset)`` once the client has been deployed and
        its data materialized; ``1.0`` before deployment or when no data was
        configured.
        """
        return float(len(self.dataset)) if self.dataset is not None else 1.0

    async def run(self) -> None:
        """React indefinitely to the server: receive, update, adopt, reply.

        Blocks in ``receive()`` for the server's parameters, runs
        ``local_update``, loads the result into ``self.model``, advances
        ``self.round``, and sends a ``Contribution`` (weighted by
        ``num_examples``) back to the server. Runs until ``self.running``
        goes ``False``; a stop request cancels the coroutine while it is
        blocked in ``receive()``.
        """
        while self.running:
            msg = await self.receive()
            received = Parameters(msg["tensors"])
            updated = self.local_update(received)
            self.model.set_parameters(updated)
            self.round = msg["round"] + 1
            await self.send(
                "server",
                round=msg["round"],
                tensors=[t.copy() for t in updated.tensors],
                weight=self.num_examples,
                model_descriptor=self.model.descriptor,
            )
