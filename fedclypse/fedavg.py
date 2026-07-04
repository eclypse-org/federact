# -*- coding: utf-8 -*-
from __future__ import annotations

import asyncio
from typing import Any, Callable, List, Optional

from fedclypse.aggregation import fedavg
from fedclypse.contribution import Contribution
from fedclypse.entity import Entity
from fedclypse.parameters import Parameters
from fedclypse.selection import select_all


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
        super().__init__(
            entity_id, model_factory=model_factory, data=data, **service_kwargs
        )
        self.rounds = rounds
        self.selection = selection

    def aggregate(self, contributions: List[Contribution]) -> Parameters:
        return fedavg(contributions)

    async def run(self) -> None:
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
        """The local training step. The reference does nothing (returns the received
        parameters unchanged); override to run real local optimization."""
        return params

    @property
    def num_examples(self) -> float:
        return float(len(self.dataset)) if self.dataset is not None else 1.0

    async def run(self) -> None:
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
