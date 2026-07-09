# -*- coding: utf-8 -*-
"""Composable Aggregator/Learner role behaviours over the situated Entity.

An Entity's roles are the mixins it subclasses. Each role contributes to a
shared ``run()`` skeleton: a single dispatch loop routes messages by ``kind``
(``"request"`` -> the Learner, ``"reply"`` -> the Aggregator), and the
Aggregator additionally runs a proactive driver. A star server is an
``Aggregator``; a client a ``Learner``; a hierarchical mid-node is both -- the
two loops run concurrently on the entity's single asyncio event loop, sharing
``self.model`` cooperatively.

On a both-roles node the two loops also share a single ``self.round`` (written
by both handlers), and the current design propagates values **upward** only: a
mid-node reports its aggregate to its parent but does not forward the parent's
model down to its children within a round. Two-way hierarchical aggregation and
staleness-weighted async on a both-roles node need per-role ``model``/``round``
separation and are deferred to a later roadmap item.
"""
from __future__ import annotations

import asyncio
from typing import Any, Callable, List, Optional

from fedclypse.aggregation import fedavg
from fedclypse.core.contribution import Contribution
from fedclypse.core.entity import Entity
from fedclypse.core.parameters import Parameters
from fedclypse.selection import select_all
from fedclypse.synchronization import Synchronizer, Synchronous

__all__ = ["Roles", "Aggregator", "Learner"]


class Roles(Entity):
    """Base that composes each subclassed role's loop into ``Entity.run()``.

    ``run()`` gathers one shared dispatch loop plus every role's proactive
    driver on the entity's single event loop, then parks (the dispatch loop
    blocks in ``receive()`` until teardown cancels it). Subclasses cooperate
    via ``super()``: ``_handle`` is a ``kind``-routed chain of responsibility
    and ``_role_drivers`` concatenates each role's drivers.
    """

    def _role_drivers(self) -> List[Any]:
        """Collect the proactive driver coroutines contributed by each role.

        Returns:
            List[Any]: The role driver coroutine objects to run concurrently
            with the dispatch loop. Empty for a purely reactive entity;
            subclasses append via ``super()._role_drivers() + [...]``.
        """
        return []

    async def _handle(self, kind: Optional[str], message: dict) -> None:
        """Route a received message to the matching role handler by ``kind``.

        The base is the end of the chain: an unclaimed ``kind`` is dropped.
        Role mixins override this, handle their own ``kind``, and delegate
        the rest to ``super()._handle``.

        Args:
            kind (Optional[str]): The message's ``kind`` tag (``"request"`` or
                ``"reply"``), or ``None`` when absent.
            message (dict): The received message body (eclypse has injected
                ``sender_id``).
        """
        return None

    async def _dispatch_loop(self) -> None:
        """Receive messages until teardown, routing each by ``kind``.

        This is also the entity's park: it blocks in ``receive()`` between
        messages, so ``run()`` never returns on its own and ``sim.stop()``
        cancels it cleanly.
        """
        while self.running:
            message = await self.receive()
            await self._handle(message.get("kind"), message)

    async def run(self) -> None:
        """Run the dispatch loop plus every role driver until stopped.

        Gathers the shared ``_dispatch_loop`` with every coroutine from
        ``_role_drivers``. A driver may return after finishing its budget; the
        dispatch loop keeps the entity parked (blocked in ``receive()``) until
        teardown cancels the gather. Teardown may print a benign "Task was
        destroyed but it is pending" warning (eclypse closes the loop before
        the gather's children finish cancelling) -- it does not affect
        convergence or the IDLE status, and is distinct from the "Event loop is
        closed" traceback caused by returning early instead of parking.
        """
        drivers = self._role_drivers()
        await asyncio.gather(self._dispatch_loop(), *drivers)


class Learner(Roles):
    """Reactive role: on each ``request``, train locally and reply to the sender.

    A Learner has no driver -- it only answers requests. On a ``request`` it
    runs ``local_update`` on the received parameters, adopts the result into
    ``self.model``, and replies a ``Contribution`` (weighted by
    ``num_examples``) to the requester (``message["sender_id"]``), not a
    hard-coded id, so it composes in any topology.
    """

    def local_update(self, params: Parameters) -> Parameters:
        """Run the local training step (identity by default; override to train).

        Args:
            params (Parameters): The parameters received from the requester.

        Returns:
            Parameters: The updated parameters to adopt and reply with.
        """
        return params

    @property
    def num_examples(self) -> float:
        """float: This learner's contribution weight (its number of examples).

        ``len(self.dataset)`` once deployed, else ``1.0``.
        """
        return float(len(self.dataset)) if self.dataset is not None else 1.0

    async def _handle(self, kind: Optional[str], message: dict) -> None:
        """Handle a ``request`` by training and replying; delegate the rest.

        Args:
            kind (Optional[str]): The message ``kind``.
            message (dict): The received message body.
        """
        if kind == "request":
            received = Parameters(message["tensors"])
            updated = self.local_update(received)
            self.model.set_parameters(updated)
            self.round = message["round"] + 1
            await self.send(
                message["sender_id"],
                kind="reply",
                round=message["round"],
                tensors=[t.copy() for t in updated.tensors],
                weight=self.num_examples,
                model_descriptor=self.model.descriptor,
            )
            return None
        return await super()._handle(kind, message)


class Aggregator(Roles):
    """Proactive role: each fire, gather a Synchronizer-gated cohort and aggregate.

    The driver kicks the cohort ONCE, then loops: await the next fire signal,
    re-select, and ``disseminate`` (the sole re-request) until ``rounds``
    fires. The reply handler buffers contributions and, when
    ``Synchronizer.ready``, aggregates and re-arms the buffer SYNCHRONOUSLY
    (before the next message is dispatched) and pushes a fire token onto a
    Queue. Using a Queue (not an Event) means concurrent fires can never
    coalesce -- essential for ``BufferedAsync``, where a single kickoff can
    trigger several fires before the driver wakes.
    """

    def __init__(
        self,
        entity_id: str,
        *,
        rounds: int,
        selection: Callable[[List[str]], List[str]] = select_all,
        synchronizer: Optional[Synchronizer] = None,
        rule: Callable[[List[Contribution]], Parameters] = fedavg,
        **kwargs: Any,
    ) -> None:
        """Initialize the aggregator with its round budget and composed mechanics.

        No asyncio primitive is created here: the entity is pickled to the
        worker, so the fire Queue is created lazily in the driver on the
        worker's loop.

        Args:
            entity_id (str): The entity's unique id.
            rounds (int): The number of aggregation fires before the driver stops.
            selection (Callable[[List[str]], List[str]]): Cohort policy over
                neighbours. Defaults to ``select_all``.
            synchronizer (Optional[Synchronizer]): The aggregation trigger.
                Defaults to ``Synchronous()``.
            rule (Callable[[List[Contribution]], Parameters]): The aggregation
                rule. Defaults to ``fedavg``.
            **kwargs (Any): Forwarded to ``Entity``/``Roles`` (and, for a
                both-roles entity, on to ``Learner``).
        """
        super().__init__(entity_id, **kwargs)
        self.rounds = rounds
        self.selection = selection
        self.synchronizer = synchronizer or Synchronous()
        self.rule = rule
        self._buffer: List[Contribution] = []
        self._cohort: List[str] = []
        self._fires: Optional[asyncio.Queue] = None

    def __getstate__(self) -> dict:
        """Exclude the runtime-only fire Queue from pickling.

        eclypse pickles the service on both deploy and undeploy. An
        ``asyncio.Queue`` holds weakrefs to the event loop and is unpicklable,
        which surfaces as a ``cannot pickle 'weakref.ReferenceType'`` error in
        eclypse's undeploy path. It is dropped here (the driver recreates it on
        the worker loop), keeping teardown clean.

        Returns:
            dict: This entity's ``__dict__`` with ``_fires`` nulled.
        """
        state = dict(self.__dict__)
        state["_fires"] = None
        return state

    def aggregate(self, contributions: List[Contribution]) -> Parameters:
        """Combine contributions, folding staleness into each weight, via ``rule``.

        Args:
            contributions (List[Contribution]): The fired batch.

        Returns:
            Parameters: The aggregated parameters.
        """
        reweighted = [
            Contribution(
                payload=c.payload,
                source=c.source,
                version=c.version,
                model_descriptor=c.model_descriptor,
                weight=c.weight * self.synchronizer.staleness_weight(c, self.round),
            )
            for c in contributions
        ]
        return self.rule(reweighted)

    async def disseminate(self, contributors: List[str], cohort: List[str]) -> None:
        """Send the updated model onward after a fire (customizable seam).

        Defaults to the contributors who just fired (correct for async;
        degenerates to the full cohort under ``Synchronous``). Override for
        other schemes.

        Args:
            contributors (List[str]): The sources whose replies just fired.
            cohort (List[str]): The freshly (re)selected cohort for the next round.
        """
        await self._request(contributors)

    async def _request(self, recipients: List[str]) -> None:
        """Send the current model as a ``request`` to ``recipients`` (if any)."""
        params = self.model.get_parameters()
        if recipients:
            await self.send(
                recipients,
                kind="request",
                round=self.round,
                tensors=[t.copy() for t in params.tensors],
                model_descriptor=self.model.descriptor,
            )

    async def _handle(self, kind: Optional[str], message: dict) -> None:
        """Buffer a ``reply``, aggregating + re-arming synchronously on fire.

        Args:
            kind (Optional[str]): The message ``kind``.
            message (dict): The received message body.
        """
        if kind == "reply":
            self._buffer.append(
                Contribution(
                    payload=Parameters(message["tensors"]),
                    source=message["sender_id"],
                    version=message["round"],
                    model_descriptor=message.get("model_descriptor"),
                    weight=message["weight"],
                )
            )
            if self.synchronizer.ready(self._buffer, self._cohort):
                batch = self._buffer
                self._buffer = []  # re-arm BEFORE the next message is dispatched
                self.model.set_parameters(self.aggregate(batch))
                self.round += 1
                self._fires.put_nowait([c.source for c in batch])
            return None
        return await super()._handle(kind, message)

    def _role_drivers(self) -> List[Any]:
        """Append the aggregator driver to any inherited role drivers."""
        return super()._role_drivers() + [self._aggregator_driver()]

    async def _aggregator_driver(self) -> None:
        """Kick the cohort once, then fire->reselect->disseminate for ``rounds``.

        Creates the fire Queue on the running (worker) loop. The kickoff is the
        only top-of-loop request; thereafter ``disseminate`` is the sole
        re-request, so each round sends exactly one request per recipient and
        the buffer holds exactly one window's replies.
        """
        self._fires = asyncio.Queue()
        self._buffer = []
        self._cohort = self.selection(await self.neighbours())
        await self._request(self._cohort)
        fires = 0
        while self.running and fires < self.rounds:
            contributors = await self._fires.get()
            fires += 1
            self._cohort = self.selection(await self.neighbours())
            if fires < self.rounds:
                await self.disseminate(contributors, self._cohort)
