# -*- coding: utf-8 -*-
"""Reference FedAvg behaviours: a thin, synchronous instance of the general roles.

``FedAvgServer`` and ``FedAvgClient`` demonstrate the star-topology FedAvg
pattern as the smallest possible specialization of
``fedclypse.schemes.roles``: ``FedAvgServer`` is an ``Aggregator`` that fixes
synchronous aggregation (``Synchronous``) with the FedAvg rule
(``fedclypse.aggregation.fedavg``), and ``FedAvgClient`` is a ``Learner`` with
no overrides at all. Both classes exist only to pin defaults and document the
resulting behaviour; the actual round-driving, buffering, and reply logic
lives in ``Aggregator``/``Learner``.
"""
from __future__ import annotations

from typing import Any, Callable, List, Optional

from fedclypse.aggregation import fedavg
from fedclypse.schemes.roles import Aggregator, Learner
from fedclypse.selection import select_all
from fedclypse.synchronization import Synchronous

__all__ = ["FedAvgServer", "FedAvgClient"]


class FedAvgServer(Aggregator):
    """Reference synchronous FedAvg aggregator.

    A thin ``Aggregator`` that fixes ``synchronizer=Synchronous()`` (a
    barrier: fire once every selected cohort member has replied) and
    ``rule=fedavg`` (the num-examples-weighted mean). Every round: select a
    cohort of client neighbours via ``selection``, send them the current
    global parameters, wait for exactly one reply per cohort member, and
    replace the global model with their FedAvg. All of this -- the driver
    loop, the reply buffering, and ``aggregate`` itself -- is inherited
    unchanged from ``Aggregator``; under a synchronous barrier every
    contribution's staleness weight is ``1.0``, so ``aggregate`` reduces
    exactly to ``fedavg(contributions)``.
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

        ``synchronizer`` and ``rule`` are not exposed here: this class fixes
        them to ``Synchronous()`` and ``fedavg`` respectively, which is what
        makes it "FedAvg" rather than the general ``Aggregator``.

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
            entity_id,
            rounds=rounds,
            selection=selection,
            synchronizer=Synchronous(),
            rule=fedavg,
            model_factory=model_factory,
            data=data,
            **service_kwargs,
        )


class FedAvgClient(Learner):
    """Reference FedAvg client: a reactive learner, unchanged from ``Learner``.

    Being a ``Learner``, it inherits ``local_update`` (identity by default;
    override to train), ``num_examples`` (its FedAvg weight), and the
    reactive ``_handle`` that, on each request, trains and replies to the
    requester (``message["sender_id"]``) -- in a star topology that requester
    is the server, matching the old hard-coded ``"server"`` reply, but
    without assuming a star: the same client composes in any topology.
    """
