"""The experiment layer: compose a Situation and a Behaviour into a runnable study.

An FL experiment factorizes into a **Situation** (who the nodes are, their roles,
data, models, and physical layout) and a **Behaviour** (how each role behaves). The
two are independent values joined by ``run``: a Situation *assigns* roles; a Behaviour
*implements* them. ``run(situation, behaviour, rounds=..., seed=...)`` instantiates the
nodes, assembles the eclypse Simulation over them, drives it, and returns the collected
trajectory (a metric ``History``). ``rounds`` is the run horizon, not part of the
Behaviour, so a sweep is a plain ``map`` over situations x behaviours x seeds -- there
is no experiment-manager object.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import (
    TYPE_CHECKING,
    Any,
)

import networkx as nx

from fedclypse.deployment.topology import from_graph
from fedclypse.runtime.simulation import (
    build_simulation,
    run_federation,
)
from fedclypse.schemes.fedavg import (
    FedAvgClient,
    FedAvgServer,
)
from fedclypse.selection import select_all

if TYPE_CHECKING:
    from collections.abc import Callable

    from eclypse.graph import Infrastructure
    from eclypse.placement.strategies import PlacementStrategy

    from fedclypse.core.entity import Entity
    from fedclypse.data.source import ClientData
    from fedclypse.runtime.metrics import History

__all__ = [
    "Behaviour",
    "NodeSpec",
    "Situation",
    "fedavg_behaviour",
    "run",
    "star_situation",
]


@dataclass
class NodeSpec:
    """A situated node, minus its behaviour.

    Carries the logical, data, and model facts for one participant; the Behaviour
    supplies the ``run()`` for its ``role``.

    Attributes:
        id (str): The node's unique id within the experiment.
        role (str): The role label the Behaviour implements for this node (e.g.
            ``"server"`` / ``"client"``).
        data (Optional[ClientData]): This node's local data, or ``None`` for a
            dataless node (e.g. a pure server). Defaults to ``None``.
        model_factory (Optional[Callable[[], Any]]): A zero-argument, picklable
            callable building this node's model (per-node factories express model
            heterogeneity). Defaults to ``None``.
    """

    id: str
    role: str
    data: ClientData | None = None
    model_factory: Callable[[], Any] | None = None


@dataclass
class Situation:
    """A federation's Logical x Physical layout as one inspectable value.

    ``nodes`` is ordered and ``graph`` uses integer nodes ``0..len(nodes)-1`` (the
    ``fedclypse.deployment.from_graph`` convention), so ``nodes[i]`` sits at graph
    node ``i`` and ``run`` assembles the Application by ``from_graph(entities, graph)``.

    Attributes:
        nodes (List[NodeSpec]): The participants, ordered to match ``graph``'s
            integer node indices.
        graph (nx.Graph): The communication topology over integer node indices
            ``0..len(nodes)-1``.
        infrastructure (Optional[Infrastructure]): The physical infrastructure to
            place the federation on. Defaults to ``None`` (``build_simulation`` builds
            its default symmetric star).
        placement (Optional[PlacementStrategy]): The placement strategy. Defaults to
            ``None`` (``build_simulation``'s default).
    """

    nodes: list[NodeSpec]
    graph: nx.Graph
    infrastructure: Infrastructure | None = None
    placement: PlacementStrategy | None = None


@dataclass
class Behaviour:
    """How each role behaves: a mapping from role label to an entity builder.

    Each builder is a callable ``(node, rounds) -> Entity`` that ``run`` invokes with
    the node's situational facts and the run horizon. Keeping the builder
    ``(node, rounds) -> Entity`` makes a Behaviour the pure *algorithm* -- ``rounds``
    is the run horizon, not part of the Behaviour (aggregator builders consume it;
    learner builders accept and ignore it).

    Attributes:
        roles (Dict[str, Callable[[NodeSpec, int], Entity]]): Role label -> entity
            builder.
    """

    roles: dict[str, Callable[[NodeSpec, int], Entity]]


def run(
    situation: Situation,
    behaviour: Behaviour,
    *,
    rounds: int,
    seed: int = 0,
    mode: str = "emulation",
    metrics: list | None = None,
    step_delay: float = 0.5,
    grace: float = 1.0,
) -> History:
    """Join a Situation and a Behaviour, run the federation, and return its trajectory.

    Checks the role seam (every assigned role must have a builder), instantiates each
    node's entity from its role's builder, assembles the Application over the
    situation's comm graph, builds the Simulation on the situation's physical layout,
    and drives it to completion. Pure in its inputs (plus ``seed``): a sweep is
    ``[run(s, b, rounds=R, seed=k) for ...]``.

    Args:
        situation (Situation): The logical + physical layout.
        behaviour (Behaviour): The role -> entity-builder map.
        rounds (int): The experiment horizon; passed to the entity builders (the
            aggregator's fire budget) and to the simulation.
        seed (int): Seed for the default infrastructure/placement/config. Defaults
            to ``0``.
        mode (str): ``"emulation"`` (Ray-backed, real execution) or ``"simulation"``
            (placement/comm/timing only). Defaults to ``"emulation"``.
        metrics (Optional[List]): ``@metric.service`` observables to collect.
            Defaults to ``None``.
        step_delay (float): Seconds slept after each manual ``step()``. Defaults to
            ``0.5``.
        grace (float): Seconds slept after the last step before ``stop()``. Defaults
            to ``1.0``.

    Returns:
        History: The collected metric trajectory, read after the run completes.

    Raises:
        ValueError: If the situation assigns a role the behaviour does not provide.
    """
    assigned = {node.role for node in situation.nodes}
    missing = assigned - set(behaviour.roles)
    if missing:
        raise ValueError(
            f"Behaviour provides roles {sorted(behaviour.roles)} but the situation "
            f"assigns unmatched role(s) {sorted(missing)}"
        )
    entities = [behaviour.roles[node.role](node, rounds) for node in situation.nodes]
    application = from_graph(entities, situation.graph)
    simulation = build_simulation(
        application,
        infrastructure=situation.infrastructure,
        rounds=rounds,
        seed=seed,
        mode=mode,
        metrics=metrics,
        placement=situation.placement,
    )
    return run_federation(simulation, rounds=rounds, step_delay=step_delay, grace=grace)


def star_situation(
    server_id: str,
    client_ids: list[str],
    *,
    data: dict[str, ClientData] | None = None,
    model_factory: Callable[[], Any] | None = None,
    infrastructure: Infrastructure | None = None,
    placement: PlacementStrategy | None = None,
) -> Situation:
    """Build a client-server star Situation: a ``server`` plus ``client`` leaves.

    Args:
        server_id (str): The server node's id (role ``"server"``, dataless).
        client_ids (List[str]): The client node ids (role ``"client"``).
        data (Optional[Dict[str, ClientData]]): Per-client local data, keyed by
            client id; a client absent from the map is dataless. Defaults to ``None``
            (all clients dataless).
        model_factory (Optional[Callable[[], Any]]): The model factory shared by every
            node (per-node models require building ``NodeSpec`` values directly).
            Defaults to ``None``.
        infrastructure (Optional[Infrastructure]): Physical infrastructure. Defaults
            to ``None``.
        placement (Optional[PlacementStrategy]): Placement strategy. Defaults to
            ``None``.

    Returns:
        Situation: A star situation whose ordered ``nodes`` are ``[server, *clients]``
        and whose ``graph`` is ``nx.star_graph(len(client_ids))`` (center node 0 = the
        server).
    """
    data = data or {}
    nodes = [NodeSpec(server_id, "server", data=None, model_factory=model_factory)]
    nodes += [
        NodeSpec(cid, "client", data=data.get(cid), model_factory=model_factory)
        for cid in client_ids
    ]
    graph = nx.star_graph(len(client_ids))
    return Situation(
        nodes=nodes, graph=graph, infrastructure=infrastructure, placement=placement
    )


def fedavg_behaviour(
    *, selection: Callable[[list[str]], list[str]] = select_all
) -> Behaviour:
    """Build the FedAvg Behaviour: a ``server`` aggregator + ``client`` learners.

    Args:
        selection (Callable[[List[str]], List[str]]): The server's cohort-selection
            policy. Defaults to ``select_all``.

    Returns:
        Behaviour: Roles ``"server"`` -> ``FedAvgServer`` (rounds threaded from the
        run) and ``"client"`` -> ``FedAvgClient`` (accepts and ignores ``rounds``).
    """

    def server(node: NodeSpec, rounds: int) -> Entity:
        return FedAvgServer(
            node.id,
            rounds=rounds,
            selection=selection,
            model_factory=node.model_factory,
            data=node.data,
        )

    def client(node: NodeSpec, rounds: int) -> Entity:  # noqa: ARG001
        return FedAvgClient(node.id, model_factory=node.model_factory, data=node.data)

    return Behaviour(roles={"server": server, "client": client})
