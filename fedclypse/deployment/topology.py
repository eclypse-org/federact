# -*- coding: utf-8 -*-
"""Logical communication-topology builders for federations.

Each builder returns an eclypse ``Application`` whose edges are the FL communication
graph — who exchanges with whom. ``get_neighbors`` follows this graph, so it is the
sole determinant of comm topology; the physical infrastructure is decoupled
(see ``fedclypse.placement``). ``from_graph`` is the general primitive; ``star``,
``ring`` and ``complete`` are thin conveniences over it. Hierarchical, random and
realistic topologies are obtained by passing the corresponding ``networkx`` graph to
``from_graph``, or by mirroring an eclypse infrastructure (``fedclypse.placement.mirror``).
"""
from __future__ import annotations

from typing import Sequence

import networkx as nx
from eclypse.graph import Application

from fedclypse.core.entity import Entity

__all__ = ["from_graph", "star", "ring", "complete"]


def from_graph(
    entities: Sequence[Entity],
    graph: nx.Graph,
    application_id: str = "fedclypse",
) -> Application:
    """Build an Application whose comm edges mirror an arbitrary networkx graph.

    Args:
        entities (Sequence[Entity]): The federation's entities. ``entities[i]`` is
            placed at graph node ``i``.
        graph (nx.Graph): A graph whose nodes are the integers ``0..len(entities)-1``
            (as networkx generators produce). Each edge ``(i, j)`` becomes a symmetric
            Application edge between ``entities[i]`` and ``entities[j]``.
        application_id (str): The Application's id. Defaults to ``"fedclypse"``.

    Returns:
        Application: The assembled communication graph, not yet registered with a
        Simulation.
    """
    app = Application(application_id, include_default_assets=False)
    for entity in entities:
        app.add_service(entity)
    for i, j in graph.edges():
        app.add_edge(entities[i].id, entities[j].id, symmetric=True)
    return app


def star(
    server: Entity,
    clients: Sequence[Entity],
    application_id: str = "fedclypse",
) -> Application:
    """Build a client-server star: the server plus a symmetric edge to each client.

    Args:
        server (Entity): The server entity at the star's hub.
        clients (Sequence[Entity]): The client entities, each wired to ``server``.
        application_id (str): The Application's id. Defaults to ``"fedclypse"``.

    Returns:
        Application: The star communication graph.
    """
    app = Application(application_id, include_default_assets=False)
    app.add_service(server)
    for client in clients:
        app.add_service(client)
        app.add_edge(server.id, client.id, symmetric=True)
    return app


def ring(
    peers: Sequence[Entity],
    application_id: str = "fedclypse",
) -> Application:
    """Build a decentralized ring, each peer connected to its two ring neighbours.

    Args:
        peers (Sequence[Entity]): The peer entities, arranged into a cycle in order.
        application_id (str): The Application's id. Defaults to ``"fedclypse"``.

    Returns:
        Application: The ring communication graph.
    """
    return from_graph(peers, nx.cycle_graph(len(peers)), application_id)


def complete(
    peers: Sequence[Entity],
    application_id: str = "fedclypse",
) -> Application:
    """Build a fully-connected mesh: every peer connected to every other peer.

    Args:
        peers (Sequence[Entity]): The peer entities.
        application_id (str): The Application's id. Defaults to ``"fedclypse"``.

    Returns:
        Application: The complete-graph communication topology.
    """
    return from_graph(peers, nx.complete_graph(len(peers)), application_id)
