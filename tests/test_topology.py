# -*- coding: utf-8 -*-
import networkx as nx

from fedclypse.core import Entity
from fedclypse.deployment import complete, from_graph, ring, star


def _entities(n):
    return [Entity(f"e{i}") for i in range(n)]


def _edge_pairs(app):
    return {frozenset((u, v)) for u, v in app.edges}


def test_from_graph_mirrors_networkx_edges():
    peers = _entities(4)
    app = from_graph(peers, nx.cycle_graph(4))
    assert set(app.nodes) == {"e0", "e1", "e2", "e3"}
    assert _edge_pairs(app) == {
        frozenset(("e0", "e1")),
        frozenset(("e1", "e2")),
        frozenset(("e2", "e3")),
        frozenset(("e3", "e0")),
    }


def test_ring_is_a_cycle():
    app = ring(_entities(4))
    for u, v in [("e0", "e1"), ("e1", "e2"), ("e2", "e3"), ("e3", "e0")]:
        assert frozenset((u, v)) in _edge_pairs(app)
    assert len(_edge_pairs(app)) == 4


def test_complete_connects_every_pair():
    app = complete(_entities(4))
    assert len(_edge_pairs(app)) == 6  # C(4,2)


def test_star_wires_server_to_every_client():
    server = Entity("server")
    clients = [Entity(f"c{i}") for i in range(3)]
    app = star(server, clients)
    assert set(app.nodes) == {"server", "c0", "c1", "c2"}
    assert _edge_pairs(app) == {
        frozenset(("server", "c0")),
        frozenset(("server", "c1")),
        frozenset(("server", "c2")),
    }
