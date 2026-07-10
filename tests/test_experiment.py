# -*- coding: utf-8 -*-
import networkx as nx
import numpy as np
import pytest

from fedclypse.core import ArrayModel, Parameters
from fedclypse.experiment import (
    Behaviour,
    NodeSpec,
    Situation,
    fedavg_behaviour,
    run,
    star_situation,
)
from fedclypse.schemes import FedAvgClient, FedAvgServer


def _mf():
    return ArrayModel(Parameters([np.zeros(3)]))


# ---- star_situation ----


def test_star_situation_builds_roles_data_and_star_graph():
    s = star_situation(
        "server", ["c0", "c1"], data={"c0": "DATA0", "c1": "DATA1"}, model_factory=_mf
    )
    assert isinstance(s, Situation)
    # ordered nodes: server at index 0, then clients
    assert [n.id for n in s.nodes] == ["server", "c0", "c1"]
    assert [n.role for n in s.nodes] == ["server", "client", "client"]
    assert s.nodes[0].data is None  # server is dataless
    assert s.nodes[1].data == "DATA0" and s.nodes[2].data == "DATA1"
    assert all(n.model_factory is _mf for n in s.nodes)
    # nx.star_graph(2): 3 nodes, center 0 connected to 1 and 2
    assert s.graph.number_of_nodes() == 3
    assert set(s.graph.edges()) == {(0, 1), (0, 2)}
    assert s.infrastructure is None and s.placement is None


def test_star_situation_defaults_missing_client_data_to_none():
    s = star_situation("server", ["c0", "c1"], model_factory=_mf)
    assert s.nodes[1].data is None and s.nodes[2].data is None


# ---- fedavg_behaviour ----


def test_fedavg_behaviour_builds_fedavg_entities_with_rounds_threaded():
    b = fedavg_behaviour()
    assert isinstance(b, Behaviour)
    assert set(b.roles) == {"server", "client"}
    server = b.roles["server"](NodeSpec("server", "server", model_factory=_mf), 7)
    assert isinstance(server, FedAvgServer) and server.rounds == 7
    client = b.roles["client"](NodeSpec("c0", "client", data="D", model_factory=_mf), 7)
    assert isinstance(client, FedAvgClient) and client.id == "c0"


# ---- run: role-seam check ----


def test_run_raises_when_situation_assigns_an_unprovided_role():
    situation = Situation(
        nodes=[NodeSpec("n0", "ghost", model_factory=_mf)], graph=nx.empty_graph(1)
    )
    with pytest.raises(ValueError, match="ghost"):
        run(situation, fedavg_behaviour(), rounds=1)
