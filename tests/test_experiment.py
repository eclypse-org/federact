# -*- coding: utf-8 -*-
import subprocess
import sys
import textwrap

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


# ---- run: end-to-end emulation smoke ----


def test_experiment_run_fedavg_emulation_smoke():
    """End-to-end: the experiment layer's `run(star_situation, behaviour, rounds=2)`
    reproduces the FedAvg convergence to 4.0, proving Situation x Behaviour composes
    over the whole stack and subsumes the hand-assembled path. Runs in a fresh
    subprocess (a remote=True Simulation must not be built in the pytest process).
    Timing-sensitive; bump step_delay/grace if it flakes.

    Same math as tests/test_runtime.py::test_fedavg_emulation_smoke: server starts at
    [0,0,0]; client_0 adds 1, client_1 adds 3 each round; equal weights -> round 1
    mean 2, round 2 mean 4.
    """
    pytest.importorskip("ray")
    script = textwrap.dedent(
        """
        import numpy as np
        from eclypse.report.metrics import metric

        from fedclypse.core import ArrayModel, Parameters
        from fedclypse.schemes import FedAvgClient, FedAvgServer
        from fedclypse.experiment import Behaviour, run, star_situation
        from fedclypse.runtime import round_metric


        @metric.service(remote=True, name="model_mean")
        def model_mean(service):
            model = getattr(service, "model", None)
            if model is None:
                return None
            tensors = model.get_parameters().tensors
            return float(np.mean(np.concatenate([t.ravel() for t in tensors])))


        class TransformClient(FedAvgClient):
            DELTAS = {"client_0": 1.0, "client_1": 3.0}

            def local_update(self, params):
                d = self.DELTAS[self.id]
                return Parameters([t + d for t in params.tensors])


        behaviour = Behaviour(
            roles={
                "server": lambda node, rounds: FedAvgServer(
                    node.id, rounds=rounds, model_factory=node.model_factory
                ),
                "client": lambda node, rounds: TransformClient(
                    node.id, model_factory=node.model_factory
                ),
            }
        )
        situation = star_situation(
            "server",
            ["client_0", "client_1"],
            model_factory=lambda: ArrayModel(Parameters([np.zeros(3)])),
        )
        history = run(
            situation,
            behaviour,
            rounds=2,
            metrics=[round_metric, model_mean],
            step_delay=0.5,
            grace=1.5,
        )

        final = history.final("model_mean", service_id="server")
        assert final is not None, "no model_mean samples collected"
        assert abs(final - 4.0) < 1e-6, f"expected 4.0, got {final}"
        print("EXPERIMENT_SMOKE_OK")
        """
    )
    result = subprocess.run(
        [sys.executable, "-c", script], capture_output=True, text=True, timeout=180
    )
    assert result.returncode == 0, result.stdout + result.stderr
    assert "EXPERIMENT_SMOKE_OK" in result.stdout
