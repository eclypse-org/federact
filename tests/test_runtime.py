# -*- coding: utf-8 -*-
import subprocess
import sys
import textwrap

import pytest

from fedclypse.entity import Entity
from fedclypse.runtime import build_simulation
from fedclypse.topology import star


def _entities(n):
    server = Entity("server")
    clients = [Entity(f"client_{i}") for i in range(n)]
    return server, clients


def test_build_simulation_pure_mode_is_not_remote():
    server, clients = _entities(2)
    app = star(server, clients)
    sim = build_simulation(app, rounds=5, mode="simulation")
    assert not sim.remote  # remote=False normalizes to None (public attr eclypse sets)
    assert sim._sim_config.max_steps == 5  # rounds -> max_steps


def test_build_simulation_emulation_mode_is_remote():
    """Constructing a remote=True Simulation eagerly builds Ray actors; doing so in
    the SAME process as a prior non-remote Simulation raises 'SimpleQueue objects
    should only be shared between processes through inheritance' (ray pickling). That
    hazard is not cleared by ray.shutdown() (verified) — only process isolation is.
    Run the real build_simulation(mode="emulation") contract in a fresh subprocess.
    """
    pytest.importorskip("ray")
    script = (
        "from fedclypse.entity import Entity\n"
        "from fedclypse.runtime import build_simulation\n"
        "from fedclypse.topology import star\n"
        "server = Entity('server')\n"
        "clients = [Entity('client_0'), Entity('client_1')]\n"
        "app = star(server, clients)\n"
        "sim = build_simulation(app, rounds=3, mode='emulation')\n"
        "assert sim.remote\n"
        "print('OK')\n"
    )
    result = subprocess.run(
        [sys.executable, "-c", script], capture_output=True, text=True, timeout=120
    )
    assert result.returncode == 0, result.stdout + result.stderr
    assert "OK" in result.stdout


def test_run_federation_drives_start_steps_stop(monkeypatch):
    import time

    import pandas as pd

    from fedclypse.runtime import run_federation

    monkeypatch.setattr(time, "sleep", lambda *_a, **_k: None)
    calls = []

    class FakeReport:
        def service(self):
            return pd.DataFrame(
                [["fedclypse_round", "server", 1, 42.0]],
                columns=["callback_id", "service_id", "n_event", "value"],
            )

    class FakeSim:
        report = FakeReport()

        def start(self):
            calls.append("start")

        def step(self):
            calls.append("step")

        def stop(self):
            calls.append("stop")

    history = run_federation(FakeSim(), rounds=3)
    assert calls == ["start", "step", "step", "step", "stop"]
    assert history.final("fedclypse_round", "server") == 42.0


def test_build_simulation_accepts_explicit_placement():
    from eclypse.placement.strategies import RoundRobinStrategy

    from fedclypse.topology import star

    server = Entity("server")
    clients = [Entity("client_0"), Entity("client_1")]
    app = star(server, clients)
    sim = build_simulation(
        app, rounds=2, mode="simulation", placement=RoundRobinStrategy()
    )
    assert sim is not None
    assert not sim.remote


def test_build_simulation_rejects_asymmetric_infrastructure():
    from eclypse.builders.infrastructure import get_star

    from fedclypse.topology import star

    server = Entity("server")
    clients = [Entity("client_0")]
    app = star(server, clients)
    asymmetric = get_star(n_clients=2, include_default_assets=False, symmetric=False)
    with pytest.raises(ValueError, match="symmetric"):
        build_simulation(app, infrastructure=asymmetric, rounds=2, mode="simulation")


def test_fedavg_emulation_smoke():
    """End-to-end FedAvg emulation (1 server + 2 clients, K=2 rounds, torch-free).
    Runs in a fresh subprocess (a remote=True Simulation must not be built in the
    pytest process — see test_build_simulation_emulation_mode_is_remote). Verifies the
    whole stack: behaviours run on workers, messages flow, aggregation is correct, and
    the converged model is read back via History. This test is timing-sensitive (manual
    emulation driving); bump step_delay/grace if it flakes.

    FedAvg math: server starts at [0,0,0]; client_0 adds 1, client_1 adds 3 each round;
    equal weights -> round 1 mean = 2, round 2 mean = 4.
    """
    pytest.importorskip("ray")
    script = textwrap.dedent(
        """
        import numpy as np
        from eclypse.report.metrics import metric

        from fedclypse.fedavg import FedAvgClient, FedAvgServer
        from fedclypse.metrics import round_metric
        from fedclypse.model import ArrayModel
        from fedclypse.parameters import Parameters
        from fedclypse.runtime import build_simulation, run_federation
        from fedclypse.topology import star


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


        server = FedAvgServer(
            "server",
            model_factory=lambda: ArrayModel(Parameters([np.zeros(3)])),
            rounds=2,
        )
        clients = [
            TransformClient(
                cid, model_factory=lambda: ArrayModel(Parameters([np.zeros(3)]))
            )
            for cid in ["client_0", "client_1"]
        ]
        app = star(server, clients)
        sim = build_simulation(
            app, rounds=2, mode="emulation", metrics=[round_metric, model_mean]
        )
        history = run_federation(sim, rounds=2, step_delay=0.5, grace=1.5)

        final = history.final("model_mean", service_id="server")
        assert final is not None, "no model_mean samples collected"
        assert abs(final - 4.0) < 1e-6, f"expected 4.0, got {final}"
        assert sim.status.name == "IDLE"
        print("SMOKE_OK")
        """
    )
    result = subprocess.run(
        [sys.executable, "-c", script], capture_output=True, text=True, timeout=180
    )
    assert result.returncode == 0, result.stdout + result.stderr
    assert "SMOKE_OK" in result.stdout
