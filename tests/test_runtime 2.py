# -*- coding: utf-8 -*-
import subprocess
import sys

import pytest

from fedclypse.entity import Entity
from fedclypse.runtime import build_simulation, star_application


def _entities(n):
    server = Entity("server")
    clients = [Entity(f"client_{i}") for i in range(n)]
    return server, clients


def test_star_application_has_server_and_clients_as_services():
    server, clients = _entities(3)
    app = star_application(server, clients)
    assert set(app.nodes) == {"server", "client_0", "client_1", "client_2"}


def test_star_application_wires_server_to_every_client():
    server, clients = _entities(3)
    app = star_application(server, clients)
    edge_pairs = {frozenset((u, v)) for u, v in app.edges}
    for i in range(3):
        assert frozenset(("server", f"client_{i}")) in edge_pairs


def test_build_simulation_pure_mode_is_not_remote():
    server, clients = _entities(2)
    app = star_application(server, clients)
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
        "from fedclypse.runtime import build_simulation, star_application\n"
        "server = Entity('server')\n"
        "clients = [Entity('client_0'), Entity('client_1')]\n"
        "app = star_application(server, clients)\n"
        "sim = build_simulation(app, rounds=3, mode='emulation')\n"
        "assert sim.remote\n"
        "print('OK')\n"
    )
    result = subprocess.run(
        [sys.executable, "-c", script], capture_output=True, text=True, timeout=120
    )
    assert result.returncode == 0, result.stdout + result.stderr
    assert "OK" in result.stdout
