# -*- coding: utf-8 -*-
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
    assert not sim.simulation_config.remote  # remote=False normalizes to None
    assert sim.simulation_config.max_steps == 5


def test_build_simulation_emulation_mode_is_remote():
    pytest.importorskip(
        "ray"
    )  # SimulationConfig(remote=True) requires ray at construction
    server, clients = _entities(2)
    app = star_application(server, clients)
    sim = build_simulation(app, rounds=3, mode="emulation")
    assert (
        sim.simulation_config.remote
    )  # remote=True normalizes to a RemoteBootstrap (truthy)
