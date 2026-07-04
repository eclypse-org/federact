# -*- coding: utf-8 -*-
from __future__ import annotations

from typing import List, Optional

from eclypse.builders.infrastructure import get_star
from eclypse.graph import Application, Infrastructure
from eclypse.placement.strategies import RandomStrategy
from eclypse.simulation import Simulation, SimulationConfig

from fedclypse.entity import Entity


def star_application(
    server: Entity,
    clients: List[Entity],
    application_id: str = "fedclypse",
) -> Application:
    """Build the eclypse Application graph for a client-server federation: the server
    plus every client as services, with a symmetric interaction edge from the server
    to each client (a star)."""
    app = Application(application_id, include_default_assets=True)
    app.add_service(server)
    for client in clients:
        app.add_service(client)
        app.add_edge(server.id, client.id, symmetric=True)
    return app


def build_simulation(
    application: Application,
    infrastructure: Optional[Infrastructure] = None,
    *,
    rounds: int,
    seed: int = 0,
    mode: str = "emulation",
    n_clients: Optional[int] = None,
) -> Simulation:
    """Wire an Application into an eclypse Simulation (does NOT run it).

    ``mode="emulation"`` sets ``remote=True`` (Ray-backed, real ``run()`` execution);
    ``mode="simulation"`` sets ``remote=False`` (placement/comm/timing only).
    Infrastructure defaults to a star sized to the number of client services.
    """
    if infrastructure is None:
        clients = (
            n_clients if n_clients is not None else max(1, len(application.nodes) - 1)
        )
        infrastructure = get_star(
            n_clients=clients,
            include_default_assets=True,
            resource_init="max",
            symmetric=True,
            seed=seed,
        )
    config = SimulationConfig(
        remote=(mode == "emulation"),
        max_steps=rounds,
        seed=seed,
        include_default_metrics=True,
    )
    simulation = Simulation(infrastructure, simulation_config=config)
    simulation.register(application, RandomStrategy(seed=seed))
    return simulation
