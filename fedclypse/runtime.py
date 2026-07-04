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
    metrics: Optional[List] = None,
) -> Simulation:
    """Wire an Application into an eclypse Simulation (does NOT run it).

    ``mode="emulation"`` sets ``remote=True`` (Ray-backed, real ``run()`` execution);
    ``mode="simulation"`` sets ``remote=False`` (placement/comm/timing only).
    ``metrics`` is a list of ``@metric.service`` objects collected during the run.
    The default infrastructure is a star sized to the number of client services, built
    with ``include_default_assets=False`` so placement is always feasible (an infra with
    default resource assets is not guaranteed to fit the Application's default service
    requirements); pass an explicit ``infrastructure`` to model resources.
    """
    if infrastructure is None:
        clients = (
            n_clients if n_clients is not None else max(1, len(application.nodes) - 1)
        )
        infrastructure = get_star(
            n_clients=clients,
            include_default_assets=False,
            resource_init="max",
            symmetric=True,
            seed=seed,
        )
    config = SimulationConfig(
        remote=(mode == "emulation"),
        max_steps=rounds,
        seed=seed,
        include_default_metrics=True,
        events=list(metrics) if metrics else None,
    )
    simulation = Simulation(infrastructure, simulation_config=config)
    simulation.register(application, RandomStrategy(seed=seed))
    return simulation


def run_federation(
    simulation: Simulation,
    rounds: int,
    *,
    step_delay: float = 0.5,
    grace: float = 1.0,
) -> "History":
    """Drive an emulation to completion and return its collected History.

    Encapsulates the verified emulation recipe: ``start``, then advance the simulation
    manually ``rounds`` times (each ``step()`` is a fire-and-forget remote call, so a
    short ``step_delay`` is needed for it to take effect), wait ``grace`` seconds for
    the last round's messages to settle, then ``stop()`` (blocking; it cancels the
    service tasks and tears down the Ray actors). The FL rounds progress autonomously
    inside the entities' ``run()`` loops — faster than the driver's steps — so the last
    sampled metric reflects the converged state. No ``ray.shutdown()`` is required.
    """
    import time

    from fedclypse.metrics import History

    simulation.start()
    for _ in range(rounds):
        simulation.step()
        time.sleep(step_delay)
    time.sleep(grace)
    simulation.stop()
    return History.from_report(simulation.report)
