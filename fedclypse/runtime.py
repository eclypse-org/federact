# -*- coding: utf-8 -*-
"""Runtime assembly and driving: wire fedclypse entities into an eclypse Simulation.

This module turns an Application (e.g. from ``fedclypse.topology``) into a runnable
eclypse experiment: ``build_simulation`` wires it to an infrastructure and a
``SimulationConfig`` (without running it), and ``run_federation`` drives a built
simulation through an emulation to completion, returning the collected metric
``History``.
"""
from __future__ import annotations

from typing import List, Optional, TYPE_CHECKING

from eclypse.builders.infrastructure import get_star
from eclypse.graph import Application, Infrastructure
from eclypse.placement.strategies import RandomStrategy
from eclypse.simulation import Simulation, SimulationConfig

if TYPE_CHECKING:
    # Only for the ``run_federation`` return-type annotation; the real,
    # non-circular import happens inside that function's body (see below).
    from fedclypse.metrics import History

__all__ = ["build_simulation", "run_federation"]


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

    Args:
        application (Application): The eclypse Application to run (e.g. from
            ``fedclypse.topology.star``).
        infrastructure (Optional[Infrastructure]): The infrastructure graph to
            place ``application`` on. Defaults to ``None``, which builds a
            star sized to ``n_clients`` (or ``len(application.nodes) - 1``
            when ``n_clients`` is not given) via ``get_star``.
        rounds (int): The simulation's step budget, forwarded to
            ``SimulationConfig.max_steps``.
        seed (int): Seed for the default infrastructure, the simulation
            config, and the placement strategy. Defaults to ``0``.
        mode (str): ``"emulation"`` for a Ray-backed remote run, or
            ``"simulation"`` for placement/comm/timing only. Defaults to
            ``"emulation"``.
        n_clients (Optional[int]): The number of clients to size the default
            infrastructure for, used only when ``infrastructure`` is not
            given. Defaults to ``None`` (inferred from ``application``).
        metrics (Optional[List]): ``@metric.service``-decorated callables to
            collect during the run, forwarded to ``SimulationConfig.events``.
            Defaults to ``None`` (only the default metrics are collected).

    Returns:
        Simulation: The assembled Simulation, with ``application`` already
        registered against a placement strategy. Not started.
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

    Args:
        simulation (Simulation): A built, registered Simulation (e.g. from
            ``build_simulation``), not yet started.
        rounds (int): The number of manual ``step()`` calls to issue.
        step_delay (float): Seconds to sleep after each ``step()`` call,
            giving the fire-and-forget remote call time to take effect.
            Defaults to ``0.5``.
        grace (float): Seconds to sleep after the last ``step()`` and before
            ``stop()``, letting the final round's in-flight messages settle.
            Defaults to ``1.0``.

    Returns:
        History: The emulation's collected metric samples, read from
        ``simulation.report`` after ``stop()`` has torn the simulation down.
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
