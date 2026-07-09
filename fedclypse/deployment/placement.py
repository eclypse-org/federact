# -*- coding: utf-8 -*-
"""Physical placement: mapping entities onto infrastructure nodes.

The communication topology (``fedclypse.deployment.topology``) and the physical placement are
independent. This module reuses eclypse's ``PlacementStrategy`` suite directly
(``StaticStrategy``, ``RoundRobinStrategy``, ``RandomStrategy``, ``FirstFitStrategy``,
``BestFitStrategy``) and adds only the conveniences the deployment spectrum needs:
``mirror`` for faithful 1-to-1 emulation (derive the comm graph from an infrastructure),
and ``collapse`` for the compute-constrained case (many entities on one node). Value-based
placement is an extension point: subclass eclypse's ``PlacementStrategy`` and override
``place`` to spawn learners on the most valuable nodes (e.g. data locality). NOTE: pinning
many entities to one node (``collapse``) co-locates their training loops in one Ray actor;
capping how many run at once (per the node's GPUs) is a separate concurrency concern, not
handled here.
"""
from __future__ import annotations

from typing import Dict, List, Sequence, Tuple

from eclypse.graph import Application, Infrastructure
from eclypse.placement.strategies import StaticStrategy

from fedclypse.core.entity import Entity

__all__ = ["mirror", "collapse"]


def mirror(
    infrastructure: Infrastructure,
    entities: Sequence[Entity],
    application_id: str = "fedclypse",
) -> Tuple[Application, StaticStrategy]:
    """Pin entities 1-to-1 onto infra nodes and mirror the infra topology into comm.

    Assigns ``entities[k]`` to the ``k``-th infrastructure node (by sorted node id) and
    builds an Application whose edges are the infrastructure's edges translated through
    that mapping — a symmetric Application edge for every infrastructure edge. This is how
    eclypse's infrastructure topology suite (hierarchical, random, realistic fabrics)
    becomes an FL communication topology for faithful emulation.

    Args:
        infrastructure (Infrastructure): The infrastructure whose topology is mirrored.
            Must have exactly ``len(entities)`` nodes.
        entities (Sequence[Entity]): The entities, one per infrastructure node.
        application_id (str): The Application's id. Defaults to ``"fedclypse"``.

    Returns:
        Tuple[Application, StaticStrategy]: The derived communication graph, and the
        placement pinning each entity to its infrastructure node.

    Raises:
        ValueError: If ``len(entities)`` does not equal the number of infrastructure nodes.
    """
    node_ids: List[str] = sorted(infrastructure.nodes)
    if len(entities) != len(node_ids):
        raise ValueError(
            f"mirror() requires exactly one entity per infra node: "
            f"{len(entities)} entities vs {len(node_ids)} infra nodes."
        )

    node_to_entity: Dict[str, Entity] = dict(zip(node_ids, entities))
    mapping: Dict[str, str] = {
        entity.id: node for node, entity in node_to_entity.items()
    }

    app = Application(application_id, include_default_assets=False)
    for entity in entities:
        app.add_service(entity)

    seen = set()
    for a, b in infrastructure.edges():
        pair = frozenset((a, b))
        if pair in seen:
            continue
        seen.add(pair)
        ea, eb = node_to_entity[a], node_to_entity[b]
        if ea.id == eb.id:
            continue
        app.add_edge(ea.id, eb.id, symmetric=True)

    return app, StaticStrategy(mapping)


def collapse(entities: Sequence[Entity], node_id: str) -> StaticStrategy:
    """Pin every entity onto a single infrastructure node (the driver-node case).

    Args:
        entities (Sequence[Entity]): The entities to co-locate.
        node_id (str): The infrastructure node id to place them all on.

    Returns:
        StaticStrategy: A placement mapping every entity to ``node_id``.
    """
    return StaticStrategy({entity.id: node_id for entity in entities})
