import pytest
from eclypse.builders.infrastructure import get_star

from fedclypse.core import Entity
from fedclypse.deployment import collapse, mirror


def _entities(n):
    return [Entity(f"e{i}") for i in range(n)]


def _edge_pairs(app):
    return {frozenset((u, v)) for u, v in app.edges}


def test_mirror_derives_star_comm_graph_and_static_mapping():
    # get_star(3): nodes center + outer_0..2, edges center<->outer_i (symmetric).
    infra = get_star(n_clients=3, include_default_assets=False, symmetric=True, seed=0)
    entities = _entities(4)  # one per infra node
    app, strategy = mirror(infra, entities)
    # sorted node ids: center, outer_0, outer_1, outer_2 -> e0..e3
    assert strategy.mapping == {
        "e0": "center",
        "e1": "outer_0",
        "e2": "outer_1",
        "e3": "outer_2",
    }
    # e0 (center) neighbours all three; each outer entity neighbours only e0.
    assert _edge_pairs(app) == {
        frozenset(("e0", "e1")),
        frozenset(("e0", "e2")),
        frozenset(("e0", "e3")),
    }


def test_mirror_requires_one_entity_per_node():
    infra = get_star(n_clients=3, include_default_assets=False, symmetric=True, seed=0)
    with pytest.raises(ValueError):
        mirror(infra, _entities(2))  # 2 entities vs 4 nodes


def test_collapse_pins_all_entities_to_one_node():
    strategy = collapse(_entities(3), "center")
    assert strategy.mapping == {"e0": "center", "e1": "center", "e2": "center"}
