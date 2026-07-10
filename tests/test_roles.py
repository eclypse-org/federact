# -*- coding: utf-8 -*-
import numpy as np

from fedclypse.aggregation import fedavg
from fedclypse.core import ArrayModel, Contribution, Parameters
from fedclypse.optimization import ServerSGD
from fedclypse.schemes import Aggregator, Learner, Roles
from fedclypse.selection import select_all
from fedclypse.synchronization import Asynchronous, BufferedAsync, Synchronous

COHORT = ["a", "b", "c"]


def _p(*vals):
    return Parameters([np.array(v, dtype=float) for v in vals])


def _dummy_contribs(n):
    return [Contribution(payload=None, version=0) for _ in range(n)]


# ---- (a) staleness composition: effective = weight x staleness_weight ----


def test_aggregator_aggregate_folds_staleness_into_weight():
    def staleness_fn(staleness):
        return 1.0 / (1.0 + staleness)

    synchronizer = Asynchronous(staleness_fn=staleness_fn)
    agg = Aggregator("agg", rounds=1, synchronizer=synchronizer)
    agg.round = 5  # the round the aggregation is computed for

    contribs = [
        Contribution(payload=_p([10.0, 0.0]), source="a", version=3, weight=2.0),
        Contribution(payload=_p([0.0, 10.0]), source="b", version=1, weight=6.0),
    ]

    result = agg.aggregate(contribs)

    # Independently recompute the expectation: reweight each contribution by
    # weight x synchronizer.staleness_weight(c, agg.round), exactly the
    # composition under test, then feed the SAME fedavg used internally.
    reweighted = [
        Contribution(
            payload=c.payload,
            source=c.source,
            version=c.version,
            weight=c.weight * synchronizer.staleness_weight(c, agg.round),
        )
        for c in contribs
    ]
    # a: staleness 5-3=2 -> 1/3 -> effective 2.0 * 1/3 = 2/3
    # b: staleness 5-1=4 -> 1/5 -> effective 6.0 * 1/5 = 1.2
    assert np.isclose(reweighted[0].weight, 2.0 / 3.0)
    assert np.isclose(reweighted[1].weight, 1.2)
    expected = fedavg(reweighted)

    assert np.allclose(result.tensors[0], expected.tensors[0])
    # Sanity: staleness weighting actually moved the result away from an
    # unweighted fedavg over the original (weight-only) contributions.
    unweighted = fedavg(contribs)
    assert not np.allclose(result.tensors[0], unweighted.tensors[0])


# ---- (b) ready-gating counts ----


def test_synchronous_ready_fires_at_full_cohort():
    s = Synchronous()
    assert s.ready(_dummy_contribs(2), COHORT) is False
    assert s.ready(_dummy_contribs(3), COHORT) is True
    assert s.ready(_dummy_contribs(4), COHORT) is True


def test_buffered_async_ready_fires_at_k():
    b = BufferedAsync(2)
    assert b.ready(_dummy_contribs(1), COHORT) is False
    assert b.ready(_dummy_contribs(2), COHORT) is True


def test_asynchronous_ready_fires_at_one():
    a = Asynchronous()
    assert a.ready(_dummy_contribs(0), COHORT) is False
    assert a.ready(_dummy_contribs(1), COHORT) is True


# ---- (c) Learner ----


def test_learner_local_update_is_identity():
    lrn = Learner("lrn")
    p = _p([1.0, 2.0, 3.0])
    assert lrn.local_update(p) is p


def test_learner_num_examples_reflects_dataset_size():
    lrn = Learner("lrn")
    assert lrn.dataset is None
    assert lrn.num_examples == 1.0
    lrn.dataset = [0, 1, 2, 3, 4]
    assert lrn.num_examples == 5.0


# ---- (d) Aggregator defaults ----


def test_aggregator_constructs_with_documented_defaults():
    agg = Aggregator("agg", rounds=3)
    assert agg.rounds == 3
    assert isinstance(agg.synchronizer, Synchronous)
    assert agg.rule is fedavg
    assert agg.selection is select_all


def test_roles_is_the_shared_base_of_both_mixins():
    assert issubclass(Aggregator, Roles)
    assert issubclass(Learner, Roles)


def test_aggregator_accepts_custom_mechanics():
    custom_sync = BufferedAsync(2)
    agg = Aggregator(
        "agg",
        rounds=2,
        selection=lambda neighbours: neighbours[:1],
        synchronizer=custom_sync,
        rule=fedavg,
        model_factory=lambda: ArrayModel(_p([0.0])),
    )
    assert agg.synchronizer is custom_sync
    assert agg.selection(["x", "y"]) == ["x"]


# ---- (e) server_opt seam ----


def test_aggregator_default_server_opt_is_serversgd_one():
    agg = Aggregator("agg", rounds=1)
    assert isinstance(agg.server_opt, ServerSGD)
    assert agg.server_opt.lr == 1.0


def test_aggregator_instances_get_distinct_server_opt():
    a = Aggregator("a", rounds=1)
    b = Aggregator("b", rounds=1)
    assert a.server_opt is not b.server_opt  # sentinel default, not a shared instance


def test_aggregator_default_server_opt_reproduces_the_aggregate():
    # The fire-path math: server_opt.step(current, aggregate - current) == aggregate
    # under the default ServerSGD(1.0). (Exercised via public pieces; the wired
    # async fire path is proven by the emulation smoke.)
    agg = Aggregator("agg", rounds=1)
    agg.model = ArrayModel(_p([0.0]))
    current = agg.model.get_parameters()
    aggregate = _p([4.0])
    delta = aggregate.add(current.scale(-1.0))
    assert np.allclose(agg.server_opt.step(current, delta).tensors[0], [4.0])


def test_aggregator_custom_server_opt_changes_the_step():
    agg = Aggregator("agg", rounds=1, server_opt=ServerSGD(lr=0.5))
    current = _p([0.0])
    delta = _p([4.0]).add(current.scale(-1.0))  # aggregate 4 - current 0 = 4
    assert np.allclose(agg.server_opt.step(current, delta).tensors[0], [2.0])
