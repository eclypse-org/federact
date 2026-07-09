# -*- coding: utf-8 -*-
import subprocess
import sys
import textwrap

import pytest

from fedclypse.core import Entity
from fedclypse.runtime import build_simulation
from fedclypse.deployment import star


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
        "from fedclypse.core import Entity\n"
        "from fedclypse.runtime import build_simulation\n"
        "from fedclypse.deployment import star\n"
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

    from fedclypse.deployment import star

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

    from fedclypse.deployment import star

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

        from fedclypse.core import ArrayModel, Parameters
        from fedclypse.schemes import FedAvgClient, FedAvgServer
        from fedclypse.runtime import build_simulation, round_metric, run_federation
        from fedclypse.deployment import star


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


def test_buffered_async_emulation_smoke():
    """End-to-end ASYNC emulation: a general `Aggregator` with `BufferedAsync(2)`
    fires its reply handler every 2 arrivals (not on a full round barrier).
    Runs in a fresh subprocess (a remote=True Simulation must not be built in the
    pytest process — see test_build_simulation_emulation_mode_is_remote). This is
    NOT FedAvgServer (which fixes `Synchronous`); it exercises the general
    `Aggregator`/`Learner` roles directly. Timing-sensitive (manual emulation
    driving); bump step_delay/grace if it flakes.

    BufferedAsync(2) math: 4 clients each report a FIXED constant [2,2,2],
    regardless of the model they receive (a client already at its local
    optimum). fedavg of any 2 identical [2,2,2] vectors is [2,2,2],
    order-independent, so the server converges to model_mean == 2.0 after its
    first fire and stays there no matter how arrivals interleave. Because k=2
    < N=4, the reply handler can fire more than once per driver "round" (it
    fires on every 2 arrivals, not on a synchronized barrier), so we assert
    fedclypse_round >= 2 (proves multiple async fires happened) rather than an
    exact count, which would be brittle to straggler overshoot.
    """
    pytest.importorskip("ray")
    script = textwrap.dedent(
        """
        import numpy as np
        from eclypse.report.metrics import metric

        from fedclypse.core import ArrayModel, Parameters
        from fedclypse.schemes import Aggregator, Learner
        from fedclypse.synchronization import BufferedAsync
        from fedclypse.runtime import build_simulation, round_metric, run_federation
        from fedclypse.deployment import star


        @metric.service(remote=True, name="model_mean")
        def model_mean(service):
            model = getattr(service, "model", None)
            if model is None:
                return None
            tensors = model.get_parameters().tensors
            return float(np.mean(np.concatenate([t.ravel() for t in tensors])))


        class ConstantClient(Learner):
            CONST = 2.0

            def local_update(self, params):
                return Parameters([np.full_like(t, self.CONST) for t in params.tensors])


        server = Aggregator(
            "server",
            model_factory=lambda: ArrayModel(Parameters([np.zeros(3)])),
            rounds=3,
            synchronizer=BufferedAsync(2),
        )
        clients = [
            ConstantClient(
                f"client_{i}", model_factory=lambda: ArrayModel(Parameters([np.zeros(3)]))
            )
            for i in range(4)
        ]
        app = star(server, clients)
        sim = build_simulation(
            app, rounds=6, mode="emulation", metrics=[round_metric, model_mean]
        )
        history = run_federation(sim, rounds=6, step_delay=0.5, grace=1.5)

        final = history.final("model_mean", service_id="server")
        assert final is not None, "no model_mean samples collected"
        assert abs(final - 2.0) < 1e-6, f"expected 2.0, got {final}"

        rounds_seen = history.final("fedclypse_round", service_id="server")
        assert rounds_seen is not None and rounds_seen >= 2, (
            f"expected >=2 fires, got {rounds_seen}"
        )
        assert sim.status.name == "IDLE"
        print("ASYNC_SMOKE_OK")
        """
    )
    result = subprocess.run(
        [sys.executable, "-c", script], capture_output=True, text=True, timeout=180
    )
    assert result.returncode == 0, result.stdout + result.stderr
    assert "ASYNC_SMOKE_OK" in result.stdout


def test_ring_emulation_smoke():
    """End-to-end: a non-star (ring) Application graph routes on the default infra.

    A 4-peer ring, each peer broadcasting to its two ring neighbours and receiving from
    exactly them for 2 rounds, run in a fresh subprocess (a remote=True Simulation must
    not be built in the pytest process). Verified working in the topology spike. Timing-
    sensitive; bump step_delay/grace if it flakes.
    """
    pytest.importorskip("ray")
    script = textwrap.dedent(
        """
        import asyncio
        import networkx as nx
        from eclypse.report.metrics import metric

        from fedclypse.core import Entity
        from fedclypse.runtime import build_simulation, run_federation
        from fedclypse.deployment import ring

        ROUNDS = 2


        class RingPeer(Entity):
            async def run(self):
                neighbours = sorted(await self.neighbours())
                self.got = 0
                for r in range(ROUNDS):
                    await self.broadcast(round=r, ping=self.id)
                    for _ in range(len(neighbours)):
                        await self.receive()
                        self.got += 1
                self.n_neighbours = len(neighbours)
                while self.running:
                    await asyncio.sleep(0.05)


        @metric.service(remote=True, name="got")
        def got(service):
            return getattr(service, "got", None)


        peers = [RingPeer(f"peer_{i}") for i in range(4)]
        app = ring(peers)
        sim = build_simulation(app, rounds=ROUNDS, mode="emulation", metrics=[got])
        history = run_federation(sim, rounds=ROUNDS, step_delay=0.5, grace=1.5)

        # every peer received 2 neighbours x 2 rounds = 4 messages
        finals = [history.final("got", service_id=f"peer_{i}") for i in range(4)]
        assert all(f == 4 for f in finals), finals
        assert sim.status.name == "IDLE"
        print("RING_SMOKE_OK")
        """
    )
    result = subprocess.run(
        [sys.executable, "-c", script], capture_output=True, text=True, timeout=180
    )
    assert result.returncode == 0, result.stdout + result.stderr
    assert "RING_SMOKE_OK" in result.stdout


def test_hierarchical_emulation_smoke():
    """End-to-end HIERARCHICAL emulation: a both-roles mid-node cleanly separates
    a parent's `request` from a child's `reply` on ONE shared receive queue, and
    a leaf value propagates two tiers up to the root. Runs in a fresh subprocess
    (a remote=True Simulation must not be built in the pytest process — see
    test_build_simulation_emulation_mode_is_remote). This is the live proof of
    the both-roles dispatch — the riskiest mechanic in the role architecture,
    validated by the Task-1 spike. Timing-sensitive (manual emulation driving);
    bump step_delay/grace if it flakes.

    Topology (`networkx.balanced_tree(2, 2)`, wired via `from_graph`, which
    places `entities[i]` at graph node `i`): root(0) -> mid_0(1), mid_1(2);
    mid_0 -> leaf_0(3), leaf_1(4); mid_1 -> leaf_2(5), leaf_3(6). Leaves are
    plain `Learner`s that report a FIXED constant [5,5,5], ignoring the model
    they receive. Each `mid_*` is a `HierMid(Aggregator, Learner)` — both roles
    at once, on ONE shared receive queue: an inbound `request` (from its
    parent) is routed by `kind` to its Learner half, an inbound `reply` (from
    one of its two children) to its Aggregator half — no message stealing
    (verified by the spike's Q3). The root is a plain `Aggregator` over its two
    mids.

    Math: each mid aggregates its 2 leaves -> mean(5,5) = 5; the root
    aggregates its 2 mids -> mean(5,5) = 5. The root starts at [0,0,0], so
    reaching 5.0 proves the leaf constant propagated UP through the both-roles
    mid tier, not merely that the root's own machinery ran.

    Two constraints are load-bearing (both required by the spike, both
    exercised here):
    - `SelectChildren` is a picklable, class-based (NOT lambda/closure) cohort
      selection that restricts a mid's Aggregator cohort to its known
      children. `get_neighbors` returns the UNDIRECTED neighbourhood, which
      for a mid includes its PARENT; the default `select_all` would put the
      parent in a `Synchronous` mid's cohort, and the parent (an Aggregator
      that never replies to a `request`) would never contribute a reply ->
      permanent deadlock.
    - `HierMid.local_update` returns `self.model.get_parameters()` — on a
      parent request it reports the mid's current AGGREGATED state upward,
      rather than adopting the parent's downward model (the `Learner`
      default's identity `local_update`). The two role loops share one
      `self.model`; adopting the root's [0,0,0] would clobber the
      leaf-aggregate before it could propagate, and the root would never see
      anything but zeros.
    """
    pytest.importorskip("ray")
    script = textwrap.dedent(
        """
        import numpy as np
        import networkx as nx
        from eclypse.report.metrics import metric

        from fedclypse.core import ArrayModel, Parameters
        from fedclypse.schemes import Aggregator, Learner
        from fedclypse.synchronization import Synchronous
        from fedclypse.runtime import build_simulation, round_metric, run_federation
        from fedclypse.deployment import from_graph


        @metric.service(remote=True, name="model_mean")
        def model_mean(service):
            model = getattr(service, "model", None)
            if model is None:
                return None
            tensors = model.get_parameters().tensors
            return float(np.mean(np.concatenate([t.ravel() for t in tensors])))


        LEAF_CONST = 5.0


        class ConstantLeaf(Learner):
            def local_update(self, params):
                return Parameters([np.full_like(t, LEAF_CONST) for t in params.tensors])


        class SelectChildren:
            # picklable child-only cohort selection (excludes the parent neighbour)
            def __init__(self, children):
                self.children = set(children)

            def __call__(self, neighbours):
                return [n for n in neighbours if n in self.children]


        class HierMid(Aggregator, Learner):
            # both-roles mid: aggregates its leaves AND reports upward to its parent
            def __init__(self, entity_id, *, children, rounds, **kwargs):
                super().__init__(
                    entity_id, rounds=rounds, selection=SelectChildren(children), **kwargs
                )

            def local_update(self, params):
                return self.model.get_parameters()


        factory = lambda: ArrayModel(Parameters([np.zeros(3)]))
        root = Aggregator(
            "root", model_factory=factory, rounds=4, synchronizer=Synchronous()
        )
        mid_0 = HierMid(
            "mid_0",
            children=["leaf_0", "leaf_1"],
            rounds=2,
            model_factory=factory,
            synchronizer=Synchronous(),
        )
        mid_1 = HierMid(
            "mid_1",
            children=["leaf_2", "leaf_3"],
            rounds=2,
            model_factory=factory,
            synchronizer=Synchronous(),
        )
        leaves = [ConstantLeaf(f"leaf_{i}", model_factory=factory) for i in range(4)]

        # balanced_tree(2,2): 0=root,1=mid_0,2=mid_1,3=leaf_0,4=leaf_1,5=leaf_2,6=leaf_3
        entities = [root, mid_0, mid_1, leaves[0], leaves[1], leaves[2], leaves[3]]
        app = from_graph(entities, nx.balanced_tree(2, 2))
        sim = build_simulation(
            app, rounds=8, mode="emulation", metrics=[round_metric, model_mean]
        )
        history = run_federation(sim, rounds=8, step_delay=0.5, grace=2.5)

        final = history.final("model_mean", service_id="root")
        assert final is not None, "no root model_mean samples"
        assert abs(final - 5.0) < 1e-6, f"expected 5.0, got {final}"
        assert sim.status.name == "IDLE"
        print("HIER_SMOKE_OK")
        """
    )
    result = subprocess.run(
        [sys.executable, "-c", script], capture_output=True, text=True, timeout=180
    )
    assert result.returncode == 0, result.stdout + result.stderr
    assert "HIER_SMOKE_OK" in result.stdout
