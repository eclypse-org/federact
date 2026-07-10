# -*- coding: utf-8 -*-
"""A fully framework-agnostic FedAvg demo: numpy logistic regression, no DL framework.

This script proves that fedclypse can run genuine federated learning with nothing
but numpy: no torch, no downloaded dataset. It generates two linearly-separable
Gaussian blobs, splits the training half IID across four clients, and federates a
logistic-regression classifier between them with FedAvg. Each client's
``local_update`` runs a few steps of plain numpy gradient descent; the server holds
a held-out test set and reports its accuracy each round via a ``@metric.service``
observable. The federation converges quickly on this well-separated task, so the
printed accuracy climbs from the ~0.5 untrained baseline to ~0.99 within the first
round or two (the emulation drives the aggregation rounds autonomously, so the
per-round samples reflect that fast convergence rather than a slow ramp).

Run it with:

    python examples/fedavg_numpy.py

It builds a single Ray-backed (``mode="emulation"``) ``Simulation`` in this
process, so no subprocess isolation is needed (unlike the pytest smoke tests,
which must isolate a ``remote=True`` Simulation in a fresh process).
"""
from __future__ import annotations

from typing import Any, Optional

import numpy as np
from eclypse.report.metrics import metric

from fedclypse.core import ArrayModel, Parameters
from fedclypse.data import ClientData, IID, InMemorySource, split
from fedclypse.experiment import Behaviour, run, star_situation
from fedclypse.schemes import FedAvgClient, FedAvgServer

# ---- hyperparameters ----
DIM = 2  # feature dimensionality
N_TRAIN = 400  # training samples (split across clients)
N_TEST = 100  # held-out test samples (kept by the server)
NUM_CLIENTS = 4
LOCAL_EPOCHS = 30  # numpy GD steps per client per round
LR = 0.5  # local learning rate
ROUNDS = 8


def make_blobs(n: int, rng: np.random.Generator) -> tuple:
    """Draw ``n`` samples from two separable Gaussian blobs (labels 0 and 1).

    Args:
        n (int): Total number of samples to draw (split evenly across the two
            classes).
        rng (np.random.Generator): The random generator driving the draw.

    Returns:
        tuple: ``(X, y)`` where ``X`` is ``(n, DIM)`` float64 and ``y`` is
        ``(n,)`` float64 in ``{0.0, 1.0}``, both shuffled together.
    """
    n0 = n // 2
    n1 = n - n0
    x0 = rng.normal(loc=-1.5, scale=1.0, size=(n0, DIM))
    x1 = rng.normal(loc=1.5, scale=1.0, size=(n1, DIM))
    x = np.concatenate([x0, x1], axis=0)
    y = np.concatenate([np.zeros(n0), np.ones(n1)])
    order = rng.permutation(n)
    return x[order], y[order]


class LogRegClient(FedAvgClient):
    """A FedAvg learner that trains a numpy logistic-regression model locally."""

    def local_update(self, params: Parameters) -> Parameters:
        """Run a few steps of plain numpy gradient descent on the local shard.

        Args:
            params (Parameters): The current global ``[w, b]`` received from
                the server.

        Returns:
            Parameters: The locally-updated ``[w, b]``.
        """
        w, b = params.tensors[0].copy(), params.tensors[1].copy()
        pairs = [self.dataset[i] for i in range(len(self.dataset))]
        x = np.array([p[0] for p in pairs])
        yv = np.array([p[1] for p in pairs], dtype=float)
        for _ in range(LOCAL_EPOCHS):
            p = 1.0 / (1.0 + np.exp(-(x @ w + b[0])))
            w -= LR * (x.T @ (p - yv) / len(yv))
            b -= LR * np.array([np.mean(p - yv)])
        return Parameters([w, b])


@metric.service(remote=True, name="accuracy")
def accuracy(service: Any) -> Optional[float]:
    """Report the global model's accuracy on the server's held-out test set.

    Args:
        service (Any): The live server entity, injected by eclypse on the
            worker.

    Returns:
        Optional[float]: The fraction of the server's test set classified
        correctly by the current global ``[w, b]``, or ``None`` before the
        model/dataset are deployed.
    """
    model = getattr(service, "model", None)
    dataset = getattr(service, "dataset", None)
    if model is None or dataset is None:
        return None
    w, b = model.get_parameters().tensors
    pairs = [dataset[i] for i in range(len(dataset))]
    x = np.array([p[0] for p in pairs])
    y = np.array([p[1] for p in pairs], dtype=float)
    predictions = (x @ w + b[0]) > 0
    return float(np.mean(predictions == y))


def main() -> None:
    """Build the federation, run FedAvg, and print the per-round test accuracy."""
    rng = np.random.default_rng(0)
    x_train, y_train = make_blobs(N_TRAIN, rng)
    x_test, y_test = make_blobs(N_TEST, rng)

    train_source = InMemorySource(data=list(zip(x_train, y_train)), labels=y_train)
    test_source = InMemorySource(data=list(zip(x_test, y_test)), labels=y_test)

    client_ids = [f"client_{i}" for i in range(NUM_CLIENTS)]
    shards = split(train_source, IID(), num_clients=NUM_CLIENTS, seed=0)
    model_factory = lambda: ArrayModel(Parameters([np.zeros(DIM), np.zeros(1)]))

    situation = star_situation(
        "server", client_ids, data=shards, model_factory=model_factory
    )
    # star_situation makes the server dataless; hand it the test set directly
    # so the accuracy metric has something to evaluate against.
    situation.nodes[0].data = ClientData(test_source, list(range(len(x_test))))

    behaviour = Behaviour(
        roles={
            "server": lambda n, r: FedAvgServer(
                n.id, rounds=r, model_factory=n.model_factory, data=n.data
            ),
            "client": lambda n, r: LogRegClient(
                n.id, model_factory=n.model_factory, data=n.data
            ),
        }
    )

    history = run(
        situation,
        behaviour,
        rounds=ROUNDS,
        metrics=[accuracy],
        step_delay=0.5,
        grace=1.5,
    )

    print("Federated logistic regression over Gaussian blobs (numpy-only FedAvg)")
    print(
        f"{NUM_CLIENTS} clients, {N_TRAIN} train / {N_TEST} test samples, "
        f"{ROUNDS} rounds"
    )
    print("-" * 60)
    series = history.series("accuracy", service_id="server")
    for round_idx, value in enumerate(series, start=1):
        _, acc = value
        print(f"round {round_idx}: test accuracy = {acc:.3f}")
    final_accuracy = series[-1][1] if series else None
    print("-" * 60)
    print(f"Final test accuracy: {final_accuracy:.3f}")


if __name__ == "__main__":
    main()
