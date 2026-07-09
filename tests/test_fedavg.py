# -*- coding: utf-8 -*-
import numpy as np

from fedclypse.core import ArrayModel, Contribution, Parameters
from fedclypse.data import ClientData, InMemorySource
from fedclypse.fedavg import FedAvgClient, FedAvgServer
from fedclypse.selection import select_all


def _p(*vals):
    return Parameters([np.array(v, dtype=float) for v in vals])


def _contrib(val, weight):
    return Contribution(payload=_p(val), weight=weight)


def _server(rounds=1):
    return FedAvgServer(
        "server", model_factory=lambda: ArrayModel(_p([0.0])), rounds=rounds
    )


def test_fedavg_server_aggregate_is_uniform_mean_for_equal_weights():
    out = _server().aggregate([_contrib([2.0], 1.0), _contrib([4.0], 1.0)])
    assert np.allclose(out.tensors[0], [3.0])


def test_fedavg_server_aggregate_weights_by_num_examples():
    # weights 1 and 3 -> (1*2 + 3*6) / 4 = 5
    out = _server().aggregate([_contrib([2.0], 1.0), _contrib([6.0], 3.0)])
    assert np.allclose(out.tensors[0], [5.0])


def test_fedavg_server_constructs_with_rounds_and_default_selection():
    server = _server(rounds=5)
    assert server.rounds == 5
    assert server.selection is select_all


def test_fedavg_client_local_update_default_is_identity():
    client = FedAvgClient("client_0")
    p = _p([1.0, 2.0])
    assert client.local_update(p) is p


def test_fedavg_client_num_examples_reflects_dataset_size():
    data = ClientData(InMemorySource([10, 11, 12, 13], labels=[0, 1, 0, 1]), [0, 2, 3])
    client = FedAvgClient("client_0", data=data)
    assert client.num_examples == 1.0  # no dataset materialized before deploy
    client.on_deploy()
    assert client.num_examples == 3.0
