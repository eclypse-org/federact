import numpy as np
import pytest

from fedclypse.aggregation import (
    IncompatibleContributionsError,
    fedavg,
    mean,
    weighted_sum,
)
from fedclypse.core import Contribution, Parameters, parameters


def _p(*arrays):
    return Parameters([np.array(a, dtype=float) for a in arrays])


def test_fedavg_weights_by_contribution_weight():
    contribs = [
        parameters(_p([0.0, 0.0]), weight=1.0),
        parameters(_p([3.0, 6.0]), weight=3.0),
    ]
    # (1*0 + 3*3) / 4 = 2.25 ; (1*0 + 3*6) / 4 = 4.5
    result = fedavg(contribs)
    assert np.allclose(result.tensors[0], [2.25, 4.5])


def test_mean_is_uniform_average():
    result = mean([parameters(_p([2.0])), parameters(_p([4.0]))])
    assert np.allclose(result.tensors[0], [3.0])


def test_weighted_sum_scales_and_sums():
    result = weighted_sum(
        [parameters(_p([1.0]), weight=2.0), parameters(_p([1.0]), weight=5.0)]
    )
    assert np.allclose(result.tensors[0], [7.0])


def test_empty_contributions_raise():
    with pytest.raises(ValueError):
        fedavg([])


def test_mismatched_shapes_raise_incompatible():
    with pytest.raises(IncompatibleContributionsError):
        fedavg([parameters(_p([1.0, 2.0])), parameters(_p([1.0]))])


def test_non_parameters_payload_raises_incompatible():
    with pytest.raises(IncompatibleContributionsError):
        fedavg([Contribution(payload={"logits": 1}), parameters(_p([1.0]))])


def test_heterogeneous_model_descriptors_raise_incompatible():
    with pytest.raises(IncompatibleContributionsError):
        fedavg(
            [
                parameters(_p([1.0]), model_descriptor="cnn"),
                parameters(_p([1.0]), model_descriptor="mlp"),
            ]
        )


def test_single_shared_descriptor_is_allowed():
    result = fedavg(
        [
            parameters(_p([2.0]), weight=1.0, model_descriptor="cnn"),
            parameters(_p([4.0]), weight=1.0, model_descriptor="cnn"),
        ]
    )
    assert np.allclose(result.tensors[0], [3.0])


def test_fedavg_zero_total_weight_raises():
    with pytest.raises(ValueError):
        fedavg([parameters(_p([1.0]), weight=0.0), parameters(_p([2.0]), weight=0.0)])
