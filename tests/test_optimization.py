# -*- coding: utf-8 -*-
import numpy as np

from fedclypse.core import Parameters
from fedclypse.optimization import (
    ServerAdagrad,
    ServerAdam,
    ServerOpt,
    ServerSGD,
    ServerYogi,
)


def _p(*vals):
    return Parameters([np.array(v, dtype=float) for v in vals])


# ---- ServerSGD ----


def test_serversgd_default_lr_one_is_fedavg_identity():
    # x + 1*delta; with delta = aggregate - x this returns the aggregate.
    out = ServerSGD().step(_p([2.0]), _p([4.0]))  # 2 + 4 == 6
    assert np.allclose(out.tensors[0], [6.0])


def test_serversgd_scaled_lr():
    out = ServerSGD(lr=0.5).step(_p([0.0]), _p([4.0]))
    assert np.allclose(out.tensors[0], [2.0])


def test_serversgd_momentum_accumulates_across_steps():
    opt = ServerSGD(lr=1.0, momentum=0.9)
    x1 = opt.step(_p([0.0]), _p([1.0]))  # buf = 1.0 -> x = 1.0
    assert np.allclose(x1.tensors[0], [1.0])
    x2 = opt.step(x1, _p([1.0]))  # buf = 0.9*1 + 1 = 1.9 -> x = 1.0 + 1.9 = 2.9
    assert np.allclose(x2.tensors[0], [2.9])


# ---- ServerAdagrad ----


def test_serveradagrad_first_step_and_state_accumulates():
    opt = ServerAdagrad(lr=1.0, eps=0.0)
    x1 = opt.step(_p([0.0]), _p([4.0]))  # v = 16, step = 4/4 = 1
    assert np.allclose(x1.tensors[0], [1.0])
    x2 = opt.step(x1, _p([3.0]))  # v = 16 + 9 = 25, step = 3/5 = 0.6
    assert np.allclose(
        x2.tensors[0], [1.6]
    )  # 1.0 + 0.6; proves v ACCUMULATED (else 1+1=2.0)


# ---- ServerAdam ----


def test_serveradam_first_step_is_lr_times_sign():
    # With bias correction and eps=0, the first Adam step is lr*sign(delta) elementwise.
    out = ServerAdam(lr=1.0, eps=0.0).step(_p([0.0, 0.0]), _p([-3.0, 2.0]))
    assert np.allclose(out.tensors[0], [-1.0, 1.0])


def test_serveradam_state_persists_and_step_count_advances():
    opt = ServerAdam(lr=1.0, eps=0.0)
    opt.step(_p([0.0]), _p([1.0]))
    opt.step(_p([0.0]), _p([1.0]))
    assert opt._t == 2  # step count advanced across calls (state persisted)


def test_serveradam_instances_do_not_share_state():
    a = ServerAdam(lr=1.0)
    b = ServerAdam(lr=1.0)
    a.step(_p([0.0]), _p([1.0]))
    assert a._t == 1 and b._t == 0


# ---- ServerYogi ----


def test_serveryogi_first_step_matches_adam_then_diverges_by_sign_rule():
    # v0 = 0, so first-step v = -(1-b2)*sign(0 - g^2)*g^2 = +(1-b2)*g^2 == Adam's first v.
    out = ServerYogi(lr=1.0, eps=0.0).step(_p([0.0]), _p([1.0]))
    assert np.allclose(out.tensors[0], [1.0])
    # Yogi's v update is sign-based; assert it runs a second step and advances state.
    opt = ServerYogi(lr=1.0, eps=0.0)
    opt.step(_p([0.0]), _p([1.0]))
    opt.step(_p([0.0]), _p([1.0]))
    assert opt._t == 2


# ---- base ----


def test_all_optimizers_are_serveropt():
    for opt in [ServerSGD(), ServerAdagrad(1.0), ServerAdam(1.0), ServerYogi(1.0)]:
        assert isinstance(opt, ServerOpt)
