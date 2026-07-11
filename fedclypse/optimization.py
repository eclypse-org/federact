"""Server-side optimizers: apply the aggregated update as a pseudo-gradient.

A ``ServerOpt`` treats the aggregated client update as a pseudo-gradient and
steps the global model, generalizing FedAvg (``ServerSGD(1.0)``) to the FedOpt
family (``ServerAdagrad``/``ServerAdam``/``ServerYogi``; Reddi et al., 2021). It
is the server half of the ClientOpt/ServerOpt decomposition, orthogonal to the
aggregation rule and the synchronization policy: the ``Aggregator`` forms the
improvement direction ``delta = aggregate - current`` and hands it to ``step``.
All optimizers are framework-agnostic numpy math over ``Parameters`` and are
stateful -- their moment accumulators persist across rounds -- but the state is
plain numpy, so it pickles cleanly across the worker boundary.
"""

from __future__ import annotations

from abc import (
    ABC,
    abstractmethod,
)

import numpy as np

from fedclypse.core.parameters import Parameters

__all__ = ["ServerAdagrad", "ServerAdam", "ServerOpt", "ServerSGD", "ServerYogi"]


class ServerOpt(ABC):
    """A server-side optimizer: steps the global model along a pseudo-gradient.

    Given the current global ``params`` and the aggregated improvement
    direction ``delta`` (``aggregate - current``, formed by the ``Aggregator``),
    ``step`` returns the new global parameters ``params + (adaptively-scaled)
    delta``. Concrete subclasses realize the FedOpt family. ``ServerOpt``
    assumes the aggregation rule yields a parameter-space (mean-like) target, so
    ``delta`` is a meaningful pseudo-gradient.

    Subclasses are stateful: their accumulators are lazily initialized
    (``zeros_like`` the params) on the first ``step`` and persist across
    subsequent steps. The state is plain numpy, so it pickles cleanly across the
    worker boundary (no custom ``__getstate__`` is required).
    """

    @abstractmethod
    def step(self, params: Parameters, delta: Parameters) -> Parameters:
        """Apply the pseudo-gradient ``delta`` to ``params``.

        Args:
            params (Parameters): The current global parameters.
            delta (Parameters): The aggregated improvement direction
                (``aggregate - current``); shape-compatible with ``params``.

        Returns:
            Parameters: The new global parameters.
        """
        ...


class ServerSGD(ServerOpt):
    """Server SGD: ``params + lr * delta`` with optional heavy-ball momentum.

    ``ServerSGD(1.0)`` is FedAvg: ``params + (aggregate - params) = aggregate``.
    """

    def __init__(self, lr: float = 1.0, momentum: float = 0.0) -> None:
        """Initialize the server learning rate and momentum.

        Args:
            lr (float): The server learning rate. Defaults to ``1.0`` (FedAvg).
            momentum (float): The heavy-ball momentum coefficient. Defaults to
                ``0.0`` (no momentum).
        """
        self.lr = lr
        self.momentum = momentum
        self._buf: list[np.ndarray] | None = None

    def step(self, params: Parameters, delta: Parameters) -> Parameters:
        """Step ``params`` by ``lr`` along the (optionally accumulated) direction.

        Args:
            params (Parameters): The current global parameters.
            delta (Parameters): The aggregated improvement direction.

        Returns:
            Parameters: ``params + lr * b``, where ``b = momentum * b + delta``.
        """
        if self._buf is None:
            self._buf = [np.zeros_like(t) for t in params.tensors]
        out = []
        for i, (p, d) in enumerate(zip(params.tensors, delta.tensors, strict=False)):
            self._buf[i] = self.momentum * self._buf[i] + d
            out.append(p + self.lr * self._buf[i])
        return Parameters(out, params.tensor_type)


class ServerAdagrad(ServerOpt):
    """Server Adagrad: accumulate the squared pseudo-gradient, scale the step."""

    def __init__(self, lr: float, eps: float = 1e-8) -> None:
        """Initialize the learning rate and numerical-stability constant.

        Args:
            lr (float): The server learning rate.
            eps (float): Added to the denominator for numerical stability.
                Defaults to ``1e-8``.
        """
        self.lr = lr
        self.eps = eps
        self._v: list[np.ndarray] | None = None

    def step(self, params: Parameters, delta: Parameters) -> Parameters:
        """Accumulate ``v += delta**2`` and step ``lr * delta / (sqrt(v) + eps)``.

        Args:
            params (Parameters): The current global parameters.
            delta (Parameters): The aggregated improvement direction.

        Returns:
            Parameters: The stepped parameters.
        """
        if self._v is None:
            self._v = [np.zeros_like(t) for t in params.tensors]
        out = []
        for i, (p, d) in enumerate(zip(params.tensors, delta.tensors, strict=False)):
            self._v[i] = self._v[i] + d * d
            out.append(p + self.lr * d / (np.sqrt(self._v[i]) + self.eps))
        return Parameters(out, params.tensor_type)


class _AdamFamily(ServerOpt):
    """Shared base for the Adam-style server optimizers (Adam, Yogi).

    Maintains bias-corrected first (``m``) and second (``v``) moments and the
    common update ``params + lr * m_hat / (sqrt(v_hat) + eps)``; subclasses
    differ only in the second-moment recurrence ``_next_v``.
    """

    def __init__(
        self, lr: float, betas: tuple[float, float] = (0.9, 0.999), eps: float = 1e-8
    ) -> None:
        """Initialize the learning rate, moment decays, and eps.

        Args:
            lr (float): The server learning rate.
            betas (Tuple[float, float]): The ``(beta1, beta2)`` exponential decay
                rates for the first and second moments. Defaults to
                ``(0.9, 0.999)``.
            eps (float): Added to the denominator for numerical stability.
                Defaults to ``1e-8``.
        """
        self.lr = lr
        self.beta1, self.beta2 = betas
        self.eps = eps
        self._t = 0
        self._m: list[np.ndarray] | None = None
        self._v: list[np.ndarray] | None = None

    @abstractmethod
    def _next_v(self, v: np.ndarray, g2: np.ndarray) -> np.ndarray:
        """Return the updated second moment given previous ``v`` and ``delta**2``.

        Args:
            v (np.ndarray): The previous second-moment accumulator.
            g2 (np.ndarray): The element-wise squared pseudo-gradient.

        Returns:
            np.ndarray: The updated second moment.
        """
        ...

    def step(self, params: Parameters, delta: Parameters) -> Parameters:
        """Update the bias-corrected moments and step ``params``.

        Args:
            params (Parameters): The current global parameters.
            delta (Parameters): The aggregated improvement direction.

        Returns:
            Parameters: The stepped parameters.
        """
        if self._m is None:
            self._m = [np.zeros_like(t) for t in params.tensors]
            self._v = [np.zeros_like(t) for t in params.tensors]
        assert self._v is not None  # always initialized together with self._m
        self._t += 1
        out = []
        for i, (p, d) in enumerate(zip(params.tensors, delta.tensors, strict=False)):
            self._m[i] = self.beta1 * self._m[i] + (1.0 - self.beta1) * d
            self._v[i] = self._next_v(self._v[i], d * d)
            m_hat = self._m[i] / (1.0 - self.beta1**self._t)
            v_hat = self._v[i] / (1.0 - self.beta2**self._t)
            out.append(p + self.lr * m_hat / (np.sqrt(v_hat) + self.eps))
        return Parameters(out, params.tensor_type)


class ServerAdam(_AdamFamily):
    """Server Adam: EMA second moment ``v = beta2*v + (1-beta2)*delta**2``."""

    def _next_v(self, v: np.ndarray, g2: np.ndarray) -> np.ndarray:
        """Return the EMA-updated second moment.

        Args:
            v (np.ndarray): The previous second moment.
            g2 (np.ndarray): The squared pseudo-gradient.

        Returns:
            np.ndarray: ``beta2 * v + (1 - beta2) * g2``.
        """
        return self.beta2 * v + (1.0 - self.beta2) * g2


class ServerYogi(_AdamFamily):
    """Server Yogi: sign-based second moment (Zaheer et al., 2018)."""

    def _next_v(self, v: np.ndarray, g2: np.ndarray) -> np.ndarray:
        """Return the Yogi-updated second moment.

        Args:
            v (np.ndarray): The previous second moment.
            g2 (np.ndarray): The squared pseudo-gradient.

        Returns:
            np.ndarray: ``v - (1 - beta2) * sign(v - g2) * g2``.
        """
        return v - (1.0 - self.beta2) * np.sign(v - g2) * g2
