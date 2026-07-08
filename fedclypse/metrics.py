# -*- coding: utf-8 -*-
"""Metrics: a ready-made round sampler and a read-only view over collected samples.

``round_metric`` is a ready-made ``@metric.service(remote=True)`` observable
that samples an entity's ``round`` attribute on the worker; ``History`` wraps
the dataframe returned by ``sim.report.service()`` to read the samples back
after an emulation completes.
"""
from __future__ import annotations

from typing import Any, List, Optional, Tuple

from eclypse.report.metrics import metric

__all__ = ["round_metric", "History"]


@metric.service(remote=True, name="fedclypse_round")
def round_metric(service: Any) -> Optional[int]:
    """A ready-made remote metric: samples an entity's ``round`` on the worker.

    Write your own observables the same way — ``@metric.service(remote=True)`` gives
    the callback the live entity, so it can read ``self.model``, ``self.last_loss``,
    etc. Samples are read back with ``History``.

    Args:
        service (Any): The live entity instance the metric is sampled from,
            injected by eclypse on the worker.

    Returns:
        Optional[int]: The entity's current ``self.round``, or ``None`` if
        the entity has no ``round`` attribute.
    """
    return getattr(service, "round", None)


class History:
    """A read-only view over an emulation's collected metric samples.

    Wraps the ``sim.report.service()`` dataframe. Metric samples are keyed by
    ``callback_id`` (the metric's registered name) — NOT ``event_id`` (the triggering
    event, e.g. ``"enact"``). ``series`` returns ``(n_event, value)`` pairs ordered by
    ``n_event``; ``final`` returns the last sampled value (the converged one, read
    after ``sim.stop()``).
    """

    def __init__(self, frame: Any) -> None:
        """Wrap an already-materialized report dataframe.

        Args:
            frame (Any): A pandas-like dataframe with (at least)
                ``callback_id``, ``service_id``, ``n_event``, and ``value``
                columns, as returned by ``sim.report.service()``.
        """
        self._frame = frame

    @classmethod
    def from_report(cls, report: Any) -> "History":
        """Build a History from an eclypse report object.

        Args:
            report (Any): An eclypse ``Report`` (e.g. ``simulation.report``),
                whose ``.service()`` dataframe is materialized and wrapped.

        Returns:
            History: A history view over ``report.service()``.
        """
        return cls(report.service())

    def series(
        self, metric_name: str, service_id: Optional[str] = None
    ) -> List[Tuple[int, Any]]:
        """Read back a metric's samples, ordered by event.

        Args:
            metric_name (str): The metric's registered ``callback_id`` (e.g.
                ``"fedclypse_round"``), matched against the frame's
                ``callback_id`` column.
            service_id (Optional[str]): Restrict to samples from this
                service id. Defaults to ``None`` (samples from every
                service).

        Returns:
            List[Tuple[int, Any]]: ``(n_event, value)`` pairs, sorted by
            ``n_event`` ascending. Empty if no matching samples exist.
        """
        df = self._frame[self._frame["callback_id"] == metric_name]
        if service_id is not None:
            df = df[df["service_id"] == service_id]
        df = df.sort_values("n_event")
        return [(int(n), v) for n, v in zip(df["n_event"], df["value"])]

    def final(self, metric_name: str, service_id: Optional[str] = None) -> Any:
        """Return a metric's last sampled value.

        Args:
            metric_name (str): The metric's registered ``callback_id``.
            service_id (Optional[str]): Restrict to samples from this
                service id. Defaults to ``None`` (samples from every
                service).

        Returns:
            Any: The value of the sample with the largest ``n_event`` (the
            most recent one — the converged value when read after
            ``sim.stop()``), or ``None`` if no matching samples exist.
        """
        samples = self.series(metric_name, service_id)
        return samples[-1][1] if samples else None
