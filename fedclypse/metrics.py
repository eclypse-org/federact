# -*- coding: utf-8 -*-
from __future__ import annotations

from typing import Any, List, Optional, Tuple

from eclypse.report.metrics import metric


@metric.service(remote=True, name="fedclypse_round")
def round_metric(service: Any) -> Optional[int]:
    """A ready-made remote metric: samples an entity's ``round`` on the worker.

    Write your own observables the same way — ``@metric.service(remote=True)`` gives
    the callback the live entity, so it can read ``self.model``, ``self.last_loss``,
    etc. Samples are read back with ``History``.
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
        self._frame = frame

    @classmethod
    def from_report(cls, report: Any) -> "History":
        return cls(report.service())

    def series(
        self, metric_name: str, service_id: Optional[str] = None
    ) -> List[Tuple[int, Any]]:
        df = self._frame[self._frame["callback_id"] == metric_name]
        if service_id is not None:
            df = df[df["service_id"] == service_id]
        df = df.sort_values("n_event")
        return [(int(n), v) for n, v in zip(df["n_event"], df["value"])]

    def final(self, metric_name: str, service_id: Optional[str] = None) -> Any:
        samples = self.series(metric_name, service_id)
        return samples[-1][1] if samples else None
