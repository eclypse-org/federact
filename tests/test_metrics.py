# -*- coding: utf-8 -*-
import pandas as pd

from fedclypse.metrics import History


def _frame(rows):
    return pd.DataFrame(rows, columns=["callback_id", "service_id", "n_event", "value"])


def test_history_series_is_sorted_by_n_event():
    f = _frame([("m", "server", 2, 20.0), ("m", "server", 1, 10.0)])
    assert History(f).series("m", "server") == [(1, 10.0), (2, 20.0)]


def test_history_final_returns_last_value():
    f = _frame([("m", "server", 1, 10.0), ("m", "server", 2, 20.0)])
    assert History(f).final("m", "server") == 20.0


def test_history_filters_by_service_id():
    f = _frame([("m", "server", 1, 1.0), ("m", "client_0", 1, 9.0)])
    assert History(f).series("m", "server") == [(1, 1.0)]


def test_history_final_is_none_when_metric_absent():
    f = _frame([("m", "server", 1, 1.0)])
    assert History(f).final("other", "server") is None
