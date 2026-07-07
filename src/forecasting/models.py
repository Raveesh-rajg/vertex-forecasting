"""Forecasting models behind one interface: fit(history) -> forecast(h).

Model choices and why:
  * SEASONAL NAIVE — the baseline every paper forgets and every honest
    evaluation needs: "same month last year". If a model can't beat this,
    the model is decoration.
  * ETS (Holt-Winters) — additive trend + multiplicative seasonality; the
    workhorse for monthly retail-style demand. Small, fast, interpretable.
  * SARIMA — (1,1,1)x(1,1,1,12) default; included as the classical
    alternative and to show the comparison, not because it usually wins.

BigQuery ML's ARIMA_PLUS (sql/bqml_arima.sql) is the managed equivalent —
the local backtest picks the champion; BQML/Vertex serves it.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


class SeasonalNaive:
    name = "seasonal_naive"

    def __init__(self, season_length: int = 12):
        self.m = season_length

    def fit(self, y: pd.Series) -> "SeasonalNaive":
        if len(y) < self.m:
            raise ValueError(f"need >= {self.m} observations")
        self._last_season = y.iloc[-self.m:].to_numpy()
        return self

    def forecast(self, h: int) -> np.ndarray:
        reps = int(np.ceil(h / self.m))
        return np.tile(self._last_season, reps)[:h]


class ETS:
    name = "ets"

    def __init__(self, season_length: int = 12):
        self.m = season_length

    def fit(self, y: pd.Series) -> "ETS":
        from statsmodels.tsa.holtwinters import ExponentialSmoothing

        self._fit = ExponentialSmoothing(
            y.to_numpy(),
            trend="add",
            seasonal="mul",          # retail seasonality scales with level
            seasonal_periods=self.m,
            initialization_method="estimated",
        ).fit(optimized=True)
        return self

    def forecast(self, h: int) -> np.ndarray:
        return np.asarray(self._fit.forecast(h))


class SARIMA:
    name = "sarima"

    def __init__(self, order=(1, 1, 1), seasonal_order=(1, 1, 1, 12)):
        self.order = order
        self.seasonal_order = seasonal_order

    def fit(self, y: pd.Series) -> "SARIMA":
        import warnings

        from statsmodels.tsa.statespace.sarimax import SARIMAX

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            self._fit = SARIMAX(
                y.to_numpy(),
                order=self.order,
                seasonal_order=self.seasonal_order,
            ).fit(disp=False)
        return self

    def forecast(self, h: int) -> np.ndarray:
        return np.asarray(self._fit.forecast(h))


ALL_MODELS = [SeasonalNaive, ETS, SARIMA]
