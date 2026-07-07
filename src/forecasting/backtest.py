"""Rolling-origin (expanding window) backtesting.

Why not a single train/test split: one split is one sample of forecast
error — it rewards whichever model got lucky on that particular year.
Rolling origin refits at every step: train on [0..t], forecast h steps,
slide, repeat. Every model sees identical folds.

Metric: sMAPE (primary), MAPE (reported). sMAPE because MAPE explodes when
actuals approach zero and asymmetrically punishes over-forecasting; sMAPE is
bounded [0, 200] and symmetric-ish. Neither is defensible for intermittent
demand — that would need pinball loss on quantile forecasts, noted in the
README as the extension.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd


def smape(actual: np.ndarray, forecast: np.ndarray) -> float:
    actual, forecast = np.asarray(actual, float), np.asarray(forecast, float)
    denom = (np.abs(actual) + np.abs(forecast)) / 2
    mask = denom != 0
    return float(np.mean(np.abs(actual - forecast)[mask] / denom[mask]) * 100)


def mape(actual: np.ndarray, forecast: np.ndarray) -> float:
    actual, forecast = np.asarray(actual, float), np.asarray(forecast, float)
    mask = actual != 0
    return float(np.mean(np.abs((actual - forecast) / actual)[mask]) * 100)


@dataclass
class BacktestResult:
    model_name: str
    horizon: int
    n_folds: int
    smape: float
    mape: float
    fold_smapes: list[float]

    def summary(self) -> str:
        return (
            f"{self.model_name:>15}  sMAPE {self.smape:6.2f}%  "
            f"MAPE {self.mape:6.2f}%  ({self.n_folds} folds, h={self.horizon}, "
            f"fold sMAPE range {min(self.fold_smapes):.1f}-{max(self.fold_smapes):.1f})"
        )


def rolling_origin(
    y: pd.Series,
    model_cls,
    horizon: int = 6,
    min_train: int = 48,
    step: int = 3,
) -> BacktestResult:
    """Expanding-window backtest. min_train=48 (4 seasons) so every fit sees
    enough seasonal cycles; step=3 trades fold count against runtime."""
    actuals, forecasts, fold_smapes = [], [], []
    n = len(y)
    n_folds = 0
    for t in range(min_train, n - horizon + 1, step):
        model = model_cls().fit(y.iloc[:t])
        fc = model.forecast(horizon)
        act = y.iloc[t:t + horizon].to_numpy()
        actuals.append(act)
        forecasts.append(fc)
        fold_smapes.append(smape(act, fc))
        n_folds += 1

    all_a = np.concatenate(actuals)
    all_f = np.concatenate(forecasts)
    return BacktestResult(
        model_name=model_cls().name,
        horizon=horizon,
        n_folds=n_folds,
        smape=smape(all_a, all_f),
        mape=mape(all_a, all_f),
        fold_smapes=fold_smapes,
    )
