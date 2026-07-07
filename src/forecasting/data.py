"""Load the demand series (Quebec monthly car sales, 1960-1968 — a standard
public forecasting benchmark: strong yearly seasonality + upward trend).
Any monthly CSV with (Month, Sales) columns drops in."""

from __future__ import annotations

import pathlib

import pandas as pd

DATA = pathlib.Path(__file__).resolve().parents[2] / "data" / "monthly_car_sales.csv"


def load_series(path: pathlib.Path = DATA) -> pd.Series:
    df = pd.read_csv(path)
    idx = pd.PeriodIndex(df["Month"], freq="M").to_timestamp()
    s = pd.Series(df["Sales"].to_numpy(), index=idx, name="sales").asfreq("MS")
    if s.isna().any():
        raise ValueError("gaps in the monthly series — fix upstream, don't impute silently")
    return s
