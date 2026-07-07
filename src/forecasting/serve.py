"""Prediction API — FastAPI app deployable to Cloud Run.

Deployment choice, argued: Cloud Run over a Vertex AI endpoint for THIS
model. A Vertex endpoint bills for a provisioned node 24/7 (~$70+/mo idle)
and shines for GPU models and traffic splitting; a statsmodels ETS refit
takes <1s, the artifact is tiny, and Cloud Run scales to zero — effectively
$0 idle for a portfolio service. The Vertex path (model registry + endpoint)
is documented in deploy/ so the trade-off is a decision, not ignorance.

The model is fit at startup from the bundled series (or CSV_PATH env).
Retraining = redeploy with fresh data; see docs/MONITORING_AND_RETRAINING.md.
"""

from __future__ import annotations

import os
import pathlib
import time

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

from forecasting.data import load_series, DATA
from forecasting.models import ETS

app = FastAPI(title="demand-forecast", version="0.1.0")

_state: dict = {}


@app.on_event("startup")
def _train() -> None:
    csv = pathlib.Path(os.environ.get("CSV_PATH", DATA))
    y = load_series(csv)
    t0 = time.time()
    _state["model"] = ETS().fit(y)
    _state["train_seconds"] = round(time.time() - t0, 3)
    _state["last_obs"] = str(y.index[-1].date())
    _state["n_obs"] = len(y)


class ForecastResponse(BaseModel):
    horizon: int
    forecast: list[float]
    model: str
    trained_on_n_obs: int
    last_observation: str


@app.get("/health")
def health() -> dict:
    return {"status": "ok", "train_seconds": _state.get("train_seconds")}


@app.get("/forecast", response_model=ForecastResponse)
def forecast(h: int = 6) -> ForecastResponse:
    if not 1 <= h <= 24:
        raise HTTPException(422, "h must be 1..24 (monthly model)")
    fc = _state["model"].forecast(h)
    return ForecastResponse(
        horizon=h,
        forecast=[round(float(v), 1) for v in fc],
        model=_state["model"].name,
        trained_on_n_obs=_state["n_obs"],
        last_observation=_state["last_obs"],
    )
