# Original project narrative (historical)

This preserves the earlier design and example results. The root README and current tests control implementation status. Planned integrations and old test counts below are not completion claims.

# Demand Forecasting & Serving | Measured backtests and Cloud Run

Monthly demand forecasting on a real public benchmark (Quebec car sales,
108 months, strong seasonality + trend), with the part most forecasting
portfolios skip: an honest rolling-origin evaluation where the naive
baseline almost wins.

## Measured results (reproducible: `pytest` + the snippet below)

```
            ets  sMAPE   9.92%   MAPE  10.01%   (19 folds, h=6)
         sarima  sMAPE  10.22%   MAPE  10.47%
 seasonal_naive  sMAPE  10.23%   MAPE   9.66%
```

The headline finding is the honest one: **ETS beats "same month last year"
by 0.3 points of sMAPE.** On a stable seasonal series that's typical, and
it's the argument for always shipping the baseline comparison — a fancier
model that hadn't been backtested against seasonal naive would have claimed
all 10% as skill. (Also note MAPE ranks naive best while sMAPE ranks it
worst — metric choice changes the leaderboard; that's why the metric is
declared primary before evaluating.)

The ets-beats-naive claim is enforced as a regression test, so if new data
flips the champion, CI says so.

## Evaluation design

Rolling-origin (expanding window): train on [0..t], forecast 6 months,
slide 3, repeat — 19 folds, every model on identical folds; min_train = 48
months so every fit sees 4 full seasonal cycles. Single train/test splits
reward luck; fold-level sMAPE here ranges 3.6-17.5%, which is exactly why.

## Serving

FastAPI app (`/forecast?h=6`, `/health`), fit-at-startup, tested with an
in-process client (9 tests total). Deployed via **Cloud Run, deliberately
not a Vertex AI endpoint**: a Vertex endpoint bills a provisioned node 24/7,
which buys nothing for a <1s-refit CPU model; Cloud Run scales to zero. The
trade-off (and when Vertex wins: GPUs, traffic splitting, model registry
governance) is in `src/forecasting/serve.py`'s docstring. BigQuery ML
ARIMA_PLUS training/eval/forecast SQL is in `sql/` as the managed
comparison to run on GCP.

## Run

```bash
pip install -r requirements.txt
PYTHONPATH=src pytest tests/ -q          # 9 tests incl. API + champion regression
PYTHONPATH=src python -c "
from forecasting.data import load_series
from forecasting.models import ALL_MODELS
from forecasting.backtest import rolling_origin
y = load_series()
for m in ALL_MODELS: print(rolling_origin(y, m).summary())"
PYTHONPATH=src uvicorn forecasting.serve:app --port 8080   # local API
./deploy/deploy_cloud_run.sh                               # on your machine
```

`docs/MONITORING_AND_RETRAINING.md` covers post-deploy: error tracked
against actuals monthly, drift on inputs, quarterly champion/challenger via
the same backtest.
