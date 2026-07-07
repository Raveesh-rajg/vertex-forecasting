# Monitoring & retraining (the MLOps-lite part)

## What to monitor, in order of importance
1. **Forecast error, measured late.** Each month when actuals land, score
   last month's forecast (sMAPE). One BigQuery scheduled query appending to
   forecasting.forecast_accuracy_log; alert threshold: trailing-3-month
   sMAPE > backtest sMAPE + 5pp (i.e., worse than the 9.9% baseline by a
   margin, not by noise — fold range in the backtest was 3.6-17.5%, so a
   single bad month is expected, a bad quarter is drift).
2. **Input drift, measured early.** Level shift or seasonality change in the
   inputs shows up before error does: compare trailing-12m mean/std to
   training-window values.
3. **Serving health**: Cloud Run gives 5xx rate/latency out of the box.

## Retraining policy
Statsmodels ETS refits in <1s — the "retraining pipeline" is: monthly
scheduled job appends the new actual, redeploys (Cloud Build trigger).
Champion/challenger: rerun the rolling-origin backtest on the extended
series each quarter; if seasonal_naive or SARIMA overtakes ETS, the
backtest report IS the promotion decision, checked into the repo.

## What deliberately isn't here
Vertex AI Pipelines, feature store, model registry — at one small model,
each is infrastructure without a customer. The signal that changes the
answer: multiple series (per-product), GPU models, or humans needing
approval gates. Written down so the omission reads as judgment, not gap.
