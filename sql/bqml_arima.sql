-- BigQuery ML ARIMA_PLUS — the managed comparison to the local champion.
-- Prereq: load data/monthly_car_sales.csv to data-portfolio-497401.forecasting.car_sales
--   bq mk --dataset data-portfolio-497401:forecasting
--   bq load --autodetect forecasting.car_sales data/monthly_car_sales.csv

-- 1. Train (auto-ARIMA + holiday/seasonality handling built in)
CREATE OR REPLACE MODEL forecasting.car_sales_arima
OPTIONS (
  model_type = 'ARIMA_PLUS',
  time_series_timestamp_col = 'month_ts',
  time_series_data_col = 'sales',
  horizon = 6,
  auto_arima = TRUE
) AS
SELECT PARSE_TIMESTAMP('%Y-%m', Month) AS month_ts, Sales AS sales
FROM forecasting.car_sales;

-- 2. Evaluate — BQML's holdout metrics; compare to the local rolling-origin
--    numbers in the README (they measure different things: BQML evaluates on
--    the end of the series once; rolling origin averages 19 folds).
SELECT * FROM ML.ARIMA_EVALUATE(MODEL forecasting.car_sales_arima);

-- 3. Forecast with intervals
SELECT
  forecast_timestamp,
  ROUND(forecast_value, 1)      AS forecast,
  ROUND(prediction_interval_lower_bound, 1) AS lo_95,
  ROUND(prediction_interval_upper_bound, 1) AS hi_95
FROM ML.FORECAST(MODEL forecasting.car_sales_arima,
                 STRUCT(6 AS horizon, 0.95 AS confidence_level));
