# Demand Forecasting & Serving

Backtest seasonal naive, ETS and SARIMA on a public monthly car-sales series, then serve forecasts through FastAPI.

## Implementation and validation

Nine local model and API tests pass. Cloud Run deployment and BigQuery ML are supplied as integration paths; neither was validated in the selected GCP account.

Automated checks: **9 tests**. The GitHub Actions run linked above the file browser is the current CI result. Local checks and external integrations are separate claims.

## Reproduce locally

Use Python 3.12. Run from this repository’s root in a fresh virtual environment.

```sh
python -m venv .venv
# Activate .venv for your shell, then:
python -m pip install -r requirements.txt
```

For repositories using `src/`, set the import path before running commands:

```powershell
# PowerShell
$env:PYTHONPATH="src"
```
```sh
# macOS/Linux
export PYTHONPATH=src
```

```sh
python -m pytest tests -q
```

## Open the local application

```sh
python -m uvicorn forecasting.serve:app --port 8080
```

## Data and interpretation

Public historical Quebec car-sales benchmark, 108 monthly observations. Small backtest differences are benchmark-specific; no production demand or future accuracy is guaranteed.

## Inspect the work

- [`tests/`](tests/) — executable checks and examples.
- [`docs/`](docs/) — methodology, integration specifications and the historical design.
- [Portfolio](https://raveesh-rajg.github.io/) — project directory.

## Completion boundary

Passing local tests establishes the checks listed in this repository. It does not establish cloud deployment, real-data quality, production security, or native BI rendering unless an explicit verification record says so.
