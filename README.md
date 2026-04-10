# Diabetes Prediction Deployment

[![Powered by Kedro](https://img.shields.io/badge/powered_by-kedro-ffc900?logo=kedro)](https://kedro.org)

Standalone deployment repository for the diabetes prediction application.
It includes the production-ready Kedro project, FastAPI service, Streamlit dashboard,
locked Python dependencies, Docker setup, and the data/artifacts needed to boot the stack immediately.

## What is included

- application code in `src/`
- shared Kedro configuration in `conf/base/`
- CI/test Kedro config in `conf/ci/`
- raw input files in `data/01_raw/`
- current production artifacts and reporting outputs in `data/`
- tests in `tests/`
- docs in `docs/`
- reproducible container setup with `Dockerfile`, `docker-compose.yml`, and entrypoint scripts
- locked dependencies via `pyproject.toml` and `uv.lock`

## Quickstart

### Local

```bash
uv sync
uv run uvicorn insper_deploy_kedro.api:app --host 0.0.0.0 --port 8000
uv run streamlit run src/insper_deploy_kedro/dashboard.py
uv run --extra dev pytest
```

### Docker

```bash
docker compose up --build
```

This starts:

- `api` on `http://localhost:8000`
- `dashboard` on `http://localhost:8501`

The container image already contains:

- the full runtime codebase
- the locked dependency set
- base Kedro configuration
- raw CSVs for seed/bootstrap
- the current production model bundle and reporting artifacts

If the persistent volume is empty, startup seeds the raw data automatically before serving.

## Health checks

- API: `http://localhost:8000/health`
- Streamlit: `http://localhost:8501/_stcore/health`

Example inference request:

```bash
curl -X POST http://localhost:8000/inference \
  -H 'Content-Type: application/json' \
  -d '{
    "instances": [
      {
        "Pregnancies": 6,
        "Glucose": 98,
        "BloodPressure": 58,
        "SkinThickness": 33,
        "Insulin": 190,
        "BMI": 34,
        "DiabetesPedigreeFunction": 0.43,
        "Age": 43
      }
    ]
  }'
```

The API only reports ready when the complete production inference bundle is available:

- `production_encoders.pkl`
- `production_scalers.pkl`
- `production_model.pkl`

## Current production bundle

The committed serving state currently points to:

- model family: `catboost.CatBoostClassifier`
- deployment policy: `prioritize_recall`
- decision threshold: `0.15`
- selected features: `Glucose`, `BMI`, `DiabetesPedigreeFunction`, `Age`

These values are materialized in:

- `data/09_ops/latest_training_run_manifest.json`
- `data/09_ops/latest_serving_manifest.json`
- `data/09_ops/latest_inference_contract.json`

## Runtime notes

- Great Expectations is used for post-cleaning and post-split data validation.
- FastAPI and Streamlit both call the real Kedro inference path.
- The repo includes current production artifacts so the stack can serve immediately.
- Running the training pipeline again will update files under `data/`.

## Local-only files

Keep these out of the repository:

- `conf/local/credentials.yml`
- `.env`
- local virtual environments
- local cache directories
- local `catboost_info/` output
