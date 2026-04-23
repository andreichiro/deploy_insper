#!/bin/sh
set -e

"$(dirname "$0")/seed-data.sh" data

artifacts_ready() {
    [ -f "data/06_models/production_imputers.pkl" ] &&
    [ -f "data/06_models/production_outlier_cappers.pkl" ] &&
    [ -f "data/06_models/production_encoders.pkl" ] &&
    [ -f "data/06_models/production_scalers.pkl" ] &&
    [ -f "data/06_models/production_model.pkl" ]
}

run_kedro() {
    if [ -n "${KEDRO_ENV:-}" ]; then
        uv run kedro run --env "$KEDRO_ENV"
    else
        uv run kedro run
    fi
}

if ! artifacts_ready; then
    echo "Artefatos de produção incompletos; treinando pipeline completa…"
    run_kedro
fi

echo "Subindo servidor da API"
exec uv run uvicorn insper_deploy_kedro.api:app --host 0.0.0.0 --port 8000
