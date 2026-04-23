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

echo "Esperando artefatos de produção…"
while ! artifacts_ready; do
    sleep 5
done

echo "Subindo dashboard Streamlit"
exec uv run streamlit run src/insper_deploy_kedro/dashboard.py \
    --server.port=8501 \
    --server.address=0.0.0.0 \
    --server.headless=true
