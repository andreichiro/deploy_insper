"""Pipeline de refit: re-fita encoders/scalers/modelo com TODOS os dados pra produção.

Reutiliza nodes de DE e modelagem com split_to_fit: [train, validation, test].
Add calibração de probabilidade como pós-processamento
"""

from typing import Any

from kedro.pipeline import Pipeline, node, pipeline

from insper_deploy_kedro.pipelines.data_engineering.nodes import (
    fit_encoders,
    fit_scalers,
    transform_encoders,
    transform_scalers,
)
from insper_deploy_kedro.pipelines.modelling.nodes import calibrate_model, train_model
from insper_deploy_kedro.registry import (
    build_inference_contract,
    build_serving_manifest,
    record_model_registry_entry,
)


def create_pipeline(**kwargs: Any) -> Pipeline:
    """Monta o DAG de refit: mesmas funções, todos os dados, c/ calibração."""
    return pipeline(
        [
            node(
                func=fit_encoders,
                inputs=[
                    "split_data",
                    "params:columns",
                    "params:refit_fit_transform",
                    "params:preprocessing",
                ],
                outputs="production_encoders",
                name="refit_encoders_node",
                tags=["refit", "encoding"],
            ),
            node(
                func=transform_encoders,
                inputs=["split_data", "production_encoders"],
                outputs="production_encoded_data",
                name="refit_transform_encoders_node",
                tags=["refit", "encoding"],
            ),
            node(
                func=fit_scalers,
                inputs=[
                    "production_encoded_data",
                    "params:columns",
                    "params:refit_fit_transform",
                    "params:preprocessing",
                ],
                outputs="production_scalers",
                name="refit_scalers_node",
                tags=["refit", "scaling"],
            ),
            node(
                func=transform_scalers,
                inputs=["production_encoded_data", "production_scalers"],
                outputs="production_master_table",
                name="refit_transform_scalers_node",
                tags=["refit", "scaling"],
            ),
            node(
                func=train_model,
                inputs=[
                    "production_master_table",
                    "selected_feature_columns",
                    "best_model_config",
                    "params:ml_runtime",
                ],
                outputs="raw_production_model",
                name="refit_model_node",
                tags=["refit", "training"],
            ),
            node(
                func=calibrate_model,
                inputs=[
                    "production_master_table",
                    "raw_production_model",
                    "params:columns",
                    "params:calibration",
                ],
                outputs="production_model",
                name="calibrate_model_node",
                tags=["refit", "calibration"],
            ),
            node(
                func=record_model_registry_entry,
                inputs=["production_model"],
                outputs="latest_model_registry_entry",
                name="record_model_registry_entry_node",
                tags=["refit", "registry"],
            ),
            node(
                func=build_serving_manifest,
                inputs=[
                    "production_model",
                    "latest_model_registry_entry",
                ],
                outputs="latest_serving_manifest",
                name="build_serving_manifest_node",
                tags=["refit", "registry", "governance"],
            ),
            node(
                func=build_inference_contract,
                inputs=[
                    "production_model",
                    "params:raw_columns",
                    "selected_feature_columns",
                ],
                outputs="latest_inference_contract",
                name="build_inference_contract_node",
                tags=["refit", "registry", "governance"],
            ),
        ]
    )
