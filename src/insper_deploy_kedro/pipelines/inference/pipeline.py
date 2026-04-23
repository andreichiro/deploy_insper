"""Pipeline de inferência: limpar -> features -> transform (sem fit) -> predict.

Reutiliza clean_data, add_features, transform_encoders, transform_scalers do DE.
Só predict é novo. Nenhum fit acontece aqui.
"""

from typing import Any

from kedro.pipeline import Pipeline, node, pipeline

from insper_deploy_kedro.pipelines.data_engineering.nodes import (
    add_features,
    clean_data,
    transform_encoders,
    transform_outlier_cappers,
    transform_scalers,
    transform_zero_imputers,
)

from .nodes import build_risk_report, predict


def create_pipeline(**kwargs: Any) -> Pipeline:
    """Monta o DAG de inferência — só transform, sem fit."""
    return pipeline(
        [
            node(
                func=clean_data,
                inputs=["raw_data_inference", "params:inference_raw_columns"],
                outputs="cleaned_inference",
                name="clean_inference_node",
                tags=["inference", "cleaning"],
            ),
            node(
                func=transform_zero_imputers,
                inputs=["cleaned_inference", "production_imputers"],
                outputs="imputed_inference",
                name="impute_inference_node",
                tags=["inference", "imputation"],
            ),
            node(
                func=transform_outlier_cappers,
                inputs=["imputed_inference", "production_outlier_cappers"],
                outputs="capped_inference",
                name="cap_outliers_inference_node",
                tags=["inference", "outlier_capping"],
            ),
            node(
                func=add_features,
                inputs=["capped_inference"],
                outputs="featured_inference",
                name="add_features_inference_node",
                tags=["inference", "feature_engineering"],
            ),
            node(
                func=transform_encoders,
                inputs=["featured_inference", "production_encoders"],
                outputs="encoded_inference",
                name="encode_inference_node",
                tags=["inference", "encoding"],
            ),
            node(
                func=transform_scalers,
                inputs=["encoded_inference", "production_scalers"],
                outputs="scaled_inference",
                name="scale_inference_node",
                tags=["inference", "scaling"],
            ),
            node(
                func=predict,
                inputs=["scaled_inference", "production_model"],
                outputs="predictions",
                name="predict_node",
                tags=["inference", "prediction"],
            ),
            node(
                func=build_risk_report,
                inputs=["featured_inference", "predictions", "production_model"],
                outputs="risk_report",
                name="build_risk_report_node",
                tags=["inference", "reporting"],
            ),
        ]
    )
