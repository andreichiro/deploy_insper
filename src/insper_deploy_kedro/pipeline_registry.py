"""Registro dos pipelines do projeto."""

from kedro.framework.project import find_pipelines
from kedro.pipeline import Pipeline


def register_pipelines() -> dict[str, Pipeline]:
    """Registra os pipelines do projeto e retorna o mapeamento nome -> Pipeline.

    O __default__ inclui só DE + modelling + refit (treino completo)

    Inferência roda separada com --pipeline inference pq precisa de um arquivo
    de entrada próprio (raw_data_inference)
    """
    pipelines = find_pipelines(raise_errors=True)
    pipelines["__default__"] = (
        pipelines["data_engineering"]
        + pipelines["modelling"]
        + pipelines["refit"]
    )
    return pipelines
