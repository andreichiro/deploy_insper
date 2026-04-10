"""Comandos do Kedro ao invés de ficar consultando toda hora"""

import click


@click.group(name="custom")
def cli():
    """Comandos do projeto de predição de diabetes."""

@cli.command()
def cheatsheet():
    """Mostra referência rápida de todos os comandos do projeto"""
    click.echo(
        """
               PROJETO DIABETES, COLA RÁPIDA                     ║

Execução de Pipelines
─────────────────
  uv run kedro run                                    # pipeline completo
  uv run kedro run --pipeline data_engineering        # limpar -> validar(GE) -> features -> split -> validar(GE) -> encode -> scale
  uv run kedro run --pipeline modelling               # otimizar 3 modelos (Optuna) -> avaliar -> selecionar -> relatório
  uv run kedro run --pipeline refit                   # retreinar vencedor com todos os dados -> calibrar probabilidades
  uv run kedro run --pipeline inference               # transformar CSV de inferência -> prever

Dashboard Streamlit
─────────────────
  uv run streamlit run src/insper_deploy_kedro/dashboard.py  # dashboard completo (métricas + inferência)

Servidor da API
─────────────────
  uv run uvicorn insper_deploy_kedro.api:app --host 0.0.0.0 --port 8000
  open http://localhost:8000/docs                     # docs Swagger
  curl http://localhost:8000/health                   # health check
  curl -X POST http://localhost:8000/inference \\
    -H "Content-Type: application/json" \\
    -H "X-API-Key: sua-chave" \\
    -d '{"instances": [{"Pregnancies":6,"Glucose":148,"BloodPressure":72,"SkinThickness":35,"Insulin":0,"BMI":33.6,"DiabetesPedigreeFunction":0.627,"Age":50}]}'

Desenvolvimento
─────────────────
  uv sync --all-extras                                # instalar todas as deps
  uv run ruff check src/ tests/                       # lint (verificar)
  uv run ruff check --fix src/ tests/                 # lint (corrigir auto)
  uv run ruff format src/ tests/                      # formatar
  uv run ruff format --check src/ tests/              # formatar (dry-run)

Testes
─────────────────

  uv run pytest                                       # todos os testes + coverage
  uv run pytest -v                                    # detalhado
  uv run pytest tests/test_api.py                     # testes da API
  uv run pytest tests/test_nodes_data_engineering.py  # testes dos nodes DE
  uv run pytest tests/test_nodes_modelling.py         # testes dos nodes de modelagem
  uv run pytest tests/test_nodes_inference.py         # testes dos nodes de inferência
  uv run pytest --no-cov -x                           # sem coverage, parar no 1o erro

Docker (API + Dashboard)
─────────────────
  docker compose up --build                            # API (8000) + Dashboard (8501)
  API_KEY=meu-segredo docker compose up --build
  docker compose up --build api                        # só a API
  docker compose up --build dashboard                  # só o dashboard

Utilitários Kedro
─────────────────
  uv run kedro viz run                                # visualizar DAG
  uv run kedro registry list                          # listar pipelines
  uv run kedro catalog list                           # listar datasets
  uv run kedro run --nodes optimize_baseline_node     # rodar node específico
  uv run kedro run --from-nodes select_best_model_node
  uv run kedro run --to-nodes evaluate_baseline_node
"""
    )
