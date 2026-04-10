# Deploy de Predição de Diabetes

[![Powered by Kedro](https://img.shields.io/badge/powered_by-kedro-ffc900?logo=kedro)](https://kedro.org)

Repositório standalone de deploy da aplicação de predição de diabetes. Ele reúne o projeto Kedro pronto para produção, a API FastAPI, o dashboard Streamlit, a configuração declarativa em YAML, as dependências travadas com `uv`, o setup Docker e os artefatos necessários para subir a stack imediatamente.

## TLDR

### Local

```bash
uv sync
uv run kedro run
uv run uvicorn insper_deploy_kedro.api:app --host 0.0.0.0 --port 8000
uv run streamlit run src/insper_deploy_kedro/dashboard.py
uv run --extra dev pytest
```

### Docker

```bash
docker compose up --build
```

Endpoints principais:

- API: <http://localhost:8000>
- Swagger: <http://localhost:8000/docs>
- Health da API: <http://localhost:8000/health>
- Dashboard: <http://localhost:8501>
- Health do Streamlit: <http://localhost:8501/_stcore/health>

## Resumo rápido do que foi implementado

### Modelo

- seleção de features orientada por blocos, usando apenas treino, CV interna e regra de 1 erro-padrão
- otimização de hiperparâmetros com Optuna para Logistic Regression, CatBoost e XGBoost
- comparação entre políticas clínicas de falso positivo vs falso negativo
- artefatos de robustez: métricas por fold, resumo de variação, bootstrap, permutation importance e auditoria de sensibilidade a perturbações
- saída de inferência transformada em relatório de risco com `prediction`, `prediction_proba`, `risk_score` e `risk_band`

### Pipelines Kedro

- `data_engineering`: limpeza, validação, feature engineering, split, encoding e scaling
- `modelling`: seleção de features, tuning, avaliação, escolha do melhor modelo, políticas, robustez e manifestos
- `refit`: retreino do bundle de produção com todos os dados
- `inference`: caminho real de inferência usado tanto pela API quanto pelo Streamlit

### FastAPI

- endpoints de health/readiness
- inferência online via o caminho real do Kedro
- validação de request/response com Pydantic
- jobs em background com persistência local e rastreio de erro/log

### Streamlit

- visão geral do bundle atual
- comparação de modelos
- robustez, políticas clínicas e relatórios de risco
- manifestos e contrato de inferência
- inferência ao vivo usando os mesmos artefatos de produção da API

### Configuração declarativa

- classes, funções, hiperparâmetros, CV, métricas, thresholds, políticas e validações são dirigidos por YAML
- o código instancia tudo via `class_path` / `function_path` quando aplicável

### Validação e qualidade

- Great Expectations no pós-limpeza e pós-split
- contratos simples de dados, frescor e drift materializados em artefatos
- suíte de testes em `tests/`

### Reprodutibilidade com Docker

- imagem com o código completo, `pyproject.toml`, `uv.lock`, configuração Kedro base e scripts de entrypoint
- bundle atual de produção e artefatos versionados dentro do repositório
- seed automático dos CSVs raw quando o volume persistente está vazio
- suporte a runtime e workspace de desenvolvimento

## O que existe neste repositório

- código da aplicação em `src/`
- configuração compartilhada do Kedro em `conf/base/`
- configuração de CI/testes em `conf/ci/`
- arquivos raw em `data/01_raw/`
- artefatos e outputs atuais em `data/`
- testes em `tests/`
- documentação em `docs/`
- setup reprodutível com `Dockerfile`, `docker-compose.yml`, `entrypoint.sh` e `dashboard-entrypoint.sh`
- dependências travadas via `pyproject.toml` e `uv.lock`

## Como subir localmente

### 1. Instalar dependências

```bash
uv sync
```

### 2. Materializar os artefatos principais

```bash
uv run kedro run
```

### 3. Subir a API

```bash
uv run uvicorn insper_deploy_kedro.api:app --host 0.0.0.0 --port 8000
```

### 4. Subir o dashboard

```bash
uv run streamlit run src/insper_deploy_kedro/dashboard.py
```

### 5. Rodar os testes

```bash
uv run --extra dev pytest
```

## Como subir com Docker

```bash
docker compose up --build
```

Isso sobe:

- `api` em `http://localhost:8000`
- `dashboard` em `http://localhost:8501`

A imagem já carrega:

- o código completo do runtime
- o conjunto travado de dependências
- a configuração base do Kedro
- os CSVs raw usados para seed/bootstrap
- o bundle atual de produção
- os artefatos de reporting materializados

Se o volume persistente estiver vazio, o bootstrap copia automaticamente os raw CSVs antes da aplicação começar a servir.

Também existe um profile de desenvolvimento:

```bash
docker compose --profile dev up --build workspace
```

Esse profile monta o código em `/workspace`, instala as dependências de desenvolvimento e compartilha o volume `app-data`.

## Estado atual do bundle de produção versionado

Os manifestos atuais apontam para:

- família do modelo: `catboost.CatBoostClassifier`
- política de deploy: `prioritize_recall`
- threshold de decisão: `0.15`
- splits usados no bundle final: `train`, `validation`, `test`
- features selecionadas no último treino: `Glucose`, `BMI`, `DiabetesPedigreeFunction`, `Age`
- número de combinações de features avaliadas: `957`

Esses valores estão materializados em:

- `data/09_ops/latest_training_run_manifest.json`
- `data/09_ops/latest_serving_manifest.json`
- `data/09_ops/latest_inference_contract.json`

## Pipelines Kedro

### `data_engineering`

- `clean_data`
- `add_features`
- validação com Great Expectations
- `add_split_column`
- nova validação pós-split
- `fit_encoders` / `transform_encoders`
- `fit_scalers` / `transform_scalers`

### `modelling`

- seleção de features
- treino e tuning dos candidatos
- avaliação de validação
- scorecard de seleção do melhor modelo
- comparação de políticas clínicas
- robustez, bootstrap, importance e sensibilidade
- manifestos e registry local
- avaliação em teste

### `refit`

- retreino do bundle de produção com todos os dados
- calibração quando configurada
- geração dos artefatos `production_*`

### `inference`

- limpeza
- features
- encoding
- scaling
- predição
- geração do relatório de risco

## FastAPI

A API usa Pydantic para schemas e validação de entrada/saída. O caminho de inferência online chama a pipeline real do Kedro, sem duplicar manualmente o fluxo de transformação.

Principais pontos:

- `/health` só fica pronto quando o bundle completo está disponível:
  - `production_encoders.pkl`
  - `production_scalers.pkl`
  - `production_model.pkl`
- `/inference` recebe lotes e devolve score/risk report
- jobs longos podem rodar em background
- logs e tracebacks usam a configuração central do projeto

Exemplo de request:

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

## Dashboard Streamlit

O dashboard cobre:

- visão geral do estado atual do bundle
- comparação de modelos
- leitura de robustez e estabilidade
- políticas clínicas de FP vs FN
- manifestos e contrato de inferência
- predição ao vivo
- score/risk report materializado a partir da inferência batch

## Configuração YAML driven

Arquivos centrais:

- `conf/base/catalog.yml`
- `conf/base/parameters/data_engineering.yml`
- `conf/base/parameters/modelling.yml`
- `conf/base/parameters/refit.yml`
- `conf/base/parameters/data_quality.yml`

O que está declarativo:

- encoder/scaler
- split e preprocessing
- runtime de CV
- objetivos e search spaces do Optuna
- métricas de avaliação
- políticas clínicas e thresholds
- calibração
- Great Expectations

## Great Expectations

Great Expectations roda em dois momentos:

1. pós-limpeza
2. pós-split

Se uma validação crítica falhar, o pipeline para. Isso ajuda a impedir que o modelo avance com schema quebrado, ranges absurdos ou splits problemáticos.

## Estrutura do projeto

```text
deploy/aula_2
├── conf/
│   ├── base/
│   └── ci/
├── data/
├── docs/
├── src/insper_deploy_kedro/
│   ├── api.py
│   ├── dashboard.py
│   ├── registry.py
│   ├── serving_runtime.py
│   └── pipelines/
│       ├── data_engineering/
│       ├── modelling/
│       ├── inference/
│       └── refit/
├── tests/
├── Dockerfile
├── docker-compose.yml
├── pyproject.toml
└── uv.lock
```

## Arquivos que devem ficar locais

Não versionar:

- `conf/local/credentials.yml`
- `.env`
- ambientes virtuais locais
- caches locais
- outputs locais de `catboost_info/`
- artefatos efêmeros gerados fora do conjunto que você decidiu versionar

## Observações operacionais

- API e Streamlit chamam o mesmo caminho de inferência do Kedro
- o repositório inclui artefatos atuais de produção para servir imediatamente
- rodar novamente o pipeline de treino atualiza arquivos em `data/`
- os manifestos versionados não dependem de um nome de usuário hardcoded de deploy
