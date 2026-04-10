FROM python:3.13.12-slim AS base

COPY --from=ghcr.io/astral-sh/uv:0.10.7 /uv /uvx /usr/local/bin/

WORKDIR /app

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    KEDRO_LOGGING_CONFIG=/app/conf/logging.yml \
    UV_COMPILE_BYTECODE=1 \
    UV_LINK_MODE=copy

COPY pyproject.toml uv.lock ./
RUN uv sync --frozen --no-dev --no-install-project

COPY src/ src/
COPY README.md ./
COPY conf/ conf/
COPY data/ data/
COPY data/ /opt/app-seed/data/
COPY entrypoint.sh dashboard-entrypoint.sh seed-data.sh ./

RUN chmod +x entrypoint.sh dashboard-entrypoint.sh seed-data.sh

RUN uv sync --frozen --no-dev

FROM base AS dev

ENV UV_PROJECT_ENVIRONMENT=/opt/venv \
    PATH="/opt/venv/bin:${PATH}"

RUN uv sync --frozen --extra dev
RUN mkdir -p /workspace && cp -R /app/. /workspace/

WORKDIR /workspace
CMD ["sh"]

FROM base AS runtime

EXPOSE 8000 8501

ENTRYPOINT ["./entrypoint.sh"]
