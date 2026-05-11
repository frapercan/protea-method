# Slim runtime image for the protea-method inference library.
#
# Pure inference path: KNN search, feature compute, reranker apply.
# No FastAPI, no SQLAlchemy, no protea-core. Heavy numerical deps
# (numpy, pandas, lightgbm, faiss-cpu) installed in the main group;
# they ship CPU-only wheels here. GPU variants are not part of this
# image — that belongs to T-OPS.12 (protea-method-runtime).

FROM python:3.12-slim AS builder

RUN apt-get update && apt-get install -y --no-install-recommends \
        build-essential \
        git \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

RUN pip install --no-cache-dir poetry==2.3.2

COPY pyproject.toml README.md ./
COPY poetry.lock* ./
RUN poetry config virtualenvs.create false \
    && poetry install --only main --no-root --no-interaction --no-ansi

COPY src/ ./src/
RUN poetry install --only main --no-interaction --no-ansi

FROM python:3.12-slim

# faiss-cpu and numpy/pandas need libgomp at runtime.
RUN apt-get update && apt-get install -y --no-install-recommends \
        libgomp1 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY --from=builder /usr/local/lib/python3.12/site-packages /usr/local/lib/python3.12/site-packages
COPY --from=builder /usr/local/bin /usr/local/bin

COPY src/ ./src/

ENV PYTHONUNBUFFERED=1 \
    PYTHONPATH=/app/src

# Library image: importable as protea_method. Override CMD to run
# pipeline scripts from downstream.
CMD ["python", "-c", "import protea_method; print('protea-method', protea_method.__name__, 'ready')"]
