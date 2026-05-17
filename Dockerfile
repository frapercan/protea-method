# LAFA submission image for the protea-method inference library.
#
# Pure inference path: KNN search, feature compute, reranker apply.
# No FastAPI, no SQLAlchemy, no protea-core. Heavy numerical deps
# (numpy, pandas, lightgbm, faiss-cpu) installed in the main group;
# they ship CPU-only wheels here. GPU variants are not part of this
# image (deferred to T-OPS.12 / protea-method-runtime).
#
# Entrypoint is ``method_main.py`` (the LAFA standard interface).
# Run with:
#
#   docker run \
#       -v ./data:/app/data:ro \
#       -v ./output:/app/output:rw \
#       protea-method-lafa:latest \
#       --query_file /app/data/test_sequences.fasta \
#       --train_sequences /app/data/train_sequences.fasta \
#       --annot_file /app/data/goa_uniprot_sprot.gaf.gz \
#       --graph /app/data/go-basic.obo \
#       --output_file /app/output/predictions.tsv.gz \
#       --query_embeds /app/data/query.parquet \
#       --reference_embeds /app/data/reference.parquet
#
# See ``docker/example_run.sh`` for a copy-pasteable invocation.

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
COPY method_main.py ./method_main.py

ENV PYTHONUNBUFFERED=1 \
    PYTHONPATH=/app/src

# LAFA standard interface entrypoint. CMD is intentionally empty: the
# caller supplies all CLI args via ``docker run``.
ENTRYPOINT ["python3", "method_main.py"]
