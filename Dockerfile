# LAFA submission image for the protea-method inference library.
#
# Two embedding paths supported by the same image:
#
#   * Bind-mount mode (slim, CI-friendly): supply --query_embeds /
#     --reference_embeds parquet files; the image's lean main install
#     handles KNN + reranker only (no torch, no PLM weights).
#   * Self-contained mode (LAFA-EMB.1): omit the two flags; the image
#     reads each FASTA, computes embeddings via the configured backend
#     (default `esm2_t36_3B`, champion config), caches them under
#     /app/output/.embed_cache, and runs predict end-to-end.
#
# The [esm] extras group is installed by default in the runtime stage so
# the image works in both modes out of the box. Users who want a slim
# bind-mount-only build can pass --build-arg INSTALL_EXTRAS="" at build
# time to skip the torch/transformers install.
#
# Run (bind-mount mode):
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
# Run (self-contained mode):
#
#   docker run \
#       -v ./data:/app/data:ro \
#       -v ./output:/app/output:rw \
#       -v ./hf-cache:/app/.hf-cache:rw \
#       protea-method-lafa:latest \
#       --query_file /app/data/test_sequences.fasta \
#       --train_sequences /app/data/train_sequences.fasta \
#       --annot_file /app/data/goa_uniprot_sprot.gaf.gz \
#       --graph /app/data/go-basic.obo \
#       --output_file /app/output/predictions.tsv.gz \
#       --backend_id esm2_t36_3B
#
# Cold-start cost (self-contained, first run, no HF cache mounted):
#   esm2_t36_3B downloads ~12 GB of weights from HuggingFace (one-off).
#   Mount -v ./hf-cache:/app/.hf-cache:rw to keep the weights between
#   container restarts. The runtime image is ~6 GB; total disk including
#   the downloaded weights is ~18 GB on first run.
#
# See docker/example_run.sh (bind-mount) and
# docker/example_run_selfcontained.sh (self-contained) for
# copy-pasteable invocations.

FROM python:3.12-slim AS builder

ARG INSTALL_EXTRAS="esm"

RUN apt-get update && apt-get install -y --no-install-recommends \
        build-essential \
        git \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

RUN pip install --no-cache-dir poetry==2.3.2

COPY pyproject.toml README.md ./
COPY poetry.lock* ./

# Phase 1: install the slim main group (no torch, no PLM extras).
RUN poetry config virtualenvs.create false \
    && poetry install --only main --no-root --no-interaction --no-ansi

# Phase 2: opt-in extras for the in-container embedder. The default
# build pulls torch + transformers + protea-backends so the image can
# run self-contained out of the box. Pass --build-arg INSTALL_EXTRAS=""
# to skip and produce a slim bind-mount-only image.
RUN if [ -n "${INSTALL_EXTRAS}" ]; then \
        poetry install --no-root --no-interaction --no-ansi \
            --extras "${INSTALL_EXTRAS}"; \
    fi

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
    PYTHONPATH=/app/src \
    HF_HOME=/app/.hf-cache \
    LAFA_EMBED_CACHE=/app/output/.embed_cache

# LAFA standard interface entrypoint. CMD is intentionally empty: the
# caller supplies all CLI args via ``docker run``.
ENTRYPOINT ["python3", "method_main.py"]
