#!/usr/bin/env bash
# Self-contained LAFA invocation (LAFA-EMB.1): the container reads the
# query and training FASTAs, computes PLM embeddings in-process via the
# selected backend, caches them, and runs predict.
#
# Mount layout:
#   ./data       -> /app/data       (read-only inputs)
#   ./output     -> /app/output     (read-write predictions + embed cache)
#   ./hf-cache   -> /app/.hf-cache  (read-write HuggingFace model cache)
#
# Inputs expected in ./data (download from HuggingFace anphan0828/lafa
# for the time-point release of interest):
#   test_sequences.fasta            queries
#   train_sequences.fasta           experimentally-annotated training set
#   goa_uniprot_sprot.gaf.gz        full GOA annotations
#   go-basic.obo                    GO ontology graph
#
# Backends accepted (--backend_id):
#   esm2_t36_3B           (default, champion config; ~12 GB weights)
#   esm2_t33_650M         (lighter; ~2.5 GB, 5x faster on CPU)
#   prost_t5_xl_uniref50  (T5 cross-check; ~5.5 GB)
#
# Cold start downloads weights into ./hf-cache the first run; subsequent
# runs with the same backend pay zero download cost.

set -euo pipefail

IMAGE="${IMAGE:-protea-method-lafa:latest}"
DATA_DIR="${DATA_DIR:-$(pwd)/data}"
OUTPUT_DIR="${OUTPUT_DIR:-$(pwd)/output}"
HF_CACHE_DIR="${HF_CACHE_DIR:-$(pwd)/hf-cache}"
BACKEND_ID="${BACKEND_ID:-esm2_t36_3B}"
EMBED_DEVICE="${EMBED_DEVICE:-auto}"
EMBED_BATCH_SIZE="${EMBED_BATCH_SIZE:-8}"

mkdir -p "${OUTPUT_DIR}" "${HF_CACHE_DIR}"

docker run --rm \
    -v "${DATA_DIR}":/app/data:ro \
    -v "${OUTPUT_DIR}":/app/output:rw \
    -v "${HF_CACHE_DIR}":/app/.hf-cache:rw \
    "${IMAGE}" \
        --query_file        /app/data/test_sequences.fasta \
        --train_sequences   /app/data/train_sequences.fasta \
        --annot_file        /app/data/goa_uniprot_sprot.gaf.gz \
        --graph             /app/data/go-basic.obo \
        --output_file       /app/output/predictions.tsv.gz \
        --backend_id        "${BACKEND_ID}" \
        --embed_device      "${EMBED_DEVICE}" \
        --embed_batch_size  "${EMBED_BATCH_SIZE}" \
        --top_k 5 \
        --metric cosine \
        --backend numpy
