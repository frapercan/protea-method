#!/usr/bin/env bash
# Copy-pasteable example invocation of the protea-method LAFA image.
#
# Mount layout follows the LAFA container guide convention:
#   ./data    -> /app/data   (read-only inputs)
#   ./output  -> /app/output (read-write predictions)
#
# Inputs expected in ./data (download from HuggingFace anphan0828/lafa
# for the time-point release of interest, e.g. Sep_2025/):
#   test_sequences.fasta            queries (all SwissProt for the release)
#   train_sequences.fasta           experimentally-annotated training set
#   goa_uniprot_sprot.gaf.gz        full GOA annotations
#   go-basic.obo                    GO ontology graph
#   query.parquet, reference.parquet  pre-computed PLM embeddings
#     (until LAFA-EMB.1 wires an in-container embedder)
#
# The two parquet files must have an ``accession`` column plus either
# an ``embedding`` list-typed column or ``e0..eN`` numeric columns.

set -euo pipefail

IMAGE="${IMAGE:-protea-method-lafa:latest}"
DATA_DIR="${DATA_DIR:-$(pwd)/data}"
OUTPUT_DIR="${OUTPUT_DIR:-$(pwd)/output}"

mkdir -p "${OUTPUT_DIR}"

docker run --rm \
    -v "${DATA_DIR}":/app/data:ro \
    -v "${OUTPUT_DIR}":/app/output:rw \
    "${IMAGE}" \
        --query_file       /app/data/test_sequences.fasta \
        --train_sequences  /app/data/train_sequences.fasta \
        --annot_file       /app/data/goa_uniprot_sprot.gaf.gz \
        --graph            /app/data/go-basic.obo \
        --output_file      /app/output/predictions.tsv.gz \
        --query_embeds     /app/data/query.parquet \
        --reference_embeds /app/data/reference.parquet \
        --top_k 5 \
        --metric cosine \
        --backend numpy
