# protea-method

LAFA submission image for **PROTEA** (PROtein functional Embedding-based
Annotation). The image performs protein function annotation via a Protein
Language Model (PLM) embedding step, KNN retrieval against an
experimentally-annotated reference set, per-aspect LightGBM rerankers, and
GO ancestor propagation. It implements the standard
[LAFA container interface](https://github.com/anphan0828/LAFA_container_guide)
so it can be invoked unmodified by FunctionBench
([functionbench.net](https://functionbench.net/)).

## Quick start

### Self-contained mode (recommended)

The image computes embeddings in-process from the FASTA inputs. Cold start
on the first run downloads the configured PLM weights into the mounted
HuggingFace cache (the default `esm2_t36_3B` pulls about 12 GB).

```bash
mkdir -p ./data ./output ./hf-cache

# Place test_sequences.fasta, train_sequences.fasta, goa_uniprot_sprot.gaf.gz,
# go-basic.obo under ./data (see the LAFA container guide for sources).

docker run --rm \
    -v "$(pwd)/data":/app/data:ro \
    -v "$(pwd)/output":/app/output:rw \
    -v "$(pwd)/hf-cache":/app/.hf-cache:rw \
    frapercan/protea-method:v0.1.0 \
        --query_file       /app/data/test_sequences.fasta \
        --train_sequences  /app/data/train_sequences.fasta \
        --annot_file       /app/data/goa_uniprot_sprot.gaf.gz \
        --graph            /app/data/go-basic.obo \
        --output_file      /app/output/predictions.tsv.gz \
        --backend_id       esm2_t36_3B \
        --embed_device     auto \
        --top_k 5
```

### Bind-mount mode (slim, CI-friendly)

If embeddings were materialised out-of-band, supply them as parquet files
and the image skips the PLM forward pass entirely (no torch, no weights
downloaded).

```bash
docker run --rm \
    -v "$(pwd)/data":/app/data:ro \
    -v "$(pwd)/output":/app/output:rw \
    frapercan/protea-method:v0.1.0 \
        --query_file       /app/data/test_sequences.fasta \
        --train_sequences  /app/data/train_sequences.fasta \
        --annot_file       /app/data/goa_uniprot_sprot.gaf.gz \
        --graph            /app/data/go-basic.obo \
        --output_file      /app/output/predictions.tsv.gz \
        --query_embeds     /app/data/query.parquet \
        --reference_embeds /app/data/reference.parquet \
        --top_k 5
```

The two parquet files must carry an `accession` (or `protein_accession`)
column plus either an `embedding` list-typed column or `e0..eN` numeric
columns.

## Mount-point contract

| Host path  | Container path     | Mode | Purpose                                 |
|------------|--------------------|------|-----------------------------------------|
| `./data`   | `/app/data`        | ro   | FASTA / GAF / OBO inputs                |
| `./output` | `/app/output`      | rw   | Predictions TSV and embed cache         |
| `./hf-cache` | `/app/.hf-cache` | rw   | HuggingFace weights cache (optional)    |

## CLI flags

Required (LAFA standard interface):

| Flag                | Description                                                 |
|---------------------|-------------------------------------------------------------|
| `--query_file`      | FASTA of query (test) sequences.                            |
| `--train_sequences` | FASTA of reference (experimentally-annotated) sequences.    |
| `--annot_file`      | GAF or GAF.GZ annotations for the reference set.            |
| `--graph`           | GO ontology in OBO format (e.g. `go-basic.obo`).            |
| `--output_file`     | Destination for the predictions TSV. Append `.gz` to gzip.  |

Embedding selection (mutually exclusive with bind-mount):

| Flag                  | Default          | Description                                                 |
|-----------------------|------------------|-------------------------------------------------------------|
| `--query_embeds`      | (unset)          | Parquet of pre-computed query embeddings.                   |
| `--reference_embeds`  | (unset)          | Parquet of pre-computed reference embeddings.               |
| `--backend_id`        | `esm2_t36_3B`    | Backend id when embeddings are computed in-container.       |
| `--embed_cache_dir`   | `$LAFA_EMBED_CACHE` | Cache for `(backend_id, fasta_sha256)` keyed embeddings. |
| `--embed_device`      | `auto`           | Torch device (`auto` / `cpu` / `cuda` / `cuda:0`).          |
| `--embed_batch_size`  | `8`              | Embedding plugin batch size.                                |

KNN behaviour:

| Flag             | Default  | Description                                                       |
|------------------|----------|-------------------------------------------------------------------|
| `--top_k`        | `5`      | KNN k value. Matches the PROTEA `bench-v1-K5-v226-lineage` setup. |
| `--metric`       | `cosine` | KNN distance metric. `cosine` or `l2`.                            |
| `--backend`      | `numpy`  | KNN backend. `numpy` (chunked brute force) or `faiss` (IVFFlat).  |
| `--num_threads`  | (unset)  | Informational hint. Set `OMP_NUM_THREADS` for the real knob.      |

Supplying `--query_embeds` and `--reference_embeds` together selects
bind-mount mode; omitting both selects self-contained mode. Supplying
only one is rejected at start-up.

## Environment variables

| Variable             | Default                       | Purpose                                            |
|----------------------|-------------------------------|----------------------------------------------------|
| `HF_HOME`            | `/app/.hf-cache`              | HuggingFace cache root (bind-mount to persist).    |
| `LAFA_EMBED_CACHE`   | `/app/output/.embed_cache`    | Filesystem cache for computed embeddings.          |
| `OMP_NUM_THREADS`    | (unset)                       | Read by numpy and faiss-cpu.                       |
| `PYTHONUNBUFFERED`   | `1`                           | Line-buffered stderr progress log.                 |

## Output format

Three-column TSV with no header, optionally gzipped. Columns:

```
Query_ID    GO_Term     Score
```

All three aspects (Molecular Function, Biological Process, Cellular
Component) are interleaved. Scores are floats in [0, 1]. The LAFA grader
sorts and thresholds downstream.

## Image tags

| Tag             | Contents                                                      |
|-----------------|---------------------------------------------------------------|
| `v0.1.0`        | First LAFA-ready cut. Self-contained build (CPU torch + ESM). |
| `v0.1.0-cpu`    | Alias for `v0.1.0` (no GPU runtime included by design).       |
| `v0.1.0-slim`   | Bind-mount-only. Built with `--build-arg INSTALL_EXTRAS=""`.  |
| `latest`        | Floating pointer to the latest tagged release.                |

Tags follow SemVer 2.0.0; the version is coordinated with
`protea-contracts` (any breaking change to the feature schema requires a
major-version bump).

## Hardware

* CPU works end-to-end. The reference numbers in the method card were
  produced on CPU.
* GPU (CUDA) is supported via `--embed_device cuda` once a CUDA-enabled
  torch wheel is available in the runtime; the default tag bundles the
  CPU wheel. For GPU runs build a custom image installing the CUDA wheel
  on top of `v0.1.0-slim` plus the embedding parquet path.
* First run in self-contained mode downloads about 12 GB of ESM-2 3B
  weights into the mounted HuggingFace cache. Image size on disk is
  about 6 GB; total footprint after the cold start is around 18 GB.

## Author and supervision

Author and sole maintainer of PROTEA: Francisco Miguel Pérez Canales
(frapercan1@gmail.com).

Co-supervisors: David Orellana-Martín, Ana M. Rojas.

PROTEA is the post-CAFA productisation of the CAFA 6 (team result, rank
2) submission stack; the author was the technical motor of the team
effort. The image packages the same algorithmic core used in production
PROTEA deployments.

## License and source

MIT License. Source code, issue tracker, and tagged releases are at
[github.com/frapercan/protea-method](https://github.com/frapercan/protea-method).

For the full PROTEA stack (platform, contracts, plugins, reranker lab,
evaluator) see [github.com/frapercan/PROTEA](https://github.com/frapercan/PROTEA).
