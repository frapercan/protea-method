# protea-method

**LAFA submission layer that wraps the PROTEA pipeline for FunctionBench.**

`protea-method` is the pure inference path of the PROTEA protein-annotation
stack, packaged as a slim standalone library. It implements the standard LAFA
container interface so that the same code that powers the PROTEA platform
worker can be shipped as a self-contained Docker image for
[FunctionBench](https://functionbench.net/) submissions.

The library contains KNN search, GO ancestor embedding (Anc2Vec), feature
enrichment (alignment, taxonomy, Anc2Vec, lineage), and LightGBM re-ranker-apply logic. It has no runtime dependency
on FastAPI, SQLAlchemy, or protea-core. No Postgres, no RabbitMQ, no workers
required for standalone use.

**Status:** v0.3.0 (beta; SemVer coordinated with `protea-contracts`; breaking
changes to the feature schema require a major bump).

<!-- protea-stack:start -->

## Repositories in the PROTEA stack

Single source of truth: [`docs/source/_data/stack.yaml`](https://github.com/frapercan/PROTEA/blob/develop/docs/source/_data/stack.yaml) in PROTEA. Run `python scripts/sync_stack.py` to regenerate this block.

| Repo | Role | Status | Summary |
|------|------|--------|---------|
| [PROTEA](https://github.com/frapercan/PROTEA) | Platform | `active` | Backend platform. Hosts the ORM, job queue, FastAPI surface, frontend, and orchestration. |
| [protea-contracts](https://github.com/frapercan/protea-contracts) | Contracts | `beta` | Shared contract surface. ABCs, pydantic payloads, feature schema, schema_sha. Imported by every other repo. |
| **protea-method** (this repo) | Inference | `active` | LAFA submission layer. Pure inference path (KNN, feature compute, reranker apply). Wrapped by the LAFA container for FunctionBench submissions. |
| [protea-sources](https://github.com/frapercan/protea-sources) | Source plugin | `skeleton` | Annotation source plugins (GOA, QuickGO, UniProt). Discovered via Python entry_points. |
| [protea-runners](https://github.com/frapercan/protea-runners) | Runner plugin | `skeleton` | Experiment runner plugins (LightGBM lab, KNN baseline, future GNN). Discovered via Python entry_points. |
| [protea-backends](https://github.com/frapercan/protea-backends) | Backend plugin | `skeleton` | Protein language model embedding backends (ESM family, T5/ProstT5, Ankh, ESM3-C). Discovered via Python entry_points. |
| [protea-reranker-lab](https://github.com/frapercan/protea-reranker-lab) | Lab | `active` | LightGBM reranker training lab. Pulls datasets from PROTEA, trains boosters, publishes them back via /reranker-models/import-by-reference. |
| [cafaeval-protea](https://github.com/frapercan/cafaeval-protea) | Evaluator | `active` | Standalone fork of cafaeval (CAFA-evaluator-PK) with the PK-coverage fix and a bit-exact parity guarantee against the upstream. |

<!-- protea-stack:end -->

## Install

```bash
pip install protea-method
```

The default install is torch-free. KNN search, feature enrichment, Anc2Vec,
lineage, and the LightGBM re-ranker run on numpy + faiss-cpu + lightgbm only.

The optional gated-attention MIL head (`protea_method.mil`) requires torch and
is opt-in via the `mil` extra:

```bash
pip install "protea-method[mil]"
```

For in-container PLM embedding (self-contained LAFA mode), install the matching
backend extras:

```bash
pip install "protea-method[esm]"   # esm2_t36_3B + esm2_t33_650M
pip install "protea-method[t5]"    # prost_t5_xl_uniref50
```

For GPU inference, install the extras first (they bring in the CPU torch wheel)
and then swap in the CUDA wheel:

```bash
pip install "protea-method[esm]"
pip install torch --index-url https://download.pytorch.org/whl/cu121
```

## Quickstart

```python
import numpy as np
from protea_method import predict, PredictConfig

# Float16 or float32 numpy arrays, shape [N, D]
query_embeddings = np.load("query_embeddings.npy")
ref_embeddings   = np.load("ref_embeddings.npy")

# GO annotations per reference sequence (list of lists of GO term strings)
ref_annotations = [["GO:0008150", "GO:0003674"], ...]

# GO parent map loaded from a parquet or JSON export
parent_map = {}  # {go_id: [parent_go_id, ...]}

cfg = PredictConfig(k=15)
predictions = predict(query_embeddings, ref_embeddings, ref_annotations, parent_map, cfg)
# predictions: list of dicts {"go_id": str, "score": float, "aspect": str}
```

When a PROTEA-trained LightGBM booster is available, predictions are
automatically re-ranked:

```python
from protea_method.reranker import load_from_bytes

with open("booster.txt", "rb") as fh:
    booster = load_from_bytes(fh.read())

cfg = PredictConfig(k=15, booster=booster)
predictions = predict(query_embeddings, ref_embeddings, ref_annotations, parent_map, cfg)
```

## Modules at a glance

| Module | Responsibility |
|--------|----------------|
| `pipeline.py` | `predict` orchestrator: wires KNN, feature enricher, reranker |
| `knn_search.py` | FAISS IVFFlat / numpy chunked KNN search (hard constraint: no pgvector) |
| `feature_enricher.py` | Adds alignment, taxonomy, Anc2Vec, and lineage features to KNN candidates |
| `reranker.py` | LightGBM booster load, dataset preparation, apply + score |
| `anc2vec.py` | Ancestor-embedding index for GO DAG proximity features |
| `lineage.py` | Taxonomic lineage distance features |
| `pca_cache.py` | Lazy PCA fitting / loading for embedding compression |
| `io/loaders.py` | FASTA / GAF / OBO readers for the container entrypoint |
| `io/lafa_tsv.py` | 3-column LAFA TSV writer (Query_ID, GO_Term, Score) |
| `embed/backend.py` | Backend resolver and `embed_fasta` orchestrator (self-contained mode) |
| `embed/cache.py` | Disk-backed embedding cache keyed by `(backend_id, fasta_sha256)` |
| `method_main.py` | LAFA container entrypoint; the script the Docker image runs |

## Running as a LAFA container

Build the image:

```bash
docker build -t protea-method-lafa:latest .
```

### Bind-mount mode (pre-computed embeddings)

Pass `--query_embeds` / `--reference_embeds` pointing at parquet files with
pre-computed embeddings. The slim image (no torch, no HuggingFace cache) is
sufficient. Mount your data and output directories:

```bash
bash docker/example_run.sh
```

The mount-point contract follows the LAFA container guide:
`./data:/app/data:ro` for inputs and `./output:/app/output:rw` for predictions.

Programmatic alternative (no Docker):

1. Compute embeddings for your FASTA via `protea-backends` (ESM, T5, Ankh, ESM-C).
2. Download reference embeddings + annotations from PROTEA via its REST API.
3. Call `predict(...)` from this package directly.

### Self-contained mode (in-container PLM embedder)

Omit `--query_embeds` and `--reference_embeds`. The container computes
embeddings in-process via a `protea-backends` plugin (default
`esm2_t36_3B`). Computed embeddings are cached under `$LAFA_EMBED_CACHE`
(`/app/output/.embed_cache` inside the image) keyed by `(backend_id,
fasta_sha256)`, so re-runs on the same FASTA skip the multi-hour PLM forward
pass.

```bash
bash docker/example_run_selfcontained.sh
```

Mount a host directory to `/app/.hf-cache` to avoid re-downloading the
~12 GB ESM-2 3B weights on every fresh container:

```bash
docker run -v hf-cache:/app/.hf-cache ...
```

Accepted `--backend_id` values:

| ID | Model | Size |
|----|-------|------|
| `esm2_t36_3B` (default) | `facebook/esm2_t36_3B_UR50D` | ~12 GB |
| `esm2_t33_650M` | `facebook/esm2_t33_650M_UR50D` | ~2.5 GB |
| `prost_t5_xl_uniref50` | `Rostlab/ProstT5` | cross-check |
| `mock_constant` | deterministic constant vector | tests only |

## Releasing to DockerHub / Submitting to FunctionBench

Three documents in `docker/` cover the full submission workflow. They are the
primary operator reference for publishing a new version of the LAFA container:

| Document | Purpose |
|----------|---------|
| [`docker/RELEASE_RUNBOOK.md`](docker/RELEASE_RUNBOOK.md) | Manual numbered checklist: build, smoke test, push to DockerHub, submit to FunctionBench, tag a GitHub release. Start here. |
| [`docker/DOCKERHUB_README.md`](docker/DOCKERHUB_README.md) | Long-form repository description to paste into the DockerHub "Full description" field when publishing a new image. |
| [`docker/FUNCTIONBENCH_METHODCARD.md`](docker/FUNCTIONBENCH_METHODCARD.md) | One-page method card to paste into the FunctionBench submission form, including validation numbers. |

The DockerHub push is intentionally manual (the runbook is operator-driven);
no CI job performs it.

## Documentation

Full API reference is built with Sphinx autodoc. To build locally:

```bash
pip install "protea-method[mil]" sphinx alabaster
cd docs
make html
# Open docs/build/html/index.html
```

The docs cover `method_main`, all loaders (`io/loaders.py`, `io/lafa_tsv.py`),
the in-container PLM embedder (`embed/`), and every public module in
`protea_method`.

## Versioning

SemVer 2.0.0, coordinated with `protea-contracts`. Any breaking change to the
feature schema (field names, dtypes, ordering) requires a major-version bump
here and forces re-training of every downstream LightGBM booster registered in
PROTEA. PROTEA persists the `feature_schema_sha` of each trained booster and
refuses to score with a booster whose schema digest drifts from the live
inference pipeline.

## Test

```bash
poetry install
poetry run pytest                            # ~50 unit tests, < 1 s
poetry run pytest -v tests/test_pipeline.py  # end-to-end smoke
poetry run ruff check .
poetry run mypy --strict src
```

All tests are import-cheap. No GPU or network access is required.

## Contributing

**Branch strategy:** all changes target `develop`; `main` tracks stable
releases only.

```bash
git clone https://github.com/frapercan/protea-method.git
cd protea-method
git checkout develop
git checkout -b feature/my-feature

poetry install

# Verify locally before opening a PR:
poetry run pytest
poetry run ruff check .
poetry run mypy --strict src

# Open a pull request targeting develop
```

Key constraints:

- **NEVER** use pgvector for KNN search. FAISS IVFFlat or numpy chunked
  brute-force are the only allowed KNN backends.
- **No runtime deps** on `sqlalchemy`, `fastapi`, or `protea-core`. New deps
  must be optional or justified in the PR description.
- Public API is SemVer-ed; coordinate breaking changes with `protea-contracts`
  versioning.

## License

MIT. See `LICENSE`.
