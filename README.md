# protea-method

Pure inference path of PROTEA, packaged as a slim standalone library.
Contains KNN search, feature enrichment, GO ancestor embedding (Anc2Vec),
and reranker-apply logic. No FastAPI, no SQLAlchemy, no protea-core.

This package is the algorithmic core that PROTEA's `predict_go_terms`
operation delegates to at runtime. It is also the package shipped to
consumers who want to run predictions without the full platform stack
(no Postgres, no RabbitMQ, no workers required).

## Install

```bash
pip install protea-method
```

PROTEA ships CPU-only torch by default. For GPU inference, install the
CUDA wheel after the base install:

```bash
pip install protea-method
# then swap in the GPU wheel:
pip install torch --index-url https://download.pytorch.org/whl/cu121
```

## Quickstart

```python
import numpy as np
from protea_method import predict, PredictConfig

# Prepare inputs (float16 or float32 numpy arrays, shape [N, D])
query_embeddings = np.load("query_embeddings.npy")
ref_embeddings   = np.load("ref_embeddings.npy")

# GO annotations per reference sequence (list of lists of GO term strings)
ref_annotations = [["GO:0008150", "GO:0003674"], ...]

# GO parent map loaded from a parquet or JSON export
parent_map = {}  # {go_id: [parent_go_id, ...]}

# Run predictions
cfg = PredictConfig(k=15)
predictions = predict(query_embeddings, ref_embeddings, ref_annotations, parent_map, cfg)
# predictions: list of dicts {"go_id": str, "score": float, "aspect": str}
```

For scenarios where a PROTEA-trained LightGBM booster is available, the
predictions are automatically re-ranked:

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

## Running predictions on a FASTA file

`protea-method` does not include a FASTA parser or an embedding backend.
Those concerns live in `protea-sources` (for sequences) and
`protea-backends` (for PLM embeddings). The typical end-to-end flow is:

1. Compute embeddings for your FASTA via `protea-backends` (ESM, T5, Ankh, ESM-C).
2. Download reference embeddings + annotations from PROTEA via its REST API
   (`GET /embedding-configs/{id}/embeddings` + `GET /prediction-sets/{id}/annotations`).
3. Call `predict(...)` from this package.

Alternatively, start the full PROTEA stack and submit a `predict_go_terms`
job via `POST /jobs` — the platform runs the same `predict` function internally.

## Versioning

SemVer 2.0.0. The version is coordinated with `protea-contracts`: any
breaking change to the feature schema (field names, dtypes, ordering)
requires a major-version bump here and forces re-training of every
downstream LightGBM booster registered in PROTEA. PROTEA persists the
`feature_schema_sha` of each trained booster and refuses to score with a
booster whose schema digest drifts from the live inference pipeline.

## Test

```bash
poetry install
poetry run pytest                      # ~50 unit tests, < 1 s
poetry run pytest -v tests/test_pipeline.py  # end-to-end smoke
poetry run ruff check .
poetry run mypy --strict src
```

All tests are import-cheap. No GPU or network access is required.
Integration against a real PROTEA instance is tested at the
`protea-core` layer, not here.

## Contributing

Contributions are welcome from research institutions and individual
developers.

**Branch strategy:** all changes target `develop`; `main` tracks
stable releases only.

```bash
git clone https://github.com/frapercan/protea-method.git
cd protea-method
git checkout develop
git checkout -b feature/my-feature

poetry install

# Make your changes, then verify locally:
poetry run pytest
poetry run ruff check .
poetry run mypy --strict src

# Open a pull request targeting develop
```

Key constraints:
- **NEVER** use pgvector for KNN search. FAISS IVFFlat or numpy chunked
  brute-force are the only allowed KNN backends here.
- **No runtime deps** on `sqlalchemy`, `fastapi`, or `protea-core`. New
  deps must be `optional` or justified in the PR description.
- Public API is SemVer-ed — coordinate breaking changes with
  `protea-contracts` versioning.

## Documentation

Source-level docstrings are rendered by Sphinx autodoc in PROTEA's main
docs at **https://protea.readthedocs.io**. A local build is available
from the PROTEA repo: `cd docs && make html`.

## License

MIT. See `LICENSE`.

<!-- protea-stack:start -->

## Repositories in the PROTEA stack

Single source of truth: [`docs/source/_data/stack.yaml`](https://github.com/frapercan/PROTEA/blob/develop/docs/source/_data/stack.yaml) in PROTEA. Run `python scripts/sync_stack.py` to regenerate this block.

| Repo | Role | Status | Summary |
|------|------|--------|---------|
| [PROTEA](https://github.com/frapercan/PROTEA) | Platform | `active` | Backend platform. Hosts the ORM, job queue, FastAPI surface, frontend, and orchestration. |
| [protea-contracts](https://github.com/frapercan/protea-contracts) | Contracts | `beta` | Shared contract surface. ABCs, pydantic payloads, feature schema, schema_sha. Imported by every other repo. |
| **protea-method** (this repo) | Inference | `active` | Pure inference path (KNN, feature compute, reranker apply). Bind-mounted by the LAFA containers. |
| [protea-sources](https://github.com/frapercan/protea-sources) | Source plugin | `skeleton` | Annotation source plugins (GOA, QuickGO, UniProt). Discovered via Python entry_points. |
| [protea-runners](https://github.com/frapercan/protea-runners) | Runner plugin | `skeleton` | Experiment runner plugins (LightGBM lab, KNN baseline, future GNN). Discovered via Python entry_points. |
| [protea-backends](https://github.com/frapercan/protea-backends) | Backend plugin | `skeleton` | Protein language model embedding backends (ESM family, T5/ProstT5, Ankh, ESM3-C). Discovered via Python entry_points. |
| [protea-reranker-lab](https://github.com/frapercan/protea-reranker-lab) | Lab | `active` | LightGBM reranker training lab. Pulls datasets from PROTEA, trains boosters, publishes them back via /reranker-models/import-by-reference. |
| [cafaeval-protea](https://github.com/frapercan/cafaeval-protea) | Evaluator | `active` | Standalone fork of cafaeval (CAFA-evaluator-PK) with the PK-coverage fix and a bit-exact parity guarantee against the upstream. |

<!-- protea-stack:end -->
