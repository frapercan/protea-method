# protea-method

Pure inference path of PROTEA, packaged as a slim standalone
library. Contains the KNN search, feature computation, and
apply-reranker logic. No FastAPI, no SQLAlchemy, no protea-core.

This is the package shipped to consumers who want to use a
PROTEA-trained reranker without the platform.

## Audiences

| Tier | Footprint | Use case |
|------|-----------|----------|
| 1 | `pip install protea-method` | Caller already has embeddings; just predict |
| 2 | `pip install protea-method[esm]` | Caller has FASTA; embed + predict end-to-end |
| 3 | Docker `protea-method-runtime` | HPC airgap, single FASTA-in/TSV-out container |

## API sketch

```python
from protea_method import Predictor

predictor = Predictor.from_uri("s3://.../booster_v18.tar.gz")
predictions = predictor.predict(query_embeddings, references)
```

Or end-to-end with bundled backend:

```python
from protea_method import end_to_end_predict

predictions = end_to_end_predict("input.fasta", booster_uri="...")
```

The platform (`protea-core`) imports `protea-method.predict()`
from inside its `predict_go_terms` operation. Operations stay
the same; the algorithmic core lives here.

## Roadmap

This is the F0 bootstrap (T0.11 of the PROTEA master plan v3).
Real content lands in F2C.1 (extract inference path from
`protea-core/predict_go_terms` to this package).

## Versioning

SemVer 2.0.0; coordinated with `protea-contracts` (a major bump
on contracts forces a major bump here when the schema change is
breaking).

## Development

```bash
poetry install
poetry run pytest
poetry run ruff check .
poetry run mypy src tests
```

<!-- protea-stack:start -->

## Repositories in the PROTEA stack

Single source of truth: [`docs/source/_data/stack.yaml`](https://github.com/frapercan/PROTEA/blob/develop/docs/source/_data/stack.yaml) in PROTEA. Run `python scripts/sync_stack.py` to regenerate this block.

| Repo | Role | Status | Summary |
|------|------|--------|---------|
| [PROTEA](https://github.com/frapercan/PROTEA) | Platform | `active` | Backend platform. Hosts the ORM, job queue, FastAPI surface, frontend, and orchestration. |
| [protea-contracts](https://github.com/frapercan/protea-contracts) | Contracts | `beta` | Shared contract surface. ABCs, pydantic payloads, feature schema, schema_sha. Imported by every other repo. |
| **protea-method** (this repo) | Inference | `skeleton` | Pure inference path (KNN, feature compute, reranker apply). Target of the F2C extraction. Bind-mounted by the LAFA containers. |
| [protea-sources](https://github.com/frapercan/protea-sources) | Source plugin | `skeleton` | Annotation source plugins (GOA, QuickGO, UniProt). Discovered via Python entry_points. |
| [protea-runners](https://github.com/frapercan/protea-runners) | Runner plugin | `skeleton` | Experiment runner plugins (LightGBM lab, KNN baseline, future GNN). Discovered via Python entry_points. |
| [protea-backends](https://github.com/frapercan/protea-backends) | Backend plugin | `skeleton` | Protein language model embedding backends (ESM family, T5/ProstT5, Ankh, ESM3-C). Discovered via Python entry_points. |
| [protea-reranker-lab](https://github.com/frapercan/protea-reranker-lab) | Lab | `active` | LightGBM reranker training lab. Pulls datasets from PROTEA, trains boosters, publishes them back via /reranker-models/import-by-reference. |
| [cafaeval-protea](https://github.com/frapercan/cafaeval-protea) | Evaluator | `active` | Standalone fork of cafaeval (CAFA-evaluator-PK) with the PK-coverage fix and a bit-exact parity guarantee against the upstream. |

<!-- protea-stack:end -->
