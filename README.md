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
