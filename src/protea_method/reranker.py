"""LightGBM re-ranker inference.

Pure helpers extracted from PROTEA's ``protea/core/reranker.py``: the
feature-column schema (re-exported from ``protea_contracts``), the
``prepare_dataset`` data-prep, the inference ``predict`` /
``apply_reranker`` calls, model serialization round-trips, and the
``infer_active_feature_families`` mapping. No FastAPI, no SQLAlchemy,
no ArtifactStore: the caller hands over either booster bytes (via
``load_from_bytes``) or a ready ``lgb.Booster``.

Schema-sha validation is the caller's responsibility. The boundary is
"give me a booster + a DataFrame, I'll score it".
"""

from __future__ import annotations

import lightgbm as lgb
import numpy as np
import pandas as pd
from protea_contracts import (
    ALL_FEATURES,
    CATEGORICAL_FEATURES,
    EMBEDDING_PCA_DIM,
    LABEL_COLUMN,
    NUMERIC_FEATURES,
)


def fit_embedding_pca(
    embeddings: np.ndarray,
    n_components: int = EMBEDDING_PCA_DIM,
    *,
    max_fit_samples: int = 50_000,
    seed: int = 42,
) -> tuple[np.ndarray, np.ndarray]:
    """Fit PCA via truncated SVD on a (possibly subsampled) embedding matrix.

    Returns ``(mean, components)`` with ``mean`` shape ``(D,)`` and
    ``components`` shape ``(n_components, D)`` (both float32). Designed
    to be called once per ``EmbeddingConfig`` pool; subsequent
    projections are a single matmul.
    """
    if embeddings.size == 0:
        raise ValueError("embeddings matrix is empty")
    n = embeddings.shape[0]
    rng = np.random.default_rng(seed)
    if n > max_fit_samples:
        idx = rng.choice(n, size=max_fit_samples, replace=False)
        x = embeddings[idx].astype(np.float32, copy=False)
    else:
        x = embeddings.astype(np.float32, copy=False)
    mean = x.mean(axis=0)
    xc = x - mean
    _, _, vh = np.linalg.svd(xc, full_matrices=False)
    k = min(n_components, vh.shape[0])
    components = vh[:k].astype(np.float32)
    return mean.astype(np.float32), components


def prepare_dataset(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series]:
    """Extract feature matrix and label vector from a training DataFrame.

    Categorical columns are label-encoded to int64 codes (missing → -1)
    via :func:`pandas.factorize`. This mirrors
    ``protea-reranker-lab.reranker.encode_categoricals`` so a booster
    trained either inline here or in the lab can be scored by the same
    ``predict`` helper without LightGBM's "categorical_feature do not
    match" error firing on cross-instance imports.

    Returns ``(X, y)`` where X has only the feature columns and y is
    the binary label.
    """
    X = df[ALL_FEATURES].copy()
    for col in NUMERIC_FEATURES:
        if col in X.columns:
            X[col] = pd.to_numeric(X[col], errors="coerce")
    for col in CATEGORICAL_FEATURES:
        if col in X.columns:
            s = X[col].replace("", pd.NA)
            s = s.astype("object").where(s.notna(), None)
            codes, _ = pd.factorize(s, use_na_sentinel=True)
            X[col] = codes
    y = df[LABEL_COLUMN].astype(int)
    return X, y


def predict(
    model: lgb.Booster,
    df: pd.DataFrame,
    *,
    categorical_codes: dict[str, list[str]] | None = None,
) -> np.ndarray:
    """Score predictions using a trained re-ranker.

    Returns an array of scores in [0, 1] where higher = more likely
    correct. For lambdarank boosters, raw scores are unbounded reals;
    we apply a sigmoid to calibrate them into the [0, 1] range
    expected by the downstream CAFA evaluator (which sweeps thresholds
    from 0 to 1). Binary boosters already emit probabilities, so we
    leave them alone.

    ``categorical_codes`` is the per-column ordered string vocabulary
    the lab used at training time (``{column: [val0, val1, ...]}``).
    When provided, each cat column is encoded against this fixed
    vocabulary so the codes match training; when omitted, falls back
    to ``pd.factorize`` over the inference batch (correct only if the
    batch happens to contain the same set of values in the same
    order, usually wrong for small or aspect-filtered batches).
    """
    if LABEL_COLUMN in df.columns:
        X, _ = prepare_dataset(df)
    else:
        model_features = list(model.feature_name())
        aligned = df.copy()
        for col in model_features:
            if col not in aligned.columns:
                aligned[col] = pd.NA
        X = aligned[model_features].copy()
        for col in model_features:
            if col in NUMERIC_FEATURES:
                X[col] = pd.to_numeric(X[col], errors="coerce")
            elif col in CATEGORICAL_FEATURES:
                s = X[col].astype("object").where(X[col].notna(), None)
                if categorical_codes and col in categorical_codes:
                    mapping = {v: i for i, v in enumerate(categorical_codes[col])}
                    X[col] = s.map(lambda v, m=mapping: m.get(v, -1)).astype("int64")
                else:
                    codes, _ = pd.factorize(s, use_na_sentinel=True)
                    X[col] = codes

    raw = np.asarray(model.predict(X))
    if raw.size == 0:
        return raw
    if float(raw.min()) < 0.0 or float(raw.max()) > 1.0:
        return np.asarray(1.0 / (1.0 + np.exp(-raw)))
    return raw


def model_from_string(model_str: str) -> lgb.Booster:
    """Deserialize a LightGBM booster from its text representation."""
    return lgb.Booster(model_str=model_str)


def load_from_bytes(model_bytes: bytes) -> lgb.Booster:
    """Deserialize a LightGBM booster from raw bytes.

    Useful when the caller fetched the booster blob from an external
    source (artifact store, bind-mounted file, in-memory buffer) and
    just needs it materialised. The bytes must be the LightGBM text
    format that ``Booster.save_model`` writes.
    """
    return lgb.Booster(model_str=model_bytes.decode("utf-8"))


def apply_reranker(
    df: pd.DataFrame,
    booster: lgb.Booster,
    *,
    feature_cols: list[str] | None = None,
) -> np.ndarray:
    """Score ``df`` with ``booster`` and return an aligned array.

    If ``feature_cols`` is None we use the booster's own
    ``feature_name()``. Missing columns are filled with ``np.nan`` so
    LightGBM routes them through its native missing-value branch
    rather than crashing on KeyError.
    """
    cols = feature_cols or list(booster.feature_name())
    aligned = df.copy()
    for col in cols:
        if col not in aligned.columns:
            aligned[col] = np.nan
    X = aligned[cols].copy()
    for col in cols:
        if not isinstance(X[col].dtype, pd.CategoricalDtype):
            X[col] = pd.to_numeric(X[col], errors="coerce")
    raw = np.asarray(booster.predict(X))
    if raw.size == 0:
        return raw
    if float(raw.min()) < 0.0 or float(raw.max()) > 1.0:
        return np.asarray(1.0 / (1.0 + np.exp(-raw)))
    return raw


def infer_active_feature_families(
    *,
    compute_alignments: bool,
    compute_taxonomy: bool,
    compute_v6_features: bool,
    compute_lineage_features: bool = False,
) -> list[str]:
    """Map predict-time feature flags onto lab feature families.

    The PROTEA predict pipeline always materialises KNN features and
    annotation-meta columns (qualifier/evidence_code/aspect); the
    optional flags enable alignment, taxonomy-pair, taxonomy-voters,
    GO-context, anc2vec, emb-pca, length and lineage families. Keep
    this in sync with ``protea_reranker_lab.contracts.FEATURE_FAMILIES``.

    ``compute_lineage_features`` defaults to ``False`` so existing
    callers that never opted into the lineage feature continue to
    produce the same feature-schema fingerprint they shipped with
    (bit-exact reproducibility against the 52-feature lab champion).
    """
    families: list[str] = ["knn", "annotation_meta"]
    if compute_alignments:
        families.append("alignment_nw")
        families.append("length")
    if compute_taxonomy:
        families.append("taxonomy_pair")
    if compute_v6_features:
        families.extend(
            ["anc2vec_neighbor", "anc2vec_query", "emb_pca", "taxonomy_voters", "go_context"]
        )
    if compute_lineage_features:
        families.append("lineage")
    return sorted(set(families))


__all__ = [
    "ALL_FEATURES",
    "CATEGORICAL_FEATURES",
    "EMBEDDING_PCA_DIM",
    "LABEL_COLUMN",
    "NUMERIC_FEATURES",
    "apply_reranker",
    "fit_embedding_pca",
    "infer_active_feature_families",
    "load_from_bytes",
    "model_from_string",
    "predict",
    "prepare_dataset",
]
