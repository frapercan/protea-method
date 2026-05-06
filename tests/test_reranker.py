"""Smoke tests for ``protea_method.reranker`` (F2C.1 extraction).

Mirrors the relevant subset of PROTEA's ``tests/test_reranker.py`` so
the extraction is a drop-in replacement at the call site. The full
parity sweep against PROTEA fixtures runs in F2C.6 (cross-repo
integration).
"""

from __future__ import annotations

import lightgbm as lgb
import numpy as np
import pandas as pd

from protea_method import (
    ALL_FEATURES,
    CATEGORICAL_FEATURES,
    LABEL_COLUMN,
    NUMERIC_FEATURES,
    apply_reranker,
    fit_embedding_pca,
    infer_active_feature_families,
    load_from_bytes,
    model_from_string,
    prepare_dataset,
)

# `predict` is the booster-scoring helper inside ``reranker``; the
# package-root ``predict`` is the higher-level orchestrator in
# ``pipeline`` (different signature). Import the helper via the
# submodule to avoid the namespace collision.
from protea_method.reranker import predict


def _make_df(n: int = 100, seed: int = 42) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    labels = (rng.random(n) < 0.3).astype(int)
    data: dict[str, list] = {
        "protein_accession": [f"P{i:05d}" for i in range(n)],
        "go_id": [f"GO:{rng.integers(1, 99999):07d}" for _ in range(n)],
        "aspect": rng.choice(["F", "P", "C"], n).tolist(),
        LABEL_COLUMN: labels.tolist(),
    }
    for col in NUMERIC_FEATURES:
        data[col] = rng.random(n).tolist()
    for col in CATEGORICAL_FEATURES:
        data[col] = rng.choice(["a", "b", "c"], n).tolist()
    return pd.DataFrame(data)


def _train(df: pd.DataFrame) -> lgb.Booster:
    X, y = prepare_dataset(df)
    train = lgb.Dataset(X, label=y)
    return lgb.train(
        {"objective": "binary", "verbose": -1, "num_leaves": 7, "learning_rate": 0.1},
        train,
        num_boost_round=10,
    )


def test_prepare_dataset_shapes() -> None:
    df = _make_df()
    X, y = prepare_dataset(df)
    assert list(X.columns) == ALL_FEATURES
    assert len(y) == len(df)
    for col in CATEGORICAL_FEATURES:
        assert X[col].dtype.kind in ("i", "u")  # label-encoded


def test_predict_returns_probabilities_in_unit_interval() -> None:
    df = _make_df()
    booster = _train(df)
    scores = predict(booster, df)
    assert scores.shape == (len(df),)
    assert float(scores.min()) >= 0.0
    assert float(scores.max()) <= 1.0


def test_predict_handles_missing_label_column() -> None:
    df = _make_df()
    booster = _train(df)
    inference_df = df.drop(columns=[LABEL_COLUMN])
    scores = predict(booster, inference_df)
    assert scores.shape == (len(df),)


def test_apply_reranker_aligns_missing_columns() -> None:
    df = _make_df()
    booster = _train(df)
    sparse = df.drop(columns=NUMERIC_FEATURES[:2])  # drop two features
    scores = apply_reranker(sparse, booster)
    assert scores.shape == (len(df),)
    assert float(scores.min()) >= 0.0
    assert float(scores.max()) <= 1.0


def test_model_from_string_roundtrip() -> None:
    df = _make_df()
    booster = _train(df)
    text = booster.model_to_string()
    restored = model_from_string(text)
    np.testing.assert_array_equal(
        np.asarray(booster.predict(prepare_dataset(df)[0])),
        np.asarray(restored.predict(prepare_dataset(df)[0])),
    )


def test_load_from_bytes_roundtrip() -> None:
    df = _make_df()
    booster = _train(df)
    blob = booster.model_to_string().encode("utf-8")
    restored = load_from_bytes(blob)
    np.testing.assert_array_equal(
        np.asarray(booster.predict(prepare_dataset(df)[0])),
        np.asarray(restored.predict(prepare_dataset(df)[0])),
    )


def test_fit_embedding_pca_shapes() -> None:
    rng = np.random.default_rng(0)
    embeddings = rng.standard_normal(size=(500, 128)).astype(np.float32)
    mean, components = fit_embedding_pca(embeddings, n_components=16)
    assert mean.shape == (128,)
    assert components.shape == (16, 128)
    assert mean.dtype == np.float32
    assert components.dtype == np.float32


def test_fit_embedding_pca_rejects_empty_matrix() -> None:
    import pytest

    with pytest.raises(ValueError):
        fit_embedding_pca(np.zeros((0, 128), dtype=np.float32))


def test_infer_active_feature_families_baseline() -> None:
    families = infer_active_feature_families(
        compute_alignments=False,
        compute_taxonomy=False,
        compute_v6_features=False,
    )
    assert families == ["annotation_meta", "knn"]


def test_infer_active_feature_families_full() -> None:
    families = infer_active_feature_families(
        compute_alignments=True,
        compute_taxonomy=True,
        compute_v6_features=True,
    )
    assert "alignment_nw" in families
    assert "taxonomy_pair" in families
    assert "anc2vec_neighbor" in families
    assert "emb_pca" in families


def test_predict_with_categorical_codes_vocabulary() -> None:
    """Encode categoricals against a fixed lab vocabulary."""
    df = _make_df()
    booster = _train(df)
    inference_df = df.drop(columns=[LABEL_COLUMN])
    codes: dict[str, list[str]] = {col: ["a", "b", "c"] for col in CATEGORICAL_FEATURES}
    scores = predict(booster, inference_df, categorical_codes=codes)
    assert scores.shape == (len(df),)
    assert float(scores.min()) >= 0.0
    assert float(scores.max()) <= 1.0


def test_fit_embedding_pca_subsamples_above_max() -> None:
    """When n > max_fit_samples the function still returns the right shape."""
    rng = np.random.default_rng(1)
    embeddings = rng.standard_normal(size=(2_000, 64)).astype(np.float32)
    mean, components = fit_embedding_pca(embeddings, n_components=8, max_fit_samples=500)
    assert mean.shape == (64,)
    assert components.shape == (8, 64)

