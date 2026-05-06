"""Coverage tests for ``protea_method.pca_cache``."""

from __future__ import annotations

import uuid
from pathlib import Path

import numpy as np

from protea_method.pca_cache import (
    _load_pca_state,
    _pca_state_path,
    _save_pca_state,
    load_or_fit_pca_state,
)


def test_save_and_load_roundtrip(tmp_path: Path) -> None:
    config_id = uuid.uuid4()
    mean = np.arange(8, dtype=np.float32)
    components = np.eye(4, 8, dtype=np.float32)
    _save_pca_state(config_id, mean, components, artifacts_dir=tmp_path)
    loaded = _load_pca_state(config_id, artifacts_dir=tmp_path)
    assert loaded is not None
    np.testing.assert_array_equal(loaded[0], mean)
    np.testing.assert_array_equal(loaded[1], components)


def test_load_returns_none_for_missing_artifact(tmp_path: Path) -> None:
    assert _load_pca_state(uuid.uuid4(), artifacts_dir=tmp_path) is None


def test_load_returns_none_for_corrupt_artifact(tmp_path: Path) -> None:
    config_id = uuid.uuid4()
    _pca_state_path(config_id, artifacts_dir=tmp_path).parent.mkdir(parents=True, exist_ok=True)
    _pca_state_path(config_id, artifacts_dir=tmp_path).write_bytes(b"not a numpy archive")
    assert _load_pca_state(config_id, artifacts_dir=tmp_path) is None


def test_load_or_fit_returns_cached_state(tmp_path: Path) -> None:
    config_id = uuid.uuid4()
    rng = np.random.default_rng(0)
    embeddings = rng.standard_normal(size=(200, 32)).astype(np.float32)
    first = load_or_fit_pca_state(config_id, embeddings, artifacts_dir=tmp_path)
    assert first is not None
    second = load_or_fit_pca_state(
        config_id,
        np.zeros((0, 0), dtype=np.float32),
        artifacts_dir=tmp_path,
    )
    assert second is not None
    np.testing.assert_array_equal(first[0], second[0])
    np.testing.assert_array_equal(first[1], second[1])


def test_load_or_fit_returns_none_for_empty_pool(tmp_path: Path) -> None:
    config_id = uuid.uuid4()
    result = load_or_fit_pca_state(
        config_id,
        np.zeros((0, 32), dtype=np.float32),
        artifacts_dir=tmp_path,
    )
    assert result is None


def test_load_or_fit_persists_artifact(tmp_path: Path) -> None:
    config_id = uuid.uuid4()
    rng = np.random.default_rng(1)
    embeddings = rng.standard_normal(size=(100, 16)).astype(np.float32)
    load_or_fit_pca_state(config_id, embeddings, artifacts_dir=tmp_path)
    assert _pca_state_path(config_id, artifacts_dir=tmp_path).exists()
