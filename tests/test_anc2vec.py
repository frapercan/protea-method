"""Coverage tests for ``protea_method.anc2vec``."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from protea_method.anc2vec import Anc2VecIndex, get_index


def _write_artifact(path: Path, go_ids: list[str], embeddings: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(
        path,
        embeddings=embeddings.astype(np.float32),
        go_ids=np.array(go_ids, dtype=object),
        ontology_release="test-2020-10",
    )


def test_index_basic_lookup(tmp_path: Path) -> None:
    artifact = tmp_path / "anc2vec.npz"
    go_ids = ["GO:0000001", "GO:0000002", "GO:0000003"]
    embeddings = np.arange(15, dtype=np.float32).reshape(3, 5)
    _write_artifact(artifact, go_ids, embeddings)
    idx = Anc2VecIndex(artifact)
    assert len(idx) == 3
    assert "GO:0000001" in idx
    assert "GO:9999999" not in idx
    assert idx.dim == 5
    assert idx.release == "test-2020-10"
    np.testing.assert_array_equal(idx.vec("GO:0000002"), embeddings[1])
    assert idx.vec("GO:9999999") is None


def test_batch_zero_fill(tmp_path: Path) -> None:
    artifact = tmp_path / "anc2vec.npz"
    go_ids = ["GO:0000001"]
    embeddings = np.array([[1.0, 2.0, 3.0]], dtype=np.float32)
    _write_artifact(artifact, go_ids, embeddings)
    idx = Anc2VecIndex(artifact)
    batched = idx.batch(["GO:0000001", "GO:9999999"])
    np.testing.assert_array_equal(batched[0], embeddings[0])
    np.testing.assert_array_equal(batched[1], np.zeros(3, dtype=np.float32))


def test_batch_nan_fill(tmp_path: Path) -> None:
    artifact = tmp_path / "anc2vec.npz"
    _write_artifact(artifact, ["GO:0000001"], np.array([[1.0, 2.0]], dtype=np.float32))
    idx = Anc2VecIndex(artifact)
    batched = idx.batch(["GO:9999999"], zero_if_missing=False)
    assert batched.shape == (1, 2)
    assert np.all(np.isnan(batched[0]))


def test_artifact_without_ontology_release(tmp_path: Path) -> None:
    artifact = tmp_path / "anc2vec.npz"
    artifact.parent.mkdir(parents=True, exist_ok=True)
    np.savez(
        artifact,
        embeddings=np.zeros((1, 4), dtype=np.float32),
        go_ids=np.array(["GO:0000001"], dtype=object),
    )
    idx = Anc2VecIndex(artifact)
    assert idx.release == ""


def test_get_index_returns_singleton(tmp_path: Path) -> None:
    artifact = tmp_path / "anc2vec.npz"
    _write_artifact(artifact, ["GO:0000001"], np.array([[1.0]], dtype=np.float32))
    a = get_index(str(artifact))
    b = get_index(str(artifact))
    assert a is b
    get_index.cache_clear()


def test_index_raises_on_missing_artifact(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError):
        Anc2VecIndex(tmp_path / "missing.npz")
