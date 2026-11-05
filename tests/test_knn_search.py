"""Coverage tests for ``protea_method.knn_search``."""

from __future__ import annotations

import numpy as np
import pytest

from protea_method.knn_search import (
    _compute_distance_matrix,
    _numpy_query_chunk_default,
    search_knn,
)


@pytest.fixture
def small_corpus() -> tuple[np.ndarray, np.ndarray, list[str]]:
    rng = np.random.default_rng(0)
    queries = rng.standard_normal(size=(5, 8)).astype(np.float32)
    refs = rng.standard_normal(size=(20, 8)).astype(np.float32)
    accessions = [f"R{i:03d}" for i in range(20)]
    return queries, refs, accessions


def test_search_knn_numpy_cosine(small_corpus: tuple[np.ndarray, np.ndarray, list[str]]) -> None:
    queries, refs, accessions = small_corpus
    results = search_knn(queries, refs, accessions, k=3, backend="numpy", metric="cosine")
    assert len(results) == 5
    for hits in results:
        assert len(hits) == 3
        distances = [d for _, d in hits]
        assert distances == sorted(distances)
        for _, d in hits:
            assert 0.0 <= d <= 2.0


def test_search_knn_numpy_l2(small_corpus: tuple[np.ndarray, np.ndarray, list[str]]) -> None:
    queries, refs, accessions = small_corpus
    results = search_knn(queries, refs, accessions, k=3, backend="numpy", metric="l2")
    for hits in results:
        distances = [d for _, d in hits]
        assert distances == sorted(distances)
        for _, d in hits:
            assert d >= 0.0


def test_search_knn_distance_threshold(
    small_corpus: tuple[np.ndarray, np.ndarray, list[str]],
) -> None:
    queries, refs, accessions = small_corpus
    unrestricted = search_knn(queries, refs, accessions, k=10, metric="cosine")
    threshold = unrestricted[0][2][1]  # use the third hit's distance as cap
    capped = search_knn(
        queries, refs, accessions, k=10, metric="cosine", distance_threshold=threshold,
    )
    for hits in capped:
        for _, d in hits:
            assert d <= threshold


def test_search_knn_pre_normalized_path(
    small_corpus: tuple[np.ndarray, np.ndarray, list[str]],
) -> None:
    queries, refs, accessions = small_corpus
    refs_norm = refs / (np.linalg.norm(refs, axis=1, keepdims=True) + 1e-9)
    fast = search_knn(
        queries, refs_norm, accessions, k=3, metric="cosine", pre_normalized=True,
    )
    slow = search_knn(queries, refs, accessions, k=3, metric="cosine", pre_normalized=False)
    for fast_hits, slow_hits in zip(fast, slow, strict=True):
        assert [acc for acc, _ in fast_hits] == [acc for acc, _ in slow_hits]


def test_search_knn_unknown_backend_raises(
    small_corpus: tuple[np.ndarray, np.ndarray, list[str]],
) -> None:
    queries, refs, accessions = small_corpus
    with pytest.raises(ValueError, match="Unknown search backend"):
        search_knn(queries, refs, accessions, k=3, backend="annoy")


def test_search_knn_unknown_metric_raises(
    small_corpus: tuple[np.ndarray, np.ndarray, list[str]],
) -> None:
    queries, refs, accessions = small_corpus
    with pytest.raises(ValueError, match="Unknown metric"):
        search_knn(queries, refs, accessions, k=3, backend="numpy", metric="hamming")


def test_search_knn_k_above_corpus_size(
    small_corpus: tuple[np.ndarray, np.ndarray, list[str]],
) -> None:
    queries, refs, accessions = small_corpus
    results = search_knn(queries, refs, accessions, k=100, backend="numpy", metric="cosine")
    for hits in results:
        assert len(hits) == 20  # capped at corpus size


def test_compute_distance_matrix_cosine(
    small_corpus: tuple[np.ndarray, np.ndarray, list[str]],
) -> None:
    queries, refs, _ = small_corpus
    dist = _compute_distance_matrix(queries, refs, "cosine")
    assert dist.shape == (5, 20)
    assert np.all(dist >= -1e-6)
    assert np.all(dist <= 2.0 + 1e-6)


def test_compute_distance_matrix_l2(
    small_corpus: tuple[np.ndarray, np.ndarray, list[str]],
) -> None:
    queries, refs, _ = small_corpus
    dist = _compute_distance_matrix(queries, refs, "l2")
    assert dist.shape == (5, 20)
    assert np.all(dist >= 0.0)


def test_compute_distance_matrix_unknown_metric(
    small_corpus: tuple[np.ndarray, np.ndarray, list[str]],
) -> None:
    queries, refs, _ = small_corpus
    with pytest.raises(ValueError, match="Unknown metric"):
        _compute_distance_matrix(queries, refs, "manhattan")


def test_numpy_query_chunk_env_override(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("PROTEA_METHOD_NUMPY_QUERY_CHUNK", "42")
    assert _numpy_query_chunk_default() == 42


def test_numpy_query_chunk_default(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("PROTEA_METHOD_NUMPY_QUERY_CHUNK", raising=False)
    assert _numpy_query_chunk_default() == 500


def test_search_knn_chunked_query_path(
    small_corpus: tuple[np.ndarray, np.ndarray, list[str]],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Force the query loop to chunk so that path is exercised."""
    monkeypatch.setenv("PROTEA_METHOD_NUMPY_QUERY_CHUNK", "2")
    queries, refs, accessions = small_corpus
    results = search_knn(queries, refs, accessions, k=3, backend="numpy", metric="cosine")
    assert len(results) == 5


def test_equal_distances_are_ordered_by_column_whatever_the_rest_of_the_row() -> None:
    """Ties are broken by position, and moving unrelated values cannot reorder them.

    Exploration 1 (2026-11-05) found the order of equal distances changing
    between two runs on a byte-identical bank, because the old selection's
    internal permutation depended on every other value of the row.
    """
    from protea_method.knn_search import _top_k_by_distance_then_position

    rng = np.random.default_rng(0)
    base = (rng.random(1000) + 1.0).astype(np.float32)
    ties = [700, 13, 512, 99]
    base[ties] = np.float32(0.25)
    others = np.setdiff1d(np.arange(1000), ties)
    for _ in range(25):
        row = base.copy()
        moved = rng.choice(others, size=60, replace=False)
        row[moved] += rng.random(60).astype(np.float32)
        top = _top_k_by_distance_then_position(row[None, :], 3)[0]
        assert top.tolist() == [13, 99, 512]


def test_twins_come_back_in_accession_order() -> None:
    """References carrying the same vector tie exactly; the bank order decides."""
    from protea_method.knn_search import _search_numpy

    v = np.array([[1.0, 0.0, 0.0]], dtype=np.float32)
    other = np.array([[0.0, 1.0, 0.0]], dtype=np.float32)
    R = np.vstack([other, v, other, v, v]).astype(np.float32)
    accessions = ["A1", "B2", "C3", "D4", "E5"]
    hits = _search_numpy(v, R, accessions, 3, distance_threshold=None, metric="cosine")
    assert [acc for acc, _ in hits[0]] == ["B2", "D4", "E5"]
