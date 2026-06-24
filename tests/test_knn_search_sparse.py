"""Tests for the inverted-index sparse k-WTA backend of ``search_knn``.

The sparse backend computes the *exact* cosine over learned k-WTA codes,
so its top-N ranking must match a dense cosine reference over the same
codes (up to tie-breaking among equal distances). These tests build small
synthetic k-WTA code banks and assert ranking + distance parity, plus the
backend's contract (cosine-only, threshold, k-above-corpus, empty refs).
"""

from __future__ import annotations

import numpy as np
import pytest

from protea_method.knn_search import (
    _build_inverted_index,
    _concat_ranges,
    search_knn,
)


def _make_kwta(
    n: int,
    dim: int,
    active_k: int,
    seed: int,
) -> np.ndarray:
    """Build ``n`` k-WTA code rows: top ``active_k`` magnitudes kept, rest 0."""
    rng = np.random.default_rng(seed)
    dense = rng.standard_normal(size=(n, dim)).astype(np.float32)
    codes = np.zeros_like(dense)
    for row in range(n):
        keep = np.argpartition(np.abs(dense[row]), dim - active_k)[-active_k:]
        # ReLU-style positive codes (k-WTA encoders emit non-negative gates).
        codes[row, keep] = np.abs(dense[row, keep])
    return codes


@pytest.fixture
def kwta_corpus() -> tuple[np.ndarray, np.ndarray, list[str]]:
    dim, active_k = 64, 8
    queries = _make_kwta(6, dim, active_k, seed=1)
    refs = _make_kwta(40, dim, active_k, seed=2)
    accessions = [f"R{i:03d}" for i in range(40)]
    return queries, refs, accessions


def test_sparse_matches_dense_ranking(
    kwta_corpus: tuple[np.ndarray, np.ndarray, list[str]],
) -> None:
    """Top-N ranking from the sparse backend equals the dense cosine reference."""
    queries, refs, accessions = kwta_corpus
    sparse = search_knn(queries, refs, accessions, k=5, backend="sparse", metric="cosine")
    dense = search_knn(queries, refs, accessions, k=5, backend="numpy", metric="cosine")

    assert len(sparse) == len(dense) == 6
    for s_hits, d_hits in zip(sparse, dense, strict=True):
        assert [acc for acc, _ in s_hits] == [acc for acc, _ in d_hits]
        for (_, s_d), (_, d_d) in zip(s_hits, d_hits, strict=True):
            assert s_d == pytest.approx(d_d, abs=1e-5)


def test_sparse_distances_in_cosine_range(
    kwta_corpus: tuple[np.ndarray, np.ndarray, list[str]],
) -> None:
    queries, refs, accessions = kwta_corpus
    results = search_knn(queries, refs, accessions, k=5, backend="sparse", metric="cosine")
    for hits in results:
        distances = [d for _, d in hits]
        assert distances == sorted(distances)
        for _, d in hits:
            # k-WTA codes are non-negative -> cosine similarity in [0, 1]
            # -> distance in [0, 1] (allow tiny float slack).
            assert -1e-5 <= d <= 1.0 + 1e-5


def test_sparse_pre_normalized_path_matches(
    kwta_corpus: tuple[np.ndarray, np.ndarray, list[str]],
) -> None:
    queries, refs, accessions = kwta_corpus
    refs_norm = refs / (np.linalg.norm(refs, axis=1, keepdims=True) + 1e-9)
    fast = search_knn(
        queries, refs_norm, accessions, k=5, backend="sparse",
        metric="cosine", pre_normalized=True,
    )
    slow = search_knn(
        queries, refs, accessions, k=5, backend="sparse",
        metric="cosine", pre_normalized=False,
    )
    for fast_hits, slow_hits in zip(fast, slow, strict=True):
        assert [acc for acc, _ in fast_hits] == [acc for acc, _ in slow_hits]


def test_sparse_distance_threshold(
    kwta_corpus: tuple[np.ndarray, np.ndarray, list[str]],
) -> None:
    queries, refs, accessions = kwta_corpus
    unrestricted = search_knn(queries, refs, accessions, k=10, backend="sparse", metric="cosine")
    threshold = unrestricted[0][2][1]
    capped = search_knn(
        queries, refs, accessions, k=10, backend="sparse",
        metric="cosine", distance_threshold=threshold,
    )
    for hits in capped:
        for _, d in hits:
            assert d <= threshold + 1e-6


def test_sparse_k_above_corpus_size(
    kwta_corpus: tuple[np.ndarray, np.ndarray, list[str]],
) -> None:
    queries, refs, accessions = kwta_corpus
    results = search_knn(queries, refs, accessions, k=100, backend="sparse", metric="cosine")
    for hits in results:
        assert len(hits) == 40  # capped at corpus size


def test_sparse_rejects_l2(
    kwta_corpus: tuple[np.ndarray, np.ndarray, list[str]],
) -> None:
    queries, refs, accessions = kwta_corpus
    with pytest.raises(ValueError, match="only metric='cosine'"):
        search_knn(queries, refs, accessions, k=3, backend="sparse", metric="l2")


def test_sparse_query_with_no_shared_dims() -> None:
    """A query whose active dims never appear in any ref scores 0 (dist 1.0)."""
    dim = 16
    # refs active only in the first half; query active only in the second half.
    refs = np.zeros((3, dim), dtype=np.float32)
    refs[0, [0, 1]] = 1.0
    refs[1, [2, 3]] = 1.0
    refs[2, [0, 3]] = 1.0
    query = np.zeros((1, dim), dtype=np.float32)
    query[0, [8, 9]] = 1.0
    accessions = ["A", "B", "C"]
    results = search_knn(query, refs, accessions, k=3, backend="sparse", metric="cosine")
    assert len(results[0]) == 3
    for _, d in results[0]:
        assert d == pytest.approx(1.0, abs=1e-6)


def test_build_inverted_index_structure() -> None:
    """The inverted index reproduces the non-zero structure column-by-column."""
    R = np.array(
        [
            [1.0, 0.0, 2.0],
            [0.0, 3.0, 0.0],
            [4.0, 0.0, 5.0],
        ],
        dtype=np.float32,
    )
    index = _build_inverted_index(R)
    assert index.n_refs == 3
    assert index.indptr.tolist() == [0, 2, 3, 5]

    def postings(d: int) -> tuple[np.ndarray, np.ndarray]:
        lo, hi = int(index.indptr[d]), int(index.indptr[d + 1])
        return index.rows[lo:hi], index.vals[lo:hi]

    rows0, vals0 = postings(0)  # dim 0: rows 0 and 2
    np.testing.assert_array_equal(rows0, np.array([0, 2]))
    np.testing.assert_array_equal(vals0, np.array([1.0, 4.0], dtype=np.float32))
    rows1, vals1 = postings(1)  # dim 1: row 1
    np.testing.assert_array_equal(rows1, np.array([1]))
    np.testing.assert_array_equal(vals1, np.array([3.0], dtype=np.float32))
    rows2, vals2 = postings(2)  # dim 2: rows 0 and 2
    np.testing.assert_array_equal(rows2, np.array([0, 2]))
    np.testing.assert_array_equal(vals2, np.array([2.0, 5.0], dtype=np.float32))


def test_concat_ranges_matches_python_reference() -> None:
    """``_concat_ranges`` equals the looped arange-concatenation."""
    starts = np.array([2, 7, 0, 5], dtype=np.int64)
    ends = np.array([5, 9, 0, 8], dtype=np.int64)  # note an empty range (0,0)
    total = int((ends - starts).sum())
    got = _concat_ranges(starts, ends, total)
    expected = np.concatenate([np.arange(s, e) for s, e in zip(starts, ends, strict=True)])
    np.testing.assert_array_equal(got, expected)


def test_sparse_matches_dense_larger_dim() -> None:
    """Parity holds at a more realistic sparsity ratio (k=32 active of 512)."""
    dim, active_k = 512, 32
    queries = _make_kwta(8, dim, active_k, seed=7)
    refs = _make_kwta(120, dim, active_k, seed=9)
    accessions = [f"P{i:04d}" for i in range(120)]
    sparse = search_knn(queries, refs, accessions, k=10, backend="sparse", metric="cosine")
    dense = search_knn(queries, refs, accessions, k=10, backend="numpy", metric="cosine")
    for s_hits, d_hits in zip(sparse, dense, strict=True):
        assert [acc for acc, _ in s_hits] == [acc for acc, _ in d_hits]
