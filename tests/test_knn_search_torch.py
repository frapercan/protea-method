"""Tests for the torch KNN backend in ``protea_method.knn_search``."""

from __future__ import annotations

import numpy as np
import pytest

torch = pytest.importorskip("torch", reason="torch not installed; skipping torch KNN tests")

from protea_method.knn_search import search_knn  # noqa: E402

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def corpus() -> tuple[np.ndarray, np.ndarray, list[str]]:
    """Synthetic corpus: N=1000 refs, D=128, Q=50 queries, reproducible."""
    rng = np.random.default_rng(42)
    queries = rng.standard_normal(size=(50, 128)).astype(np.float32)
    refs = rng.standard_normal(size=(1000, 128)).astype(np.float32)
    accessions = [f"P{i:05d}" for i in range(1000)]
    return queries, refs, accessions


@pytest.fixture
def small_corpus() -> tuple[np.ndarray, np.ndarray, list[str]]:
    """Tiny corpus for smoke / edge-case tests."""
    rng = np.random.default_rng(7)
    queries = rng.standard_normal(size=(5, 8)).astype(np.float32)
    refs = rng.standard_normal(size=(20, 8)).astype(np.float32)
    accessions = [f"R{i:03d}" for i in range(20)]
    return queries, refs, accessions


# ---------------------------------------------------------------------------
# Accuracy: torch vs numpy reference
# ---------------------------------------------------------------------------


def test_torch_cosine_matches_numpy(
    corpus: tuple[np.ndarray, np.ndarray, list[str]],
) -> None:
    """torch cosine results must match numpy reference within tolerance."""
    queries, refs, accessions = corpus
    k = 10

    np_results = search_knn(queries, refs, accessions, k=k, backend="numpy", metric="cosine")
    t_results = search_knn(queries, refs, accessions, k=k, backend="torch", metric="cosine")

    assert len(t_results) == len(np_results) == 50

    for q_i, (np_hits, t_hits) in enumerate(zip(np_results, t_results, strict=True)):
        assert len(t_hits) == len(np_hits), f"query {q_i}: hit count mismatch"
        np_accs = [a for a, _ in np_hits]
        t_accs = [a for a, _ in t_hits]
        # Indices must match exactly (no distance ties in this synthetic corpus).
        assert np_accs == t_accs, f"query {q_i}: top-k accessions differ: {np_accs} vs {t_accs}"
        # Distances must be numerically close.
        np_dists = np.array([d for _, d in np_hits])
        t_dists = np.array([d for _, d in t_hits])
        np.testing.assert_allclose(
            t_dists, np_dists, rtol=1e-5, atol=1e-6,
            err_msg=f"query {q_i}: cosine distances diverge",
        )


def test_torch_l2_matches_numpy(
    corpus: tuple[np.ndarray, np.ndarray, list[str]],
) -> None:
    """torch L2 results must match numpy reference within tolerance."""
    queries, refs, accessions = corpus
    k = 10

    np_results = search_knn(queries, refs, accessions, k=k, backend="numpy", metric="l2")
    t_results = search_knn(queries, refs, accessions, k=k, backend="torch", metric="l2")

    for q_i, (np_hits, t_hits) in enumerate(zip(np_results, t_results, strict=True)):
        np_accs = [a for a, _ in np_hits]
        t_accs = [a for a, _ in t_hits]
        assert np_accs == t_accs, f"query {q_i}: L2 accessions differ"
        np_dists = np.array([d for _, d in np_hits])
        t_dists = np.array([d for _, d in t_hits])
        np.testing.assert_allclose(
            t_dists, np_dists, rtol=1e-5, atol=1e-6,
            err_msg=f"query {q_i}: L2 distances diverge",
        )


# ---------------------------------------------------------------------------
# Smoke: chunked path consistency
# ---------------------------------------------------------------------------


def test_torch_chunked_path_consistent(
    corpus: tuple[np.ndarray, np.ndarray, list[str]],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Chunk size=2 must yield the same result as the default chunk size."""
    queries, refs, accessions = corpus
    k = 5

    full = search_knn(queries, refs, accessions, k=k, backend="torch", metric="cosine")
    monkeypatch.setenv("PROTEA_KNN_CHUNK_SIZE", "2")
    chunked = search_knn(queries, refs, accessions, k=k, backend="torch", metric="cosine")

    for q_i, (f_hits, c_hits) in enumerate(zip(full, chunked, strict=True)):
        assert [a for a, _ in f_hits] == [a for a, _ in c_hits], (
            f"query {q_i}: chunked path produced different accessions"
        )
        np.testing.assert_allclose(
            np.array([d for _, d in c_hits]),
            np.array([d for _, d in f_hits]),
            rtol=1e-5,
            atol=1e-6,
        )


# ---------------------------------------------------------------------------
# Edge cases
# ---------------------------------------------------------------------------


def test_torch_k_above_corpus_size(
    small_corpus: tuple[np.ndarray, np.ndarray, list[str]],
) -> None:
    """k > n_refs is silently capped at n_refs."""
    queries, refs, accessions = small_corpus
    results = search_knn(queries, refs, accessions, k=100, backend="torch", metric="cosine")
    for hits in results:
        assert len(hits) == 20


def test_torch_distance_threshold(
    small_corpus: tuple[np.ndarray, np.ndarray, list[str]],
) -> None:
    """Hits beyond the threshold are excluded."""
    queries, refs, accessions = small_corpus
    unrestricted = search_knn(queries, refs, accessions, k=10, backend="torch", metric="cosine")
    threshold = unrestricted[0][2][1]  # third hit's distance as cap
    capped = search_knn(
        queries, refs, accessions, k=10, backend="torch", metric="cosine",
        distance_threshold=threshold,
    )
    for hits in capped:
        for _, d in hits:
            assert d <= threshold + 1e-6


def test_torch_output_sorted_ascending(
    small_corpus: tuple[np.ndarray, np.ndarray, list[str]],
) -> None:
    """Results must be sorted ascending by distance."""
    queries, refs, accessions = small_corpus
    for metric in ("cosine", "l2"):
        results = search_knn(
            queries, refs, accessions, k=10, backend="torch", metric=metric
        )
        for hits in results:
            dists = [d for _, d in hits]
            assert dists == sorted(dists), f"{metric}: hits not sorted"


def test_torch_unknown_metric_raises(
    small_corpus: tuple[np.ndarray, np.ndarray, list[str]],
) -> None:
    queries, refs, accessions = small_corpus
    with pytest.raises(ValueError, match="Unknown metric"):
        search_knn(queries, refs, accessions, k=3, backend="torch", metric="hamming")


def test_torch_device_cpu_override(
    corpus: tuple[np.ndarray, np.ndarray, list[str]],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """PROTEA_KNN_DEVICE=cpu must run without error on any machine."""
    monkeypatch.setenv("PROTEA_KNN_DEVICE", "cpu")
    queries, refs, accessions = corpus
    results = search_knn(queries, refs, accessions, k=5, backend="torch", metric="cosine")
    assert len(results) == 50
    for hits in results:
        assert len(hits) == 5


# ---------------------------------------------------------------------------
# GPU-specific: skip when CUDA absent
# ---------------------------------------------------------------------------


@pytest.mark.gpu
def test_torch_cosine_on_cuda(
    corpus: tuple[np.ndarray, np.ndarray, list[str]],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """On a CUDA machine, torch cosine must match numpy reference."""
    if not torch.cuda.is_available():
        pytest.skip("CUDA not available")
    monkeypatch.setenv("PROTEA_KNN_DEVICE", "cuda")
    queries, refs, accessions = corpus
    k = 10
    np_results = search_knn(queries, refs, accessions, k=k, backend="numpy", metric="cosine")
    t_results = search_knn(queries, refs, accessions, k=k, backend="torch", metric="cosine")
    for q_i, (np_hits, t_hits) in enumerate(zip(np_results, t_results, strict=True)):
        np_accs = [a for a, _ in np_hits]
        t_accs = [a for a, _ in t_hits]
        assert np_accs == t_accs, f"CUDA query {q_i}: accessions differ"


@pytest.mark.gpu
def test_torch_l2_on_cuda(
    corpus: tuple[np.ndarray, np.ndarray, list[str]],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """On a CUDA machine, torch L2 must match numpy reference."""
    if not torch.cuda.is_available():
        pytest.skip("CUDA not available")
    monkeypatch.setenv("PROTEA_KNN_DEVICE", "cuda")
    queries, refs, accessions = corpus
    k = 10
    np_results = search_knn(queries, refs, accessions, k=k, backend="numpy", metric="l2")
    t_results = search_knn(queries, refs, accessions, k=k, backend="torch", metric="l2")
    for q_i, (np_hits, t_hits) in enumerate(zip(np_results, t_results, strict=True)):
        np_accs = [a for a, _ in np_hits]
        t_accs = [a for a, _ in t_hits]
        assert np_accs == t_accs, f"CUDA L2 query {q_i}: accessions differ"
