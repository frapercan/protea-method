"""Tests for the torch KNN backend in ``protea_method.knn_search``."""

from __future__ import annotations

from typing import Any

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


def assert_same_ranking(
    np_hits: list[tuple[str, float]],
    t_hits: list[tuple[str, float]],
    where: str,
    rtol: float = 1e-5,
    atol: float = 1e-6,
) -> None:
    """Two backends rank the same, up to ties float32 cannot resolve.

    Comparing accession lists element by element asks float32 to order numbers it
    cannot distinguish. On the fixture corpus, query 44 puts P00314 at 222.10226440
    under numpy and P00142 at 222.10224915 under torch: a relative difference of
    6.9e-8, below float32's own epsilon of about 1.2e-7. Which of the two comes
    sixth is decided by summation order, not by the data, and demanding one of the
    two answers makes a passing test a statement about the accumulation order of
    whichever backend was written first.

    So: the distance sequences must agree within tolerance, and an accession must
    match wherever the distance at that position is separated from its neighbours
    by more than the tolerance. Where it is not, either order is correct and the
    test says so instead of picking.
    """
    np_d = [d for _, d in np_hits]
    t_d = [d for _, d in t_hits]
    np.testing.assert_allclose(t_d, np_d, rtol=rtol, atol=atol, err_msg=f"{where}: distances diverge")

    def tie_group(dists: list[float], i: int) -> set[int]:
        """Positions whose distance is indistinguishable from position ``i``."""
        return {
            j
            for j in range(len(dists))
            if abs(dists[j] - dists[i]) <= atol + rtol * abs(dists[i])
        }

    for i, (np_acc, t_acc) in enumerate(zip([a for a, _ in np_hits], [a for a, _ in t_hits], strict=True)):
        if np_acc == t_acc:
            continue
        group = tie_group(np_d, i)
        assert len(group) > 1, f"{where}: position {i} differs and is not a tie ({np_acc} vs {t_acc})"
        # Within a tie group the two backends must still return the same set,
        # only in a different order. A genuinely wrong neighbour is still caught.
        assert {np_hits[j][0] for j in group} == {t_hits[j][0] for j in group}, (
            f"{where}: position {i} differs and the tied set differs too"
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
        assert_same_ranking(np_hits, t_hits, f"query {q_i} L2")


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
# OOM recovery: the rows that did not fit must still be answered
# ---------------------------------------------------------------------------


def _oom() -> RuntimeError:
    """The message torch raises when an allocation does not fit."""
    return RuntimeError("CUDA out of memory. Tried to allocate 20.00 GiB")


def test_an_oom_reprocesses_the_rows_it_could_not_fit(
    corpus: tuple[np.ndarray, np.ndarray, list[str]],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Shrinking the chunk must not skip the rows that provoked the shrink.

    This is a regression test, and the bug it guards against did not announce
    itself. Recovery kept the first half of the chunk and left the second half
    to an outer loop that had already stepped past it, so the result list came
    back short. Results are positional, so nothing raised: every query after
    the first OOM was handed a later query's neighbours and scored against its
    own ground truth. The observable symptom was a model that looked weak.
    """
    queries, refs, accessions = corpus
    k = 10
    expected = search_knn(queries, refs, accessions, k=k, backend="numpy", metric="cosine")

    monkeypatch.setenv("PROTEA_KNN_DEVICE", "cpu")
    monkeypatch.setenv("PROTEA_KNN_CHUNK_SIZE", "16")

    real_topk = torch.topk
    refusals = 0

    def flaky_topk(tensor: Any, k_arg: int, **kwargs: object) -> object:
        """Refuse anything wider than 4 rows, forcing 16 -> 8 -> 4."""
        nonlocal refusals
        if tensor.shape[0] > 4:
            refusals += 1
            raise _oom()
        return real_topk(tensor, k_arg, **kwargs)

    monkeypatch.setattr(torch, "topk", flaky_topk)

    got = search_knn(queries, refs, accessions, k=k, backend="torch", metric="cosine")

    assert refusals > 0, "the OOM path was never entered, so nothing was tested"
    assert len(got) == len(queries), "queries were dropped rather than retried"
    for q_i, (exp_hits, got_hits) in enumerate(zip(expected, got, strict=True)):
        assert [a for a, _ in got_hits] == [a for a, _ in exp_hits], (
            f"query {q_i} was answered with another query's neighbours"
        )


def test_an_oom_on_a_single_row_fails_instead_of_returning_short(
    small_corpus: tuple[np.ndarray, np.ndarray, list[str]],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """When halving can no longer help, the search must give up loudly."""
    queries, refs, accessions = small_corpus
    monkeypatch.setenv("PROTEA_KNN_DEVICE", "cpu")
    monkeypatch.setenv("PROTEA_KNN_CHUNK_SIZE", "4")

    def always_oom(tensor: Any, k_arg: int, **kwargs: object) -> object:
        raise _oom()

    monkeypatch.setattr(torch, "topk", always_oom)

    with pytest.raises(RuntimeError, match="out of memory"):
        search_knn(queries, refs, accessions, k=3, backend="torch", metric="cosine")


def test_a_non_oom_error_is_not_retried(
    small_corpus: tuple[np.ndarray, np.ndarray, list[str]],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Only memory pressure justifies shrinking; other faults propagate."""
    queries, refs, accessions = small_corpus
    monkeypatch.setenv("PROTEA_KNN_DEVICE", "cpu")

    def broken_topk(tensor: Any, k_arg: int, **kwargs: object) -> object:
        raise RuntimeError("device-side assert triggered")

    monkeypatch.setattr(torch, "topk", broken_topk)

    with pytest.raises(RuntimeError, match="device-side assert"):
        search_knn(queries, refs, accessions, k=3, backend="torch", metric="cosine")


def test_a_short_backend_return_is_refused(
    small_corpus: tuple[np.ndarray, np.ndarray, list[str]],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The positional contract is checked at the boundary, not trusted.

    Any backend can regress the same way, so the guard lives in ``search_knn``
    rather than in the backend that happened to break.
    """
    queries, refs, accessions = small_corpus
    monkeypatch.setattr(
        "protea_method.knn_search._search_torch",
        lambda *a, **kw: [[(accessions[0], 0.0)]],  # one row for five queries
    )

    with pytest.raises(RuntimeError, match="result rows"):
        search_knn(queries, refs, accessions, k=3, backend="torch", metric="cosine")


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
        assert_same_ranking(np_hits, t_hits, f"CUDA L2 query {q_i}")
