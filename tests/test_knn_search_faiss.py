"""FAISS-backend coverage for ``protea_method.knn_search``.

The numpy backend has its own dedicated tests in
``tests/test_knn_search.py``; this module fills the FAISS branches
(``Flat`` / ``IVFFlat`` / ``HNSW`` index types, the cosine-vs-L2
dispatch, the duplicate-suppression and threshold-cutoff inner loop).

``faiss-cpu`` is a hard project dependency (declared in
``pyproject.toml``) so the import never fails on supported platforms.
"""

from __future__ import annotations

import numpy as np
import pytest

faiss = pytest.importorskip("faiss")  # pragma: no cover

from protea_method.knn_search import _build_faiss_index, search_knn  # noqa: E402


@pytest.fixture
def small_corpus() -> tuple[np.ndarray, np.ndarray, list[str]]:
    rng = np.random.default_rng(7)
    queries = rng.standard_normal(size=(4, 8)).astype(np.float32)
    refs = rng.standard_normal(size=(40, 8)).astype(np.float32)
    accessions = [f"R{i:03d}" for i in range(40)]
    return queries, refs, accessions


def test_faiss_flat_cosine(small_corpus: tuple[np.ndarray, np.ndarray, list[str]]) -> None:
    queries, refs, accessions = small_corpus
    results = search_knn(
        queries, refs, accessions, k=3,
        backend="faiss", metric="cosine", faiss_index_type="Flat",
    )
    assert len(results) == 4
    for hits in results:
        assert len(hits) == 3
        distances = [d for _, d in hits]
        assert distances == sorted(distances)
        for _, d in hits:
            assert -1e-3 <= d <= 2.0 + 1e-3


def test_faiss_flat_l2(small_corpus: tuple[np.ndarray, np.ndarray, list[str]]) -> None:
    queries, refs, accessions = small_corpus
    results = search_knn(
        queries, refs, accessions, k=3,
        backend="faiss", metric="l2", faiss_index_type="Flat",
    )
    for hits in results:
        distances = [d for _, d in hits]
        assert distances == sorted(distances)
        for _, d in hits:
            assert d >= -1e-6


def test_faiss_ivfflat_cosine(small_corpus: tuple[np.ndarray, np.ndarray, list[str]]) -> None:
    queries, refs, accessions = small_corpus
    results = search_knn(
        queries, refs, accessions, k=3,
        backend="faiss", metric="cosine",
        faiss_index_type="IVFFlat", faiss_nlist=4, faiss_nprobe=4,
    )
    assert len(results) == 4
    for hits in results:
        for _, d in hits:
            assert -1e-3 <= d <= 2.0 + 1e-3


def test_faiss_hnsw_cosine(small_corpus: tuple[np.ndarray, np.ndarray, list[str]]) -> None:
    queries, refs, accessions = small_corpus
    results = search_knn(
        queries, refs, accessions, k=3,
        backend="faiss", metric="cosine",
        faiss_index_type="HNSW", faiss_hnsw_m=8, faiss_hnsw_ef_search=16,
    )
    assert len(results) == 4
    for hits in results:
        for _, d in hits:
            assert -1e-3 <= d <= 2.0 + 1e-3


def test_faiss_distance_threshold(
    small_corpus: tuple[np.ndarray, np.ndarray, list[str]],
) -> None:
    queries, refs, accessions = small_corpus
    capped = search_knn(
        queries, refs, accessions, k=20,
        backend="faiss", metric="cosine", faiss_index_type="Flat",
        distance_threshold=0.5,
    )
    for hits in capped:
        for _, d in hits:
            assert d <= 0.5 + 1e-6


def test_faiss_duplicate_accession_suppression(
    small_corpus: tuple[np.ndarray, np.ndarray, list[str]],
) -> None:
    """When the same accession appears twice in the ref bank, faiss
    dedupes by accession in the result.
    """
    queries, refs, accessions = small_corpus
    refs_dup = np.vstack([refs, refs[:5]])
    accessions_dup = list(accessions) + accessions[:5]
    hits_per_query = search_knn(
        queries, refs_dup, accessions_dup, k=10,
        backend="faiss", metric="cosine", faiss_index_type="Flat",
    )
    for hits in hits_per_query:
        accs = [a for a, _ in hits]
        assert len(accs) == len(set(accs))


def test_faiss_unknown_index_type_raises(
    small_corpus: tuple[np.ndarray, np.ndarray, list[str]],
) -> None:
    queries, refs, accessions = small_corpus
    with pytest.raises(ValueError, match="Unknown faiss_index_type"):
        search_knn(
            queries, refs, accessions, k=3,
            backend="faiss", metric="cosine", faiss_index_type="LSH",
        )


def test_build_faiss_index_l2_flat(
    small_corpus: tuple[np.ndarray, np.ndarray, list[str]],
) -> None:
    """Cover the L2-flat branch directly so the build helper is exercised."""
    _, refs, _ = small_corpus
    index = _build_faiss_index(
        refs.astype(np.float32),
        dim=refs.shape[1],
        n_refs=refs.shape[0],
        metric="l2",
        index_type="Flat",
        nlist=10,
        nprobe=10,
        hnsw_m=16,
        hnsw_ef_search=32,
        use_ip=False,
    )
    assert index.ntotal == refs.shape[0]
