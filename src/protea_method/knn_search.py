"""K-nearest-neighbor search backends for GO term prediction.

Backends
--------
numpy
    Exact brute-force cosine or L2 distance via matrix multiplication.
    No dependencies beyond NumPy. Suitable for reference sets up to ~100K.

faiss
    Wraps the FAISS library (``faiss-cpu``).
    Supports exact (Flat) and approximate (IVFFlat, HNSW) indices.
    Significantly faster for large reference sets (>100K vectors).

Metric convention
-----------------
Both backends return **distances** (lower = more similar):

- ``cosine``  -> D = 1 - cosine_similarity in [0, 2]
- ``l2``      -> D = squared Euclidean distance in [0, infinity)

Returned type
-------------
``search_knn`` returns::

    list[list[tuple[str, float]]]

One inner list per query; each tuple is ``(ref_accession, distance)``,
sorted ascending by distance, length <= ``k`` (may be shorter if
``distance_threshold`` filters them out).

Original file: PROTEA's ``protea/core/knn_search.py``. The only
PROTEA-internal dependency was ``get_tuning().operation.numpy_query_chunk``
for the per-chunk memory ceiling; replaced here with the environment
variable ``PROTEA_METHOD_NUMPY_QUERY_CHUNK`` (default 500) so the
library stays free of protea-core.
"""

from __future__ import annotations

import os
from typing import Any

import numpy as np


def _numpy_query_chunk_default() -> int:
    """Per-chunk query count for the numpy backend.

    The full ``n_queries x n_refs x 4B`` distance matrix dominates RAM
    (with 500K refs a naive call materialises 10+ GB per aspect).
    500 keeps a 500K-ref bank under ~1 GB peak. Override via
    ``PROTEA_METHOD_NUMPY_QUERY_CHUNK``.
    """
    raw = os.environ.get("PROTEA_METHOD_NUMPY_QUERY_CHUNK")
    if raw is None:
        return 500
    return int(raw)


def search_knn(
    query_embeddings: np.ndarray,
    ref_embeddings: np.ndarray,
    ref_accessions: list[str],
    k: int,
    *,
    distance_threshold: float | None = None,
    backend: str = "numpy",
    metric: str = "cosine",
    pre_normalized: bool = False,
    faiss_index_type: str = "Flat",
    faiss_nlist: int = 100,
    faiss_nprobe: int = 10,
    faiss_hnsw_m: int = 32,
    faiss_hnsw_ef_search: int = 64,
) -> list[list[tuple[str, float]]]:
    """Search for the k nearest reference proteins for each query embedding.

    Parameters
    ----------
    query_embeddings:
        Shape ``(n_queries, dim)``. Need not be normalised.
    ref_embeddings:
        Shape ``(n_refs, dim)``. Need not be normalised.
    ref_accessions:
        Length ``n_refs``. Maps index positions to accession strings.
    k:
        Maximum number of neighbours to return per query.
    distance_threshold:
        If set, discard neighbours with distance > threshold.
    backend:
        ``"numpy"`` (exact brute-force) or ``"faiss"``.
    metric:
        ``"cosine"`` or ``"l2"``.
    pre_normalized:
        When ``True`` and ``metric == "cosine"``, the caller guarantees that
        ``ref_embeddings`` rows are already L2-normalised, so the backend
        skips the per-call normalisation. No-op for ``l2``.
    faiss_index_type:
        One of ``"Flat"``, ``"IVFFlat"``, ``"HNSW"`` (ignored for numpy).
    faiss_nlist:
        Number of Voronoi cells for ``IVFFlat``.
    faiss_nprobe:
        Cells visited at search time for ``IVFFlat``.
    faiss_hnsw_m:
        Connections per node for ``HNSW``.
    faiss_hnsw_ef_search:
        Beam width at search time for ``HNSW``.

    Returns
    -------
    list[list[tuple[str, float]]]
        Outer list: one entry per query.
        Inner list: ``(ref_accession, distance)`` sorted ascending by distance.
    """
    if backend == "faiss":
        return _search_faiss(
            query_embeddings,
            ref_embeddings,
            ref_accessions,
            k,
            distance_threshold=distance_threshold,
            metric=metric,
            index_type=faiss_index_type,
            nlist=faiss_nlist,
            nprobe=faiss_nprobe,
            hnsw_m=faiss_hnsw_m,
            hnsw_ef_search=faiss_hnsw_ef_search,
        )
    if backend == "numpy":
        return _search_numpy(
            query_embeddings,
            ref_embeddings,
            ref_accessions,
            k,
            distance_threshold=distance_threshold,
            metric=metric,
            pre_normalized=pre_normalized,
        )
    raise ValueError(f"Unknown search backend: {backend!r}. Choose 'numpy' or 'faiss'.")


def _search_numpy(
    Q: np.ndarray,
    R: np.ndarray,
    ref_accessions: list[str],
    k: int,
    *,
    distance_threshold: float | None,
    metric: str,
    pre_normalized: bool = False,
) -> list[list[tuple[str, float]]]:
    """Exact brute-force search via chunked matrix multiplication.

    Chunk-invariant work (R normalisation for cosine, ``||R||^2`` for L2,
    and the transposed ``R.T`` view) is computed once and reused across
    chunks.

    Top-k selection uses ``np.argpartition`` (O(n_refs) per query)
    followed by a partial sort of the k-slice, instead of a full
    ``argsort`` -- for k << n_refs this is ~30x faster on 500k-vector
    banks.
    """
    if metric == "cosine":
        if pre_normalized:
            R_ready = R
        else:
            R_ready = R / (np.linalg.norm(R, axis=1, keepdims=True) + 1e-9)
        R2 = None
    elif metric == "l2":
        R_ready = R
        R2 = (R**2).sum(axis=1)  # shape (n_refs,), reused by every chunk
    else:
        raise ValueError(f"Unknown metric: {metric!r}. Choose 'cosine' or 'l2'.")

    R_T = R_ready.T  # contiguous view; shared by all chunks
    n_refs = R.shape[0]
    k_eff = min(k, n_refs)

    results: list[list[tuple[str, float]]] = []
    n_queries = Q.shape[0]
    query_chunk = _numpy_query_chunk_default()

    for start in range(0, n_queries, query_chunk):
        Q_chunk = Q[start : start + query_chunk]
        if metric == "cosine":
            Q_n = Q_chunk / (np.linalg.norm(Q_chunk, axis=1, keepdims=True) + 1e-9)
            dist = 1.0 - (Q_n @ R_T)
        else:  # l2
            Q2 = (Q_chunk**2).sum(axis=1, keepdims=True)
            dist = np.maximum(0.0, Q2 + R2 - 2.0 * (Q_chunk @ R_T))

        n_rows = dist.shape[0]
        if k_eff < n_refs:
            part = np.argpartition(dist, k_eff - 1, axis=1)[:, :k_eff]
            row_range = np.arange(n_rows)[:, None]
            part_d = dist[row_range, part]
            sort_in_part = np.argsort(part_d, axis=1)
            top_per_row = part[row_range, sort_in_part]
        else:
            top_per_row = np.argsort(dist, axis=1)[:, :k_eff]

        for row_i in range(n_rows):
            row = dist[row_i]
            top = top_per_row[row_i]
            if distance_threshold is not None:
                top = top[row[top] <= distance_threshold]
            results.append([(ref_accessions[int(i)], float(row[i])) for i in top])
        del dist
    return results


def _compute_distance_matrix(
    Q: np.ndarray,
    R: np.ndarray,
    metric: str,
) -> np.ndarray:
    """Dense distance matrix between all query/ref pairs.

    Kept as a standalone helper for tests that exercise the metric-
    dispatch error path. Production code (``_search_numpy``) inlines
    the computation so it can hoist chunk-invariant work out of the
    loop.
    """
    if metric == "cosine":
        Q_n = Q / (np.linalg.norm(Q, axis=1, keepdims=True) + 1e-9)
        R_n = R / (np.linalg.norm(R, axis=1, keepdims=True) + 1e-9)
        return np.asarray(1.0 - (Q_n @ R_n.T))
    if metric == "l2":
        Q2 = (Q**2).sum(axis=1, keepdims=True)
        R2 = (R**2).sum(axis=1)
        return np.asarray(np.maximum(0.0, Q2 + R2 - 2.0 * (Q @ R.T)))
    raise ValueError(f"Unknown metric: {metric!r}. Choose 'cosine' or 'l2'.")


def _search_faiss(
    Q: np.ndarray,
    R: np.ndarray,
    ref_accessions: list[str],
    k: int,
    *,
    distance_threshold: float | None,
    metric: str,
    index_type: str,
    nlist: int,
    nprobe: int,
    hnsw_m: int,
    hnsw_ef_search: int,
) -> list[list[tuple[str, float]]]:
    """FAISS-based approximate or exact nearest-neighbour search."""
    try:
        import faiss
    except ImportError as exc:
        raise ImportError("FAISS is not installed. Run `pip install faiss-cpu`.") from exc

    n_refs, dim = R.shape

    Q_f = np.ascontiguousarray(Q, dtype=np.float32)
    R_f = np.ascontiguousarray(R, dtype=np.float32)

    use_ip = metric == "cosine"
    if use_ip:
        faiss.normalize_L2(Q_f)
        faiss.normalize_L2(R_f)

    index = _build_faiss_index(
        R_f,
        dim,
        n_refs,
        metric=metric,
        index_type=index_type,
        nlist=nlist,
        nprobe=nprobe,
        hnsw_m=hnsw_m,
        hnsw_ef_search=hnsw_ef_search,
        use_ip=use_ip,
    )

    k_search = min(k * 4, n_refs)
    raw_distances, indices = index.search(Q_f, k_search)

    results: list[list[tuple[str, float]]] = []
    for dist_row, idx_row in zip(raw_distances, indices, strict=False):
        hits: list[tuple[str, float]] = []
        seen: set[str] = set()
        for raw_d, idx in zip(dist_row, idx_row, strict=False):
            if idx < 0:  # FAISS sentinel for "not enough neighbours"
                continue
            d = float(1.0 - raw_d) if use_ip else float(raw_d)
            if distance_threshold is not None and d > distance_threshold:
                break
            acc = ref_accessions[idx]
            if acc in seen:
                continue
            seen.add(acc)
            hits.append((acc, d))
            if len(hits) >= k:
                break
        results.append(hits)

    return results


def _build_faiss_index(
    R_f: np.ndarray,
    dim: int,
    n_refs: int,
    *,
    metric: str,
    index_type: str,
    nlist: int,
    nprobe: int,
    hnsw_m: int,
    hnsw_ef_search: int,
    use_ip: bool,
) -> Any:
    import faiss

    faiss_metric = faiss.METRIC_INNER_PRODUCT if use_ip else faiss.METRIC_L2

    if index_type == "Flat":
        index = faiss.IndexFlatIP(dim) if use_ip else faiss.IndexFlatL2(dim)
    elif index_type == "IVFFlat":
        effective_nlist = max(1, min(nlist, n_refs))
        quantizer = faiss.IndexFlatIP(dim) if use_ip else faiss.IndexFlatL2(dim)
        index = faiss.IndexIVFFlat(quantizer, dim, effective_nlist, faiss_metric)
        index.train(R_f)
        index.nprobe = min(nprobe, effective_nlist)
    elif index_type == "HNSW":
        index = faiss.IndexHNSWFlat(dim, hnsw_m, faiss_metric)
        index.hnsw.efSearch = hnsw_ef_search
    else:
        raise ValueError(
            f"Unknown faiss_index_type: {index_type!r}. "
            f"Choose 'Flat', 'IVFFlat', or 'HNSW'."
        )

    index.add(R_f)
    return index


__all__ = ["search_knn"]
