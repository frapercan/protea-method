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

torch
    Chunked GPU/CPU KNN via ``torch.cdist`` + ``torch.topk``.
    Uses CUDA when available, falls back to CPU automatically.
    Controlled by ``PROTEA_KNN_DEVICE`` (``"auto"``, ``"cuda"``,
    ``"cpu"``); chunk size via ``PROTEA_KNN_CHUNK_SIZE`` (default 4096).
    This is the recommended backend for production runs on large corpora
    where torch is already installed (e.g., LAFA embedding pipeline).

sparse
    Inverted-index cosine over learned k-WTA codes. Each code vector is
    sparse (only its top-k dimensions are non-zero, e.g. 128 active of
    2048), so the cosine numerator only needs the dimensions a query and
    a reference share. The backend builds an inverted index over active
    reference dimensions and accumulates dot products by visiting, per
    query, only the references that co-activate at least one of the
    query's active dimensions. Pure NumPy (no scipy / FAISS / torch
    dependency); see ``_search_sparse`` for the complexity analysis.
    Only the ``cosine`` metric is supported for the learned k-WTA codes;
    ``l2`` is rejected. Returns the identical ``(accession, distance)``
    contract as the dense backends with ``distance = 1 - cosine_similarity``.

Metric convention
-----------------
All backends return **distances** (lower = more similar):

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

import logging
import os
from typing import Any

import numpy as np

logger = logging.getLogger(__name__)


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


def _check_alignment(
    hits: list[list[tuple[str, float]]], n_queries: int, backend: str
) -> None:
    """Refuse a result that does not line up with the queries that produced it.

    Position i belongs to query i, and callers zip this against their own
    accession list. A short return therefore does not lose the tail: it shifts
    the alignment and hands each query a later one's neighbours, which reads as
    a plausible answer rather than as a failure. Hence checked, not trusted.
    """
    if len(hits) != n_queries:
        raise RuntimeError(
            f"Backend {backend!r} returned {len(hits):,} result rows for "
            f"{n_queries:,} queries. Results are positional, so a mismatch "
            f"misaligns every query after the gap; refusing to return."
        )


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
        ``"numpy"`` (exact brute-force), ``"faiss"``, ``"torch"`` (chunked
        GPU/CPU KNN via ``torch.cdist + topk``; ``PROTEA_KNN_DEVICE`` default
        ``"auto"``, ``PROTEA_KNN_CHUNK_SIZE`` default 4096), or ``"sparse"``
        (inverted-index cosine over learned k-WTA codes; cosine only).
    metric:
        ``"cosine"`` or ``"l2"``.
    pre_normalized:
        When ``True`` and ``metric == "cosine"``, the caller guarantees that
        ``ref_embeddings`` rows are already L2-normalised, so the backend
        skips the per-call normalisation. No-op for ``l2`` or ``torch``.
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
        hits = _search_faiss(
            query_embeddings, ref_embeddings, ref_accessions, k,
            distance_threshold=distance_threshold, metric=metric,
            index_type=faiss_index_type, nlist=faiss_nlist, nprobe=faiss_nprobe,
            hnsw_m=faiss_hnsw_m, hnsw_ef_search=faiss_hnsw_ef_search,
        )
    elif backend == "numpy":
        hits = _search_numpy(
            query_embeddings, ref_embeddings, ref_accessions, k,
            distance_threshold=distance_threshold, metric=metric, pre_normalized=pre_normalized,
        )
    elif backend == "torch":
        hits = _search_torch(
            query_embeddings, ref_embeddings, ref_accessions, k,
            distance_threshold=distance_threshold, metric=metric,
        )
    elif backend == "sparse":
        if metric != "cosine":
            raise ValueError(
                f"Sparse backend supports only metric='cosine', got {metric!r}. "
                f"k-WTA codes are scored by sparse cosine overlap."
            )
        hits = _search_sparse(
            query_embeddings, ref_embeddings, ref_accessions, k,
            distance_threshold=distance_threshold, pre_normalized=pre_normalized,
        )
    else:
        raise ValueError(
            f"Unknown search backend: {backend!r}. Choose 'numpy', 'faiss', 'torch', or 'sparse'."
        )

    _check_alignment(hits, query_embeddings.shape[0], backend)
    return hits


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


def _torch_knn_chunk_size() -> int:
    """Number of query rows to process per GPU kernel launch.

    4096 keeps peak VRAM under ~1 GB for D=1024 and N=500K refs
    (4096 x 500K x 4B = ~8 GB — so scale down if OOM occurs; the
    retry loop in ``_search_torch`` halves this automatically).
    Override via ``PROTEA_KNN_CHUNK_SIZE``.
    """
    raw = os.environ.get("PROTEA_KNN_CHUNK_SIZE")
    if raw is None:
        return 4096
    return int(raw)


def _torch_device() -> Any:
    """Resolve the compute device for the torch KNN backend.

    Priority:
    1. ``PROTEA_KNN_DEVICE`` env var (``"auto"``, ``"cuda"``, ``"cpu"``).
    2. Auto-detect: CUDA if available, else CPU.
    """
    import torch  # local import: torch is an optional dependency

    requested = os.environ.get("PROTEA_KNN_DEVICE", "auto").lower()
    if requested == "auto":
        if torch.cuda.is_available():
            return torch.device("cuda")
        return torch.device("cpu")
    return torch.device(requested)


def _torch_target_device(n_refs: int, dim: int) -> Any:
    """The device the corpus will actually live on.

    A corpus that fills most of free VRAM leaves no room for a query chunk,
    so the search falls back to CPU rather than failing on the first matmul.
    """
    import torch  # local import

    device = _torch_device()
    if device.type != "cuda":
        return device
    corpus_bytes = n_refs * dim * 4  # float32
    free_bytes, _ = torch.cuda.mem_get_info(device)
    if corpus_bytes > free_bytes * 0.7:
        logger.warning(
            "Corpus size (%.1f GB) exceeds 70%% of free VRAM (%.1f GB). "
            "Falling back to CPU. Implement corpus-chunked variant for "
            "corpora larger than available VRAM.",
            corpus_bytes / 1e9,
            free_bytes / 1e9,
        )
        return torch.device("cpu")
    return device


def _chunk_topk(
    Q_chunk_np: np.ndarray,
    R_t: Any,
    *,
    metric: str,
    k_eff: int,
    device: Any,
) -> tuple[np.ndarray, np.ndarray]:
    """Score one query chunk against the whole corpus, keep the k nearest.

    Returns ``(distances, indices)`` on the host, both ``(chunk_rows, k_eff)``.
    Separated out because this is the only part that can run out of memory, so
    the caller can retry it with fewer rows and decide nothing else again.
    """
    import torch  # local import

    Q_t = torch.from_numpy(Q_chunk_np).to(device)
    if metric == "cosine":
        Q_t = torch.nn.functional.normalize(Q_t, p=2, dim=1)
        # distance = 1 - cosine similarity
        dist = 1.0 - (Q_t @ R_t.T)
    else:  # l2 -- squared Euclidean, consistent with numpy backend
        # Expanded form rather than torch.cdist: it matches the numpy backend
        # and avoids the sqrt. The clamp absorbs the small negative values the
        # matmul shortcut can produce near zero distance.
        Q2 = (Q_t ** 2).sum(dim=1, keepdim=True)  # (C, 1)
        R2_t = (R_t ** 2).sum(dim=1)              # (N,)
        dist = torch.clamp(Q2 + R2_t - 2.0 * (Q_t @ R_t.T), min=0.0)
    top_dist, top_idx = torch.topk(dist, k_eff, dim=1, largest=False, sorted=True)
    return top_dist.cpu().numpy(), top_idx.cpu().numpy()


def _hits_from_topk(
    top_dist: np.ndarray,
    top_idx: np.ndarray,
    ref_accessions: list[str],
    *,
    k_eff: int,
    distance_threshold: float | None,
) -> list[list[tuple[str, float]]]:
    """Name the neighbours of each row in one chunk's topk arrays."""
    rows: list[list[tuple[str, float]]] = []
    for row_i in range(top_dist.shape[0]):
        hits: list[tuple[str, float]] = []
        for col_i in range(k_eff):
            dist_val = float(top_dist[row_i, col_i])
            if distance_threshold is not None and dist_val > distance_threshold:
                break
            hits.append((ref_accessions[int(top_idx[row_i, col_i])], dist_val))
        rows.append(hits)
    return rows


def _search_torch(
    Q: np.ndarray,
    R: np.ndarray,
    ref_accessions: list[str],
    k: int,
    *,
    distance_threshold: float | None,
    metric: str,
) -> list[list[tuple[str, float]]]:
    """Chunked GPU (or CPU) KNN via matmul + ``torch.topk``.

    Design
    ------
    - Corpus ``R`` is loaded onto the device once per call. If it would fill
      most of free VRAM the search runs on CPU instead.
    - Queries are processed in chunks of ``PROTEA_KNN_CHUNK_SIZE``
      (default 4096). Each chunk is moved to the device, distances are
      computed against the full corpus, ``topk`` extracts the k nearest,
      and results are copied back to CPU before the next chunk.
    - On out of memory the chunk is halved and the SAME rows are retried, so
      the queries that did not fit are still answered. The smaller size is
      kept for the rest of the call. A single row that does not fit raises.
    - ``torch.float32`` is used throughout (fp16/bf16 sacrifices
      retrieval precision; not worth it here).

    Metric
    ------
    cosine
        Both tensors are L2-normalised on device. Distance = 1 - dot
        product. ``torch.cdist`` is avoided for cosine (the normalised
        dot-product path is numerically cleaner).
    l2
        Squared Euclidean by the expanded form, clamped at zero, matching
        the numpy backend.
    """
    if metric not in {"cosine", "l2"}:
        raise ValueError(f"Unknown metric: {metric!r}. Choose 'cosine' or 'l2'.")

    import torch  # local import

    n_refs, dim = R.shape
    n_queries = Q.shape[0]
    k_eff = min(k, n_refs)
    device = _torch_target_device(n_refs, dim)

    with torch.no_grad():
        R_t = torch.from_numpy(np.ascontiguousarray(R, dtype=np.float32)).to(device)
        if metric == "cosine":
            R_t = torch.nn.functional.normalize(R_t, p=2, dim=1)

        results: list[list[tuple[str, float]]] = []

        # A cursor, not a fixed stride: on OOM the chunk is halved and the SAME
        # cursor is retried, so the rows that did not fit are still processed
        # and the cursor advances only by what a pass actually consumed. A
        # stride loop would step over the tail of a shrunk chunk, dropping
        # those queries and silently shifting every later query's neighbours
        # onto the wrong protein. The smaller size is kept for the whole call:
        # a corpus that overflowed once will overflow again.
        rows_per_chunk = _torch_knn_chunk_size()
        start = 0
        while start < n_queries:
            take = min(rows_per_chunk, n_queries - start)
            try:
                top_dist, top_idx = _chunk_topk(
                    np.ascontiguousarray(Q[start : start + take], dtype=np.float32),
                    R_t,
                    metric=metric,
                    k_eff=k_eff,
                    device=device,
                )
            except RuntimeError as exc:
                if "out of memory" not in str(exc).lower() or take == 1:
                    # A single query against the corpus that does not fit
                    # cannot be helped by halving, and returning short results
                    # would be worse than failing, so the search gives up here.
                    raise
                rows_per_chunk = max(1, take // 2)
                logger.warning(
                    "CUDA OOM on chunk of %d rows at query %d; retrying the "
                    "same rows with a chunk of %d.",
                    take,
                    start,
                    rows_per_chunk,
                )
                continue

            rows = _hits_from_topk(
                top_dist, top_idx, ref_accessions,
                k_eff=k_eff, distance_threshold=distance_threshold,
            )
            results.extend(rows)
            # Advance by the rows that produced output, not by the requested
            # stride: tying the cursor to the output is what keeps a chunk that
            # answered fewer rows than asked from skipping the difference.
            start += len(rows)

    _release_corpus_vram(R_t, device)
    return results


def _release_corpus_vram(R_t: Any, device: Any) -> None:
    """Free the corpus tensor and empty the CUDA cache when the search ran on GPU.

    Without this, looping ``_search_torch`` across the three GO aspects pins
    about 10 GB on a 12 GB card, and the corpus-fits-in-VRAM check inside
    ``_torch_target_device`` then flips the device back to CPU for the rest of
    the run. The run completes, slower, and says nothing: the only symptom is
    that the second and third aspects did not use the card the first one did.

    Discovered 2026-05-27 on an RTX 3060 with ankh-large, 1536 dimensions over
    527k proteins, at 3.2 GB per aspect copy.

    The campaign no longer reaches this path, because PROTEA pins
    ``PROTEA_KNN_DEVICE`` to cpu rather than letting it resolve to "auto". It is
    still worth carrying, for the reason the pin exists: "auto" resolves to CUDA
    whenever a card is visible, so any caller reaching this library without that
    pin gets the leak.
    """
    if device.type == "cuda":
        import torch as _torch

        del R_t
        _torch.cuda.empty_cache()


def _l2_normalize_rows(M: np.ndarray) -> np.ndarray:
    """Return ``M`` with each row scaled to unit L2 norm (zero rows stay zero)."""
    norms = np.linalg.norm(M, axis=1, keepdims=True)
    return np.asarray(M / (norms + 1e-9), dtype=np.float32)


class _InvertedIndex:
    """Flat CSR-style inverted index over the active dimensions of a code bank.

    For each feature dimension ``d`` the references that fire in ``d`` (and
    their values) live in a contiguous slice
    ``rows[indptr[d] : indptr[d + 1]]`` of the flat ``rows`` / ``vals``
    arrays. Storing the postings flat (rather than one array per dimension)
    lets a query gather the postings of all its active dimensions in a
    single fancy-index + ``np.add.at`` scatter, with no per-dimension Python
    loop.

    Attributes
    ----------
    rows:
        Flat int64 array of reference row indices, grouped by dimension.
    vals:
        Flat float32 array of the matching reference values.
    indptr:
        Length ``dim + 1`` int64 array of slice boundaries per dimension.
    n_refs:
        Number of reference rows the index was built over.
    """

    __slots__ = ("indptr", "n_refs", "rows", "vals")

    def __init__(
        self,
        rows: np.ndarray,
        vals: np.ndarray,
        indptr: np.ndarray,
        n_refs: int,
    ) -> None:
        self.rows = rows
        self.vals = vals
        self.indptr = indptr
        self.n_refs = n_refs


def _concat_ranges(starts: np.ndarray, ends: np.ndarray, total: int) -> np.ndarray:
    """Vectorised concatenation of the integer ranges ``[starts[i], ends[i])``.

    Equivalent to ``np.concatenate([np.arange(s, e) for s, e in zip(starts, ends)])``
    but without the Python loop. Used to gather the flat posting positions of
    a query's active dimensions in one fancy-index.

    Builds the output as all-ones, then at the first position of each segment
    overwrites the value with the jump from the previous segment's last index
    to this segment's start, so a single cumulative sum lands on every range
    element. Empty segments contribute nothing and are skipped.
    """
    counts = ends - starts
    out = np.ones(total, dtype=np.int64)
    # Output offset where each segment begins; drop empty segments so the
    # reset is written at a real (non-duplicated) position.
    seg_offsets = np.concatenate(([0], np.cumsum(counts)[:-1]))
    nonempty = counts > 0
    seg_offsets_ne = seg_offsets[nonempty]
    starts_ne = starts[nonempty]
    # First non-empty segment starts at its own start; each later non-empty
    # segment jumps from the running value (its predecessor's last index) to
    # its start. The running value just before offset o equals the previous
    # non-empty segment's last index, so the stored delta is start - prev_last.
    resets = np.empty(starts_ne.shape[0], dtype=np.int64)
    resets[0] = starts_ne[0]
    prev_last = (ends[nonempty] - 1)[:-1]
    resets[1:] = starts_ne[1:] - prev_last
    out[seg_offsets_ne] = resets
    return np.asarray(np.cumsum(out), dtype=np.int64)


def _build_inverted_index(R: np.ndarray) -> _InvertedIndex:
    """Build a flat CSR-style inverted index over the active dims of ``R``.

    Iterates the non-zero structure once (O(nnz(R)); for k-WTA codes
    nnz(R) = ``n_refs * active_k``, far below the dense ``n_refs * dim``)
    and groups it by feature dimension.
    """
    dim = R.shape[1]
    nz_rows, nz_cols = np.nonzero(R)
    nz_vals = R[nz_rows, nz_cols].astype(np.float32, copy=False)
    # Group by column: a stable argsort on the column index gives contiguous
    # per-dimension runs; np.searchsorted then records each run boundary.
    order = np.argsort(nz_cols, kind="stable")
    cols_sorted = nz_cols[order]
    rows_sorted = nz_rows[order].astype(np.int64, copy=False)
    vals_sorted = nz_vals[order]
    indptr = np.searchsorted(cols_sorted, np.arange(dim + 1)).astype(np.int64, copy=False)
    return _InvertedIndex(rows_sorted, vals_sorted, indptr, R.shape[0])


def _sparse_query_scores(
    q: np.ndarray, index: _InvertedIndex, n_refs: int
) -> np.ndarray:
    """Accumulate sparse-cosine similarity of one query against all refs.

    Gathers the postings (ref rows + values) of every active query dimension
    in one fancy-index, weights each posting by the query value of its
    dimension, then scatter-adds into an ``n_refs`` score buffer with a
    single C-level ``np.bincount`` (no per-dimension Python loop). Only the
    references that co-activate at least one of the query's active dimensions
    are visited.
    """
    active_dims = np.nonzero(q)[0]
    starts = index.indptr[active_dims]
    ends = index.indptr[active_dims + 1]
    counts = ends - starts
    total = int(counts.sum())
    if total == 0:
        return np.zeros(n_refs, dtype=np.float32)
    seg = _concat_ranges(starts, ends, total)
    # Per-posting weight = ref value * the query's value in that dim.
    contrib = index.vals[seg] * np.repeat(q[active_dims], counts)
    return np.bincount(
        index.rows[seg], weights=contrib, minlength=n_refs
    ).astype(np.float32, copy=False)


def _topk_hits(
    scores: np.ndarray,
    ref_accessions: list[str],
    k_eff: int,
    distance_threshold: float | None,
) -> list[tuple[str, float]]:
    """Select the ``k_eff`` smallest cosine distances (largest similarities).

    Cosine distance = 1 - similarity. Uses argpartition on the negated scores
    when ``k_eff`` is below ``n_refs``; a stable argsort breaks ties.
    """
    n_refs = scores.shape[0]
    if k_eff < n_refs:
        part = np.argpartition(-scores, k_eff - 1)[:k_eff]
        top = part[np.argsort(-scores[part], kind="stable")]
    else:
        top = np.argsort(-scores, kind="stable")[:k_eff]
    hits: list[tuple[str, float]] = []
    for idx in top:
        dist = float(1.0 - scores[idx])
        if distance_threshold is not None and dist > distance_threshold:
            break
        hits.append((ref_accessions[int(idx)], dist))
    return hits


def _search_sparse(
    Q: np.ndarray,
    R: np.ndarray,
    ref_accessions: list[str],
    k: int,
    *,
    distance_threshold: float | None,
    pre_normalized: bool = False,
) -> list[list[tuple[str, float]]]:
    """Inverted-index cosine over sparse learned k-WTA codes.

    Each row of ``Q`` and ``R`` is a k-WTA code: only its top-``active_k``
    dimensions are non-zero (e.g. 128 of 2048). Cosine similarity between two
    unit-normalised codes is their dot product, which only touches the
    dimensions they share. Rows are L2-normalised once (skipped when
    ``pre_normalized``) so the numerator equals the raw dot product; a flat
    CSR-style inverted index (``_build_inverted_index``) maps each active
    reference dimension to its postings; per query the score accumulation
    (``_sparse_query_scores``) and top-k selection (``_topk_hits``) visit
    only the co-activating references.

    The metric is validated by the caller (``search_knn``); only ``cosine``
    is supported. The result is the **exact** sparse cosine, so the top-``k``
    ranking matches a dense cosine over the same codes up to tie-breaking
    among equal distances. The value of this backend is exactness and a
    memory footprint of only nnz (not the dense ``n_refs * dim`` bank); at
    the current 6-12% density a single BLAS ``gemm`` still wins wall-clock,
    so prefer the dense ``numpy``/``faiss`` backends for raw throughput.
    """
    R_ready = R if pre_normalized else _l2_normalize_rows(R)
    Q_ready = _l2_normalize_rows(Q)
    n_refs = R_ready.shape[0]
    k_eff = min(k, n_refs)
    index = _build_inverted_index(R_ready)

    results: list[list[tuple[str, float]]] = []
    for q_row in range(Q_ready.shape[0]):
        scores = _sparse_query_scores(Q_ready[q_row], index, n_refs)
        results.append(_topk_hits(scores, ref_accessions, k_eff, distance_threshold))
    return results


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

    # `index` is a faiss.Index subclass and the concrete subtype depends on
    # `index_type`. Annotate as Any so mypy strict accepts reassigning to
    # IVF / HNSW variants below without union-narrowing complaints.
    index: Any
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


__all__ = ["_compute_distance_matrix", "search_knn"]
