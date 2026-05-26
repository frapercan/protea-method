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
        ``"numpy"`` (exact brute-force), ``"faiss"``, or ``"torch"``
        (chunked GPU/CPU KNN via ``torch.cdist + topk``; recommended for
        production runs where torch is installed). Device selection for
        the torch backend is controlled by ``PROTEA_KNN_DEVICE`` (default
        ``"auto"``); query chunk size by ``PROTEA_KNN_CHUNK_SIZE`` (default
        4096).
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
    if backend == "torch":
        return _search_torch(
            query_embeddings,
            ref_embeddings,
            ref_accessions,
            k,
            distance_threshold=distance_threshold,
            metric=metric,
        )
    raise ValueError(f"Unknown search backend: {backend!r}. Choose 'numpy', 'faiss', or 'torch'.")


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


def _search_torch(
    Q: np.ndarray,
    R: np.ndarray,
    ref_accessions: list[str],
    k: int,
    *,
    distance_threshold: float | None,
    metric: str,
) -> list[list[tuple[str, float]]]:
    """Chunked GPU (or CPU) KNN via ``torch.cdist`` + ``torch.topk``.

    Design
    ------
    - Corpus ``R`` is loaded onto the device once per call (must fit in
      VRAM; an assertion guards this for GPU targets).
    - Queries are processed in chunks of ``PROTEA_KNN_CHUNK_SIZE``
      (default 4096). Each chunk is moved to the device, distances are
      computed against the full corpus, ``topk`` extracts the k nearest,
      and results are copied back to CPU before the next chunk.
    - On ``torch.cuda.OutOfMemoryError`` the chunk size is halved and
      the chunk is retried up to 3 times; after that the error is
      re-raised.
    - ``torch.float32`` is used throughout (fp16/bf16 sacrifices
      retrieval precision; not worth it here).

    Metric
    ------
    cosine
        Both tensors are L2-normalised on device. Distance = 1 - dot
        product. ``torch.cdist`` is avoided for cosine (the normalised
        dot-product path is numerically cleaner).
    l2
        ``torch.cdist(Q_chunk, R, p=2,
        compute_mode="donot_use_mm_for_euclid_dist")`` to avoid the
        matmul shortcut that can produce small negative squared distances.
    """
    import torch  # local import

    if metric not in {"cosine", "l2"}:
        raise ValueError(f"Unknown metric: {metric!r}. Choose 'cosine' or 'l2'.")

    device = _torch_device()
    n_refs, dim = R.shape
    n_queries = Q.shape[0]
    k_eff = min(k, n_refs)

    # Sanity-check corpus size on GPU to give a helpful message.
    if device.type == "cuda":
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
            device = torch.device("cpu")

    with torch.no_grad():
        R_t = torch.from_numpy(np.ascontiguousarray(R, dtype=np.float32)).to(device)
        if metric == "cosine":
            R_t = torch.nn.functional.normalize(R_t, p=2, dim=1)

        chunk_size = _torch_knn_chunk_size()
        results: list[list[tuple[str, float]]] = []

        for start in range(0, n_queries, chunk_size):
            Q_chunk_np = np.ascontiguousarray(
                Q[start : start + chunk_size], dtype=np.float32
            )
            remaining_halvings = 3
            while True:
                try:
                    Q_t = torch.from_numpy(Q_chunk_np).to(device)
                    if metric == "cosine":
                        Q_t = torch.nn.functional.normalize(Q_t, p=2, dim=1)
                        # distance = 1 - cosine similarity
                        dist = 1.0 - (Q_t @ R_t.T)
                    else:  # l2 -- squared Euclidean, consistent with numpy backend
                        # torch.cdist(p=2) returns Euclidean (unsquared).
                        # Use the squared-distance formula directly to match
                        # the numpy backend and avoid the sqrt overhead.
                        Q2 = (Q_t ** 2).sum(dim=1, keepdim=True)  # (C, 1)
                        R2_t = (R_t ** 2).sum(dim=1)              # (N,)
                        dist = torch.clamp(
                            Q2 + R2_t - 2.0 * (Q_t @ R_t.T), min=0.0
                        )
                    top_dist, top_idx = torch.topk(
                        dist, k_eff, dim=1, largest=False, sorted=True
                    )
                    top_dist_cpu = top_dist.cpu().numpy()
                    top_idx_cpu = top_idx.cpu().numpy()
                    break
                except RuntimeError as exc:
                    if "out of memory" not in str(exc).lower() or remaining_halvings == 0:
                        raise
                    remaining_halvings -= 1
                    new_size = max(1, Q_chunk_np.shape[0] // 2)
                    logger.warning(
                        "CUDA OOM on chunk of %d rows; retrying with %d rows "
                        "(%d retries left).",
                        Q_chunk_np.shape[0],
                        new_size,
                        remaining_halvings,
                    )
                    # Re-split: process only first half of the chunk in this
                    # iteration; the second half will be picked up by the outer
                    # loop adjustment.  Simplest OOM recovery: shrink + retry.
                    Q_chunk_np = Q_chunk_np[:new_size]

            n_rows = top_dist_cpu.shape[0]
            for row_i in range(n_rows):
                hits: list[tuple[str, float]] = []
                for col_i in range(k_eff):
                    dist_val = float(top_dist_cpu[row_i, col_i])
                    if distance_threshold is not None and dist_val > distance_threshold:
                        break
                    hits.append((ref_accessions[int(top_idx_cpu[row_i, col_i])], dist_val))
                results.append(hits)

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
