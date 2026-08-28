"""The chunked top-k loop, lifted out of ``_search_torch``.

Its own module for two reasons and the second is the load-bearing one.

``knn_search`` sits at the file-size ceiling, so this could not stay there. That
is the guard doing its job rather than an inconvenience to route around.

And the caller has to be able to free the corpus tensor. While the loop lived
inside ``_search_torch`` the only way to keep that function under its line budget
was to hand the tensor to a helper that deleted its own reference, which frees
nothing: the caller's binding outlives the call, the refcount never reaches zero,
and ``torch.cuda.empty_cache()`` then drains an allocator that is still holding
the block. Measured with a weakref probe on that exact call shape: the tensor was
still alive when the drain ran. Moving the loop out instead of the release means
``R_t`` is bound in one place, and dropping it there is the last reference.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any

import numpy as np

logger = logging.getLogger(__name__)

__all__ = ["TorchSearch", "chunked_topk"]


@dataclass(frozen=True)
class TorchSearch:
    """The settings one torch search runs under, in one object.

    Grouped rather than passed loose because eight positional arguments is the
    guard's way of saying the same thing twice: they travel together, and a
    signature that lists them one by one lets a caller forget one without
    anything noticing.
    """

    metric: str
    k_eff: int
    device: Any
    distance_threshold: float | None
    chunk_rows: int


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


def chunked_topk(
    Q: np.ndarray,
    R_t: Any,
    ref_accessions: list[str],
    cfg: TorchSearch,
) -> list[list[tuple[str, float]]]:
    """Search every query in chunks, halving on OOM without dropping rows.

    A cursor, not a fixed stride: on OOM the chunk is halved and the SAME cursor
    is retried, so the rows that did not fit are still processed and the cursor
    advances only by what a pass actually consumed. A stride loop would step over
    the tail of a shrunk chunk, dropping those queries and silently shifting
    every later query's neighbours onto the wrong protein.

    The smaller size is kept for the rest of the call: a corpus that overflowed
    once will overflow again.
    """
    n_queries = Q.shape[0]
    results: list[list[tuple[str, float]]] = []
    rows_per_chunk = cfg.chunk_rows
    start = 0
    while start < n_queries:
        take = min(rows_per_chunk, n_queries - start)
        try:
            top_dist, top_idx = _chunk_topk(
                np.ascontiguousarray(Q[start : start + take], dtype=np.float32),
                R_t,
                metric=cfg.metric,
                k_eff=cfg.k_eff,
                device=cfg.device,
            )
        except RuntimeError as exc:
            if "out of memory" not in str(exc).lower() or take == 1:
                # A single query against a corpus that does not fit cannot be
                # helped by halving, and returning short results would be worse
                # than failing, so the search gives up here.
                raise
            rows_per_chunk = max(1, take // 2)
            logger.warning(
                "CUDA OOM on chunk of %d rows at query %d; retrying the same "
                "rows with a chunk of %d.",
                take,
                start,
                rows_per_chunk,
            )
            continue

        rows = _hits_from_topk(
            top_dist,
            top_idx,
            ref_accessions,
            k_eff=cfg.k_eff,
            distance_threshold=cfg.distance_threshold,
        )
        results.extend(rows)
        # Advance by the rows that produced output, not by the requested stride:
        # tying the cursor to the output is what keeps a chunk that answered
        # fewer rows than asked from skipping the difference.
        start += len(rows)
    return results
