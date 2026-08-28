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
from typing import Any

import numpy as np

logger = logging.getLogger(__name__)

__all__ = ["chunked_topk"]


def chunked_topk(
    Q: np.ndarray,
    R_t: Any,
    ref_accessions: list[str],
    *,
    metric: str,
    k_eff: int,
    device: Any,
    distance_threshold: float | None,
    chunk_rows: int,
    chunk_topk: Any,
    hits_from_topk: Any,
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
    rows_per_chunk = chunk_rows
    start = 0
    while start < n_queries:
        take = min(rows_per_chunk, n_queries - start)
        try:
            top_dist, top_idx = chunk_topk(
                np.ascontiguousarray(Q[start : start + take], dtype=np.float32),
                R_t,
                metric=metric,
                k_eff=k_eff,
                device=device,
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

        rows = hits_from_topk(
            top_dist,
            top_idx,
            ref_accessions,
            k_eff=k_eff,
            distance_threshold=distance_threshold,
        )
        results.extend(rows)
        # Advance by the rows that produced output, not by the requested stride:
        # tying the cursor to the output is what keeps a chunk that answered
        # fewer rows than asked from skipping the difference.
        start += len(rows)
    return results
