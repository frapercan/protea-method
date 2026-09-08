"""The distance, computed so that its ordering does not depend on the machine.

Its own module because the concern is one thing and ``knn_search`` is already at
its size budget: what belongs here is everything that makes the k-th neighbour a
function of the query and the bank rather than of the CPU that happened to
consume the batch.

The measurement that forced it, 2026-09-08: the same axis C baseline predicted
twice differed by 82 and -73 rows out of ten million, because every job splits
its batches over two machines and OpenBLAS chooses its thread partition from the
detected CPU. 838,885 rows carried the same donor at a different distance, every
delta an exact multiple of 2^-24, up to 13 ulps.
"""

from __future__ import annotations

import warnings

import numpy as np

__all__ = [
    "ORDER_INVARIANT_BACKENDS",
    "cosine_distance_f64",
    "l2_distance_f64",
    "warn_if_order_dependent",
]

#: Reference vectors promoted to float64 per block. The block bounds memory:
#: a 528k-vector bank at 1024 dims is 2.2 GB in float32 and 4.3 GB in float64,
#: so promoting the whole bank at once is not an option. Blocking over
#: REFERENCES splits no accumulation -- each distance is one dot product over
#: the dim axis -- so the result is exactly the unblocked float64 result.
_REF_BLOCK = 50_000


def cosine_distance_f64(Q_n: np.ndarray, R_ready: np.ndarray) -> np.ndarray:
    """``1 - Q@R.T`` accumulated in float64, returned in float32.

    WHY. In float32 the answer depends on the ORDER OpenBLAS reduces in, and
    that order is chosen at runtime from the CPU and the thread count. Measured
    on 2026-09-08 across two machines: the same bank and the same code gave
    four different results for four thread counts, and where the k-th distance
    ties, a one-ulp change swaps which donor is admitted. The observed spread
    reached 13 float32 ulps, 7.75e-7.

    Rounding the distance to a grid was tried and rejected: it is a probability
    reduction dressed as an invariant, its bucket edges are arbitrary, and its
    residual moves whenever the noise moves. Accumulating in float64 leaves a
    spread near 1e-16, far below the ~6e-8 float32 resolution, so the downcast
    erases it ALWAYS rather than usually.

    Measured cost: x2.4 to x3.0 on the matmul, and it does NOT recover with
    more threads -- 8 to 12 threads moved float32 from 1.44s to 1.33s and
    float64 from 4.00s to 3.96s. The penalty is bandwidth-bound and structural;
    the lever is the block size or the precision, never the parallelism.
    """
    out = np.empty((Q_n.shape[0], R_ready.shape[0]), dtype=np.float32)
    Q64 = Q_n.astype(np.float64)
    for s in range(0, R_ready.shape[0], _REF_BLOCK):
        blk = R_ready[s : s + _REF_BLOCK].astype(np.float64)
        out[:, s : s + blk.shape[0]] = (1.0 - (Q64 @ blk.T)).astype(np.float32)
    return out


def l2_distance_f64(
    Q_chunk: np.ndarray, R_ready: np.ndarray, R2: np.ndarray | None
) -> np.ndarray:
    """Squared euclidean via the expanded form, accumulated in float64.

    Same argument as the cosine case; kept separate because the expansion has
    its own cancellation and clamping near zero.

    ``R2`` is optional in the type because the caller computes it exactly when
    the metric is l2 and leaves it None for cosine. Reaching here without it is
    a caller bug, not a runtime condition, so it is refused by name rather than
    recomputed silently -- recomputing would hide the mistake and pay for it
    once per chunk.
    """
    if R2 is None:
        raise ValueError(
            "l2_distance_f64 needs the precomputed ||R||^2; it is built once per "
            "call alongside metric='l2' and must be passed through"
        )
    out = np.empty((Q_chunk.shape[0], R_ready.shape[0]), dtype=np.float32)
    Q64 = Q_chunk.astype(np.float64)
    Q2 = (Q64**2).sum(axis=1, keepdims=True)
    for s in range(0, R_ready.shape[0], _REF_BLOCK):
        blk = R_ready[s : s + _REF_BLOCK].astype(np.float64)
        d = Q2 + R2[s : s + blk.shape[0]].astype(np.float64) - 2.0 * (Q64 @ blk.T)
        out[:, s : s + blk.shape[0]] = np.maximum(0.0, d).astype(np.float32)
    return out


#: Backends whose k-th neighbour does not depend on the order the machine
#: reduces in. Only ``numpy`` accumulates in float64 (see
#: ``_cosine_distance_f64``); ``torch`` reduces on the device with its own
#: partitioning, and ``faiss`` searches a float32 index this package does not
#: build. Naming the covered set rather than fixing one path and staying quiet
#: is deliberate: every prediction set stored to date used ``numpy``, but
#: ``search_backend`` DEFAULTS to ``faiss`` in five export and training
#: payloads, so the uncovered path is the one a dataset export or a reranker
#: run falls into without anybody choosing it -- and the reranker is precisely
#: where donor identity becomes a feature and the ulp noise would be read as
#: signal.
ORDER_INVARIANT_BACKENDS = frozenset({"numpy"})

#: Warned once per process; a per-call warning would drown a batch log.
_WARNED_NON_INVARIANT: set[str] = set()


def warn_if_order_dependent(backend: str) -> None:
    """Say so when the selection about to run is not reduction-order invariant.

    A warning and not a refusal, because ``faiss`` and ``torch`` are legitimate
    choices for work that does not compare runs. What is not legitimate is
    NOT KNOWING: the defect this guards was invisible for weeks because a
    prediction set records no such thing, and was found only when the same cell
    was recomputed for an unrelated reason.
    """
    if backend in ORDER_INVARIANT_BACKENDS or backend in _WARNED_NON_INVARIANT:
        return
    _WARNED_NON_INVARIANT.add(backend)
    warnings.warn(
        f"search backend {backend!r} selects the k-th neighbour with float32 "
        "arithmetic whose reduction order depends on the CPU and the thread "
        "count, so two machines can disagree about which donor is admitted at "
        f"a tie. Invariant backends: {sorted(ORDER_INVARIANT_BACKENDS)}.",
        RuntimeWarning,
        stacklevel=3,
    )
