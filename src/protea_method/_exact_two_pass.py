"""The exact search in two passes: float32 shortlists, float64 decides.

WHY. The exact backend accumulates every distance in float64, so that the k-th
neighbour does not depend on the order a machine reduces in
(:mod:`protea_method._order_invariance`). That is right, and it was slow. On the
development store, on 2026-11-21, a batch of 1,024 queries against the 1.45
million references of one cut spent 154 of its 171 seconds searching. Most of
that went to promoting reference blocks to float64, multiplying in float64, and
selecting among every reference of every row.

THE IDEA. A float32 product is fast, on the CPU and much more on a GPU, and its
error is bounded. For a dot product of ``d`` terms, computed in any order and
with any blocking, the error is at most ``gamma_d * sum |q_i r_i|``, with
``gamma_d = d u / (1 - d u)`` and ``u = 2**-24`` (Higham, *Accuracy and Stability
of Numerical Algorithms*, 2nd ed., section 3.1). So every float32 distance lies
within a known ``eps`` of the distance the float64 path returns
(:func:`first_pass_margin`). Two facts follow:

1. The k-th smallest float32 distance ``t`` bounds the exact k-th from above.
   The k references at or below ``t`` in float32 are all at or below
   ``t + eps`` exactly.
2. Any reference the exact path can keep, one whose exact distance is at most
   the exact k-th, is therefore at most ``t + 2 eps`` in float32.

The first pass keeps, for each query, every reference within ``t + 2 eps``. The
second pass computes the float64 distance of those alone, with the function the
exact path has always used, and selects with the same rule: by distance, then by
position in the bank. So the neighbours, their order and their distances are the
exact path's.

The proof assumes nothing about the device or the reduction order of the first
pass, which is why that pass may run on a GPU. Whatever order a card reduces in,
it moves a float32 distance by less than ``eps``, and the card decides nothing.
It also cannot drop a query: a pass that runs out of memory retries with fewer
rows, and a row whose shortlist cannot be trusted (fewer than k references, or a
distance that is not a number) is decided over the whole bank, as before.

The float64 distances are computed by a smaller product than the full one, so
their summation order can differ in the last bit of a float64. The downcast to
float32 erases that difference except when a value lies within about 1e-16 of a
float32 rounding boundary. That is the same caveat under which the full float64
path agrees with itself across machines (``_order_invariance``), and the check
on the development store found no such case.

THE DEVICE. ``PROTEA_METHOD_EXACT_FIRST_PASS_DEVICE`` is ``auto`` by default (a
CUDA card when torch sees one, else the CPU), ``cuda`` or ``cpu``. The answer is
the same on all three; only the time differs. On a card shared with another
process, the pass takes it only when it needs less than half the free memory,
and it releases what it took before returning.
"""

from __future__ import annotations

import contextlib
import logging
import os
from collections.abc import Iterator
from dataclasses import dataclass
from typing import Any

import numpy as np

from protea_method._order_invariance import cosine_distance_f64, l2_distance_f64

logger = logging.getLogger(__name__)

__all__ = [
    "FIRST_PASS_DEVICE_ENV",
    "dot_error_bound",
    "exact_top_k",
    "first_pass_margin",
    "top_k_by_distance_then_position",
    "widened_norm",
]

#: The float32 unit roundoff.
_U32 = 2.0**-24

#: References promoted to float32 per block on the CPU, the same block the float64
#: path uses: blocking over references splits no dot product.
_REF_BLOCK = 50_000

#: A float32 norm times this bounds the norm from above: a thousandth is far more
#: than the relative rounding of a norm over a few thousand terms.
_NORM_WIDENING = 1.001

#: Names the device of the first pass: ``auto``, ``cuda`` or ``cpu``.
FIRST_PASS_DEVICE_ENV = "PROTEA_METHOD_EXACT_FIRST_PASS_DEVICE"

#: Query rows per GPU launch before any out-of-memory halving.
_GPU_ROWS = 1024

#: The share of the card's free memory the first pass may take. The card can be
#: shared with another process loading a model, and the pass gives way rather
#: than make that process the one that runs out: above this share it runs on the
#: CPU, with the same result.
_GPU_SHARE = 0.5

#: Queries per float64 product in the second pass: tall enough that the product
#: is a matrix product like the single pass's, never a matrix-vector one.
_GROUP_ROWS = 64


def dot_error_bound(dim: int) -> float:
    """``gamma_dim`` in float32: the relative error bound of a ``dim``-term dot product."""
    nu = dim * _U32
    if nu >= 0.5:
        raise ValueError(f"no useful float32 error bound for {dim}-dimensional vectors")
    return nu / (1.0 - nu)


def first_pass_margin(Q: np.ndarray, r_max: float, metric: str) -> np.ndarray:
    """Per query, a bound on how far a float32 distance can lie from the exact one.

    cosine, ``1 - q.r``: the product's error, ``gamma |q| |r|``, plus one rounding
    of the subtraction in each pass and the downcast, all within ``4u``.

    l2, ``|q|^2 + |r|^2 - 2 q.r``: two products and the sums, within
    ``(2 gamma + 6u)(|q| + |r|)^2``. The clamp at zero moves no distance further.

    ``r_max`` is the largest reference norm (:func:`widened_norm`); the query
    norms are computed here in float32 and widened the same way.
    """
    gamma = dot_error_bound(Q.shape[1])
    norms = np.linalg.norm(Q.astype(np.float32, copy=False), axis=1)
    q: np.ndarray = np.asarray(norms, dtype=np.float64) * _NORM_WIDENING
    if metric == "cosine":
        return np.asarray(gamma * q * r_max + 4.0 * _U32)
    return np.asarray((2.0 * gamma + 6.0 * _U32) * (q + r_max) ** 2)


def widened_norm(norm: float) -> float:
    """A float32 norm widened by a thousandth, which more than covers its own rounding."""
    return norm * _NORM_WIDENING


def top_k_by_distance_then_position(dist: np.ndarray, k: int) -> list[np.ndarray]:
    """The k nearest columns of each row, equal distances ordered by column.

    WHY. ``np.argpartition`` followed by a quicksort ``argsort`` puts EQUAL
    values in an order that comes from the partition's internal pivots, and
    those depend on every other value in the row. Two runs of the same recipe
    on a byte-identical bank (exploration 1, 2026-11-05) gave identical tied
    distances in a different order, because some unrelated column of the row
    moved by one float32 ulp between them, and 9.7 % of the stored rows
    changed: which donor represents a term, its k position, and at the k-th
    slot which donors are admitted at all. Ties are common, not a corner: 91
    per cent of queries have neighbours at distance ~0, themselves or twins
    carrying the same sequence.

    So the k-th smallest distance is found first, every column at or below it
    is a candidate, and the candidates are ordered by distance and then by
    column position with a stable ``lexsort``. Which donors are chosen and in
    what order then depends only on the candidates' own distances and
    positions. The bank is ordered by accession (``COLLATE "C"``) and each
    aspect's view keeps that order, so the position IS the accession: equal
    distances are broken by accession, the same way on every run.
    """
    n_rows, n_cols = dist.shape
    if k >= n_cols:
        kth = dist.max(axis=1)
    else:
        kth = np.partition(dist, k - 1, axis=1)[:, k - 1]
    out: list[np.ndarray] = []
    for r in range(n_rows):
        row = dist[r]
        cand = np.flatnonzero(row <= kth[r])
        # lexsort sorts by the LAST key first: distance, then column position.
        order = np.lexsort((cand, row[cand]))
        out.append(cand[order[:k]])
    return out


def _exact(Q: np.ndarray, R: np.ndarray, R2: np.ndarray | None, metric: str) -> np.ndarray:
    """The float64-accumulated distances, as the exact path has always computed them."""
    if metric == "cosine":
        return cosine_distance_f64(Q, R)
    return l2_distance_f64(Q, R, R2)


def _shortlist_cpu(
    Q: np.ndarray, R: np.ndarray, R2: np.ndarray | None, k: int, metric: str
) -> list[np.ndarray]:
    """The first pass on the CPU: float32 distances, then every column within the margin."""
    Q32 = Q.astype(np.float32, copy=False)
    d32 = np.empty((Q.shape[0], R.shape[0]), dtype=np.float32)
    Q2 = (Q32 * Q32).sum(axis=1, keepdims=True)
    r_max = 0.0
    for start in range(0, R.shape[0], _REF_BLOCK):
        blk = R[start : start + _REF_BLOCK].astype(np.float32, copy=False)
        r_max = max(r_max, float(np.linalg.norm(blk, axis=1).max(initial=0.0)))
        prod = Q32 @ blk.T
        if metric == "cosine":
            d32[:, start : start + blk.shape[0]] = 1.0 - prod
        else:
            if R2 is None:
                raise ValueError("the l2 first pass needs the references' squared norms")
            r2 = R2[start : start + blk.shape[0]].astype(np.float32)
            d32[:, start : start + blk.shape[0]] = np.maximum(0.0, Q2 + r2 - 2.0 * prod)
    kth = np.partition(d32, k - 1, axis=1)[:, k - 1].astype(np.float64)
    limit = kth + 2.0 * first_pass_margin(Q, widened_norm(r_max), metric)
    return [np.flatnonzero(d32[i] <= limit[i]) for i in range(Q.shape[0])]


@contextlib.contextmanager
def _full_float32(torch: Any) -> Iterator[None]:
    """Plain IEEE float32 products for the block, TF32 off, restored afterwards.

    TF32 keeps ten mantissa bits, so its error is not the bound this module proves.
    """
    matmul_tf32 = torch.backends.cuda.matmul.allow_tf32
    precision = torch.get_float32_matmul_precision()
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.set_float32_matmul_precision("highest")
    try:
        yield
    finally:
        torch.backends.cuda.matmul.allow_tf32 = matmul_tf32
        torch.set_float32_matmul_precision(precision)


def _gpu() -> Any | None:
    """torch, when the first pass is to run on a CUDA card, else ``None``."""
    requested = os.environ.get(FIRST_PASS_DEVICE_ENV, "auto").lower()
    if requested == "cpu":
        return None
    try:
        import torch
    except ImportError:
        torch = None
    if torch is not None and torch.cuda.is_available():
        return torch
    if requested == "cuda":
        logger.warning(
            "%s=cuda but no CUDA card is usable here; the first pass runs on the CPU, "
            "with the same result.",
            FIRST_PASS_DEVICE_ENV,
        )
    return None


@dataclass(frozen=True)
class _OnCard:
    """What every launch of the first pass reads: the bank on the card, and the rule."""

    torch: Any
    R_t: Any
    R2_t: Any
    k: int
    metric: str


def _fits_on_card(torch: Any, device: Any, Q: np.ndarray, R: np.ndarray) -> bool:
    """Whether the pass takes no more than its share of the card's free memory."""
    needed = (R.shape[0] * R.shape[1] + 2 * min(_GPU_ROWS, Q.shape[0]) * R.shape[0]) * 4
    free, _total = torch.cuda.mem_get_info(device)
    if needed <= _GPU_SHARE * free:
        return True
    logger.info(
        "the first pass needs %.1f GB and the card has %.1f GB free; it runs on the CPU.",
        needed / 1e9,
        free / 1e9,
    )
    return False


def _launch_all(card: _OnCard, Q: np.ndarray, margin: np.ndarray) -> list[np.ndarray]:
    """Every query row through the card, halving the launch on out-of-memory.

    A cursor, not a stride: on out-of-memory the SAME rows are retried with fewer
    at a time, so none is skipped (see :mod:`protea_method._chunked_topk`). A row
    that does not fit alone raises, and the caller runs the pass on the CPU.
    """
    out: list[np.ndarray] = []
    rows, start = _GPU_ROWS, 0
    while start < Q.shape[0]:
        take = min(rows, Q.shape[0] - start)
        try:
            out.extend(_gpu_rows(card, Q[start : start + take], margin[start : start + take]))
            start += take
        except card.torch.cuda.OutOfMemoryError:
            if take == 1:
                raise
            rows = max(1, take // 2)
            logger.warning("CUDA out of memory at %d query rows; retrying with %d.", take, rows)
    return out


def _shortlist_gpu(
    torch: Any, Q: np.ndarray, R: np.ndarray, R2: np.ndarray | None, k: int, metric: str
) -> list[np.ndarray] | None:
    """The first pass on the card; ``None`` when it should run on the CPU instead.

    The bank's tensors are bound and released here, where their last reference
    lives (see :mod:`protea_method._chunked_topk` on why a helper cannot do it).
    """
    device = torch.device("cuda")
    if not _fits_on_card(torch, device, Q, R):
        return None
    R_t = R2_t = None
    try:
        with torch.no_grad(), _full_float32(torch):
            R_t = torch.from_numpy(np.ascontiguousarray(R)).to(device).float()
            if R2 is not None:
                R2_t = torch.from_numpy(np.ascontiguousarray(R2)).to(device).float()
            margin = first_pass_margin(
                Q, widened_norm(float(torch.linalg.vector_norm(R_t, dim=1).max())), metric
            )
            return _launch_all(_OnCard(torch, R_t, R2_t, k, metric), Q, margin)
    except torch.cuda.OutOfMemoryError:
        logger.warning("one query row does not fit on the card; the first pass runs on the CPU.")
        return None
    finally:
        R_t = R2_t = None
        torch.cuda.empty_cache()


def _gpu_rows(card: _OnCard, Q: np.ndarray, margin: np.ndarray) -> list[np.ndarray]:
    """One launch: float32 distances on the card, then each row's columns within the margin."""
    torch, device = card.torch, card.R_t.device
    Q_t = torch.from_numpy(np.ascontiguousarray(Q, dtype=np.float32)).to(device)
    prod = Q_t @ card.R_t.T
    if card.metric == "cosine":
        d32 = 1.0 - prod
    else:
        q2 = (Q_t * Q_t).sum(dim=1, keepdim=True)
        d32 = torch.clamp(q2 + card.R2_t - 2.0 * prod, min=0.0)
    kth = torch.kthvalue(d32, card.k, dim=1).values.double()
    limit = kth + 2.0 * torch.from_numpy(margin).to(device)
    # A float32 x is at most the limit L whenever it is at most L rounded UP to
    # float32, so comparing against that keeps every column the proof needs.
    limit32 = limit.float()
    up = torch.nextafter(limit32, torch.tensor(float("inf"), device=device))
    limit32 = torch.where(limit32.double() < limit, up, limit32)
    hits = (d32 <= limit32[:, None]).nonzero().cpu().numpy()
    rows = np.split(hits[:, 1], np.searchsorted(hits[:, 0], np.arange(1, Q.shape[0])))
    return [np.asarray(r, dtype=np.int64) for r in rows]


def _first_pass(
    Q: np.ndarray, R: np.ndarray, R2: np.ndarray | None, k: int, metric: str, cpu_rows: int
) -> list[np.ndarray]:
    """Each query's shortlist, on the card when it is usable, else on the CPU."""
    torch = _gpu()
    shortlists = _shortlist_gpu(torch, Q, R, R2, k, metric) if torch is not None else None
    if shortlists is not None:
        return shortlists
    shortlists = []
    for start in range(0, Q.shape[0], max(1, cpu_rows)):
        rows = slice(start, start + max(1, cpu_rows))
        shortlists.extend(_shortlist_cpu(Q[rows], R, R2, k, metric))
    return shortlists


def _decide(
    Q: np.ndarray,
    R: np.ndarray,
    R2: np.ndarray | None,
    k: int,
    metric: str,
    shortlists: list[np.ndarray],
) -> list[tuple[np.ndarray, np.ndarray]]:
    """The second pass: float64 distances of each shortlist, then the k nearest.

    A group of queries at a time against the union of the group's shortlists: a
    matrix product like the single pass's, on fewer columns. Each element of a
    product is summed over the vector's terms in an order set by the depth of the
    product alone, so it rounds to the float32 the single pass gave. A row at a
    time would be a matrix-vector product, summed in another order, and near
    distance zero the float32 keeps the difference; that is why the groups are
    kept at least ``_GROUP_ROWS`` tall.
    """
    decided: dict[int, tuple[np.ndarray, np.ndarray]] = {}
    trusted = [i for i, cols in enumerate(shortlists) if cols.size >= k]
    for g in range(0, len(trusted), _GROUP_ROWS):
        rows = trusted[g : g + _GROUP_ROWS]
        union = np.unique(np.concatenate([shortlists[i] for i in rows]))
        dist_u = _exact(Q[rows], R[union], None if R2 is None else R2[union], metric)
        for n, i in enumerate(rows):
            cols = shortlists[i]
            dist = dist_u[n, np.searchsorted(union, cols)][None, :]
            local = top_k_by_distance_then_position(dist, k)[0]
            decided[i] = (cols[local], dist[0, local])
    for i, cols in enumerate(shortlists):
        if cols.size < k:
            # Cannot be trusted to hold the k nearest (a distance that is not a
            # number does not compare): decided over the whole bank, as before.
            dist = _exact(Q[i : i + 1], R, R2, metric)
            local = top_k_by_distance_then_position(dist, k)[0]
            decided[i] = (local, dist[0, local])
    return [decided[i] for i in range(len(shortlists))]


def exact_top_k(
    Q: np.ndarray,
    R: np.ndarray,
    R2: np.ndarray | None,
    k: int,
    *,
    metric: str,
    cpu_rows: int = 500,
) -> list[tuple[np.ndarray, np.ndarray]]:
    """For each query, its k nearest columns and their distances, exactly as the float64 path.

    ``Q`` and ``R`` are what the float64 path would multiply: the normalised
    queries and references for cosine, the raw ones for l2, with ``R2`` the
    references' squared norms that path is given. On the CPU the first pass takes
    ``cpu_rows`` queries at a time, which bounds its distance matrix; on a card
    the bank is uploaded once and the pass sizes its own launches.
    """
    if k <= 0 or R.shape[0] == 0:
        empty = np.empty(0, dtype=np.int64)
        return [(empty, np.empty(0, dtype=np.float32)) for _ in range(Q.shape[0])]
    if not isinstance(R, np.ndarray):
        # A sparse bank (scipy CSR, the k-WTA codes): its product is already
        # cheap and has no dense float32 pass, so it is decided over the whole
        # bank, as before.
        dist = _exact(Q, R, R2, metric)
        return [
            (top, dist[i, top]) for i, top in enumerate(top_k_by_distance_then_position(dist, k))
        ]
    return _decide(Q, R, R2, k, metric, _first_pass(Q, R, R2, k, metric, cpu_rows))
