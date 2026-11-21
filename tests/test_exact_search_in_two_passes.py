"""The exact search in two passes returns what the single float64 pass returned.

The reference below is the exact path as it stood before the two passes: float64
distances over the whole bank, then the k nearest by distance and then by
position. Every case compares the two neighbour for neighbour: the same
references, in the same order, at the same float32 distances. The cases are the
ones that decide a k-th neighbour: exact ties (twins, and a query that is its own
reference at distance zero), near ties a few ulps apart, a float16 bank as
PROTEA's pool stores it, the l2 metric, a threshold, a k larger than the bank, a
row that is not a number and a sparse bank. The first pass runs on the CPU here;
the card is checked when one is present.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pytest

from protea_method import _exact_two_pass as two_pass
from protea_method._exact_two_pass import (
    FIRST_PASS_DEVICE_ENV,
    dot_error_bound,
    first_pass_margin,
    top_k_by_distance_then_position,
    widened_norm,
)
from protea_method._order_invariance import cosine_distance_f64, l2_distance_f64
from protea_method.knn_search import _search_numpy

Hits = list[list[tuple[str, float]]]


def _single_pass(
    Q: np.ndarray,
    R: Any,
    k: int,
    *,
    metric: str,
    pre_normalized: bool = False,
    threshold: float | None = None,
) -> Hits:
    """The exact path before the two passes, row by row."""
    if metric == "cosine":
        R_ready = R if pre_normalized else R / (np.linalg.norm(R, axis=1, keepdims=True) + 1e-9)
        Qn = Q / (np.linalg.norm(Q, axis=1, keepdims=True) + 1e-9)
        dist = cosine_distance_f64(Qn, R_ready)
    else:
        dist = l2_distance_f64(Q, R, (R**2).sum(axis=1))
    out = []
    for r, top in enumerate(top_k_by_distance_then_position(dist, min(k, R.shape[0]))):
        if threshold is not None:
            top = top[dist[r, top] <= threshold]
        out.append([(str(int(i)), float(dist[r, i])) for i in top])
    return out


def _two_passes(
    Q: np.ndarray,
    R: Any,
    k: int,
    *,
    metric: str,
    pre_normalized: bool = False,
    threshold: float | None = None,
) -> Hits:
    accessions = [str(i) for i in range(R.shape[0])]
    return _search_numpy(
        Q,
        R,
        accessions,
        k,
        distance_threshold=threshold,
        metric=metric,
        pre_normalized=pre_normalized,
    )


@pytest.fixture(autouse=True)
def _first_pass_on_the_cpu(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv(FIRST_PASS_DEVICE_ENV, "cpu")


def _bank_with_ties(rng: np.random.Generator, n: int, dim: int) -> np.ndarray:
    """Random references with exact twins and near twins a few float32 ulps apart."""
    R = rng.standard_normal((n, dim)).astype(np.float32)
    R[n // 2 : n // 2 + 40] = R[:40]  # exact twins: equal distances to every query
    near = R[100:140].copy()
    near[:, 0] = np.nextafter(near[:, 0], np.float32(np.inf))
    near[:, 1] = np.nextafter(near[:, 1], np.float32(-np.inf))
    R[n // 2 + 40 : n // 2 + 80] = near  # near twins: distances a few ulps apart
    return R


@pytest.mark.parametrize("metric", ["cosine", "l2"])
@pytest.mark.parametrize("k", [1, 5, 30])
def test_the_same_neighbours_with_ties_and_near_ties(metric: str, k: int) -> None:
    rng = np.random.default_rng(7)
    R = _bank_with_ties(rng, 3000, 96)
    # Queries that are references (distance zero, with their twins) and fresh ones.
    Q = np.vstack([R[:20], R[100:110], rng.standard_normal((20, 96)).astype(np.float32)])
    assert _two_passes(Q, R, k, metric=metric) == _single_pass(Q, R, k, metric=metric)


def test_the_same_neighbours_on_a_float16_bank_at_protein_language_model_width() -> None:
    rng = np.random.default_rng(11)
    R32 = _bank_with_ties(rng, 2500, 1280)
    R16 = (R32 / np.linalg.norm(R32, axis=1, keepdims=True)).astype(np.float16)
    Q = np.vstack([R16[:15].astype(np.float32), rng.standard_normal((15, 1280)).astype(np.float32)])
    assert _two_passes(Q, R16, 30, metric="cosine", pre_normalized=True) == _single_pass(
        Q, R16, 30, metric="cosine", pre_normalized=True
    )


def test_the_threshold_applies_after_the_selection() -> None:
    rng = np.random.default_rng(3)
    R = _bank_with_ties(rng, 1500, 64)
    Q = np.vstack([R[:10], rng.standard_normal((10, 64)).astype(np.float32)])
    for threshold in (0.0, 0.5, 0.9):
        assert _two_passes(Q, R, 10, metric="cosine", threshold=threshold) == _single_pass(
            Q, R, 10, metric="cosine", threshold=threshold
        )


def test_a_k_larger_than_the_bank_returns_the_whole_bank_in_order() -> None:
    rng = np.random.default_rng(5)
    R = rng.standard_normal((12, 32)).astype(np.float32)
    R[6:] = R[:6]
    Q = rng.standard_normal((4, 32)).astype(np.float32)
    assert _two_passes(Q, R, 50, metric="cosine") == _single_pass(Q, R, 50, metric="cosine")


def test_an_empty_bank_has_no_neighbours() -> None:
    Q = np.ones((3, 8), dtype=np.float32)
    R = np.empty((0, 8), dtype=np.float32)
    assert _two_passes(Q, R, 5, metric="cosine") == [[], [], []]


def test_a_row_that_is_not_a_number_is_decided_over_the_whole_bank() -> None:
    rng = np.random.default_rng(9)
    R = rng.standard_normal((800, 32)).astype(np.float32)
    Q = rng.standard_normal((5, 32)).astype(np.float32)
    Q[2, 3] = np.nan
    assert _two_passes(Q, R, 5, metric="cosine") == _single_pass(Q, R, 5, metric="cosine")


def test_a_sparse_bank_is_decided_over_the_whole_bank() -> None:
    sparse = pytest.importorskip("scipy.sparse")
    rng = np.random.default_rng(13)
    dense = rng.standard_normal((600, 48)).astype(np.float32)
    dense[np.abs(dense) < 1.0] = 0.0
    dense /= np.linalg.norm(dense, axis=1, keepdims=True) + 1e-9
    R = sparse.csr_array(dense)
    Q = rng.standard_normal((6, 48)).astype(np.float32)
    hits = _two_passes(Q, R, 7, metric="cosine", pre_normalized=True)
    assert hits == _single_pass(Q, dense, 7, metric="cosine", pre_normalized=True)


class TestTheMarginIsABound:
    def test_gamma_for_a_protein_language_model_width(self) -> None:
        # 1280 terms: about 7.6e-5, the figure the module's docstring rests on.
        assert dot_error_bound(1280) == pytest.approx(7.63e-5, rel=1e-3)

    @pytest.mark.parametrize("metric", ["cosine", "l2"])
    def test_no_float32_distance_lies_outside_its_margin(self, metric: str) -> None:
        rng = np.random.default_rng(17)
        R = rng.standard_normal((2000, 1280)).astype(np.float32) * rng.uniform(0.1, 3.0, (2000, 1))
        Q = rng.standard_normal((25, 1280)).astype(np.float32)
        if metric == "cosine":
            R = (R / np.linalg.norm(R, axis=1, keepdims=True)).astype(np.float32)
            Q = (Q / np.linalg.norm(Q, axis=1, keepdims=True)).astype(np.float32)
            exact = cosine_distance_f64(Q, R)
            fast = (1.0 - Q @ R.T).astype(np.float32)
        else:
            R2 = (R**2).sum(axis=1)
            exact = l2_distance_f64(Q, R, R2)
            fast = np.maximum(0.0, (Q * Q).sum(axis=1, keepdims=True) + R2 - 2.0 * (Q @ R.T))
        margin = first_pass_margin(Q, widened_norm(float(np.linalg.norm(R, axis=1).max())), metric)
        assert np.all(np.abs(fast.astype(np.float64) - exact.astype(np.float64)) <= margin[:, None])

    def test_the_shortlist_holds_every_reference_the_exact_path_can_keep(self) -> None:
        rng = np.random.default_rng(19)
        R = _bank_with_ties(rng, 2000, 128)
        R = (R / np.linalg.norm(R, axis=1, keepdims=True)).astype(np.float32)
        Q = np.vstack([R[:10], R[100:105]]).astype(np.float32)
        shortlists = two_pass._shortlist_cpu(Q, R, None, 30, "cosine")
        exact = cosine_distance_f64(Q, R)
        for row, cols in enumerate(shortlists):
            kth = np.partition(exact[row], 29)[29]
            assert set(np.flatnonzero(exact[row] <= kth)) <= set(cols.tolist())


def test_the_card_gives_the_same_neighbours(monkeypatch: pytest.MonkeyPatch) -> None:
    torch = pytest.importorskip("torch")
    if not torch.cuda.is_available():
        pytest.skip("no CUDA card")
    monkeypatch.setenv(FIRST_PASS_DEVICE_ENV, "cuda")
    rng = np.random.default_rng(23)
    R = _bank_with_ties(rng, 4000, 1280)
    Q = np.vstack([R[:20], rng.standard_normal((20, 1280)).astype(np.float32)])
    for metric in ("cosine", "l2"):
        assert _two_passes(Q, R, 30, metric=metric) == _single_pass(Q, R, 30, metric=metric)
