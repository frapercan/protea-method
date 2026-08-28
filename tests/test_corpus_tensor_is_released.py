"""The corpus tensor is freed before the allocator is drained.

The first attempt at this handed the tensor to a helper that deleted its own
parameter. That frees nothing: the caller's binding outlives the call, so the
refcount never reaches zero and ``empty_cache`` drains an allocator that is still
holding the block. The tests written for it asserted that ``empty_cache`` had been
CALLED, which is true of the broken shape too, so they passed.

These assert the thing that matters instead: at the moment the drain runs, is the
tensor still alive? A weakref answers that and a call count does not.

Without a release, looping ``_search_torch`` across the three GO aspects pins
about 10 GB on a 12 GB card, and the corpus-fits-in-VRAM check inside
``_torch_target_device`` then flips the device back to CPU for the rest of the
run. The run completes, slower, and says nothing.
"""

from __future__ import annotations

import sys
import types
import weakref

import numpy as np
import pytest


class _Corpus:
    """Stands in for the corpus tensor, and can be watched by a weakref."""

    def __init__(self) -> None:
        self.shape = (4, 3)

    def to(self, _device):  # noqa: ANN001
        return self


def _fake_torch(alive_at_drain: list[bool], probe: list) -> types.SimpleNamespace:
    def empty_cache() -> None:
        alive_at_drain.append(probe[0]() is not None)

    class _NoGrad:
        def __enter__(self):
            return self

        def __exit__(self, *exc):
            return False

    def from_numpy(_arr):  # noqa: ANN001
        # Created here and handed straight over, never held: a closure keeping
        # its own reference would make this fixture the thing that pins the
        # tensor, and the test would fail on the correct code.
        corpus = _Corpus()
        probe.append(weakref.ref(corpus))
        return corpus

    return types.SimpleNamespace(
        no_grad=_NoGrad,
        from_numpy=from_numpy,
        cuda=types.SimpleNamespace(empty_cache=empty_cache),
        nn=types.SimpleNamespace(
            functional=types.SimpleNamespace(normalize=lambda t, **_: t)
        ),
    )


@pytest.mark.parametrize("metric", ["l2", "cosine"])
def test_the_corpus_is_dead_by_the_time_the_allocator_is_drained(monkeypatch, metric):
    """The whole point, and the thing the previous tests could not see."""
    from protea_method import knn_search

    alive_at_drain: list[bool] = []
    probe: list = []
    fake = _fake_torch(alive_at_drain, probe)
    monkeypatch.setitem(sys.modules, "torch", fake)
    monkeypatch.setattr(
        knn_search, "_torch_target_device", lambda *_a, **_k: types.SimpleNamespace(type="cuda")
    )
    monkeypatch.setattr(knn_search, "_torch_knn_chunk_size", lambda: 8)
    from protea_method import _chunked_topk as ct

    monkeypatch.setattr(
        ct, "_chunk_topk", lambda *_a, **_k: (np.zeros((1, 1)), np.zeros((1, 1), dtype=int))
    )
    monkeypatch.setattr(ct, "_hits_from_topk", lambda *_a, **_k: [[("R000", 0.0)]])

    knn_search._search_torch(
        np.zeros((1, 3), dtype=np.float32),
        np.zeros((4, 3), dtype=np.float32),
        ["R000", "R001", "R002", "R003"],
        k=1,
        distance_threshold=None,
        metric=metric,
    )

    assert alive_at_drain == [False], (
        "the corpus tensor was still referenced when empty_cache ran, so the "
        "allocator had nothing to hand back"
    )


def test_a_cpu_run_never_touches_the_cuda_allocator(monkeypatch):
    """Draining on CPU would import torch's cuda module for nothing, and may not exist."""
    from protea_method import knn_search

    alive_at_drain: list[bool] = []
    probe: list = []
    fake = _fake_torch(alive_at_drain, probe)
    monkeypatch.setitem(sys.modules, "torch", fake)
    monkeypatch.setattr(
        knn_search, "_torch_target_device", lambda *_a, **_k: types.SimpleNamespace(type="cpu")
    )
    monkeypatch.setattr(knn_search, "_torch_knn_chunk_size", lambda: 8)
    from protea_method import _chunked_topk as ct

    monkeypatch.setattr(
        ct, "_chunk_topk", lambda *_a, **_k: (np.zeros((1, 1)), np.zeros((1, 1), dtype=int))
    )
    monkeypatch.setattr(ct, "_hits_from_topk", lambda *_a, **_k: [[("R000", 0.0)]])

    knn_search._search_torch(
        np.zeros((1, 3), dtype=np.float32),
        np.zeros((4, 3), dtype=np.float32),
        ["R000"],
        k=1,
        distance_threshold=None,
        metric="l2",
    )
    assert alive_at_drain == []
