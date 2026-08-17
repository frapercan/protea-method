"""Freeing the corpus tensor between searches, and why its absence was invisible.

Looping ``_search_torch`` across the three GO aspects pins about 10 GB on a 12 GB
card. The corpus-fits-in-VRAM check inside ``_torch_target_device`` then flips
the device back to CPU for the rest of the run.

Nothing fails when that happens. The run completes, slower, and the only symptom
is that the second and third aspects did not use the card the first one did,
which no log states. That is why this was found by watching memory on one machine
in May rather than by any check.
"""

from __future__ import annotations

from types import SimpleNamespace

import pytest

from protea_method.knn_search import _release_corpus_vram


class _Sentinel:
    """Stands in for the corpus tensor, so we can see whether it was dropped."""


def test_a_cpu_device_frees_nothing_and_imports_nothing(monkeypatch):
    """The CPU path must not import torch, since it may not be installed."""
    monkeypatch.setitem(__import__("sys").modules, "torch", None)

    _release_corpus_vram(_Sentinel(), SimpleNamespace(type="cpu"))


def test_a_cuda_device_empties_the_cache(monkeypatch):
    calls: list[str] = []
    fake = SimpleNamespace(cuda=SimpleNamespace(empty_cache=lambda: calls.append("empty")))
    monkeypatch.setitem(__import__("sys").modules, "torch", fake)

    _release_corpus_vram(_Sentinel(), SimpleNamespace(type="cuda"))

    assert calls == ["empty"]


def test_the_cpu_path_does_not_empty_the_cache(monkeypatch):
    """Calling it on CPU would be harmless but would mean the guard is not reading device."""
    calls: list[str] = []
    fake = SimpleNamespace(cuda=SimpleNamespace(empty_cache=lambda: calls.append("empty")))
    monkeypatch.setitem(__import__("sys").modules, "torch", fake)

    _release_corpus_vram(_Sentinel(), SimpleNamespace(type="cpu"))

    assert calls == []


def test_an_mps_device_is_treated_as_not_cuda(monkeypatch):
    """Only the CUDA allocator is being drained; another accelerator is not this bug."""
    calls: list[str] = []
    fake = SimpleNamespace(cuda=SimpleNamespace(empty_cache=lambda: calls.append("empty")))
    monkeypatch.setitem(__import__("sys").modules, "torch", fake)

    _release_corpus_vram(_Sentinel(), SimpleNamespace(type="mps"))

    assert calls == []


def test_the_search_calls_it_before_returning():
    """The regression this exists for: a search that frees nothing leaks per aspect."""
    import inspect

    from protea_method import knn_search

    source = inspect.getsource(knn_search._search_torch)

    assert "_release_corpus_vram(R_t, device)" in source
    # Before the return, or the tensor survives the call it was meant to outlive.
    assert source.index("_release_corpus_vram") < source.rindex("return results")


@pytest.mark.parametrize("device_type", ["cpu", "cuda", "mps", "xpu"])
def test_it_never_raises_whatever_the_device(monkeypatch, device_type):
    """It runs at the end of every search, so raising here would fail a completed run."""
    fake = SimpleNamespace(cuda=SimpleNamespace(empty_cache=lambda: None))
    monkeypatch.setitem(__import__("sys").modules, "torch", fake)

    _release_corpus_vram(_Sentinel(), SimpleNamespace(type=device_type))
