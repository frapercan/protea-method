"""Tests for the LAFA-EMB.1 embedding orchestration."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

import numpy as np
import pytest

from protea_method.embed import (
    BACKEND_IDS,
    BackendSpec,
    EmbeddingCache,
    embed_fasta,
    resolve_backend,
)
from protea_method.embed.cache import hash_fasta, resolve_cache_dir

FIXTURES = Path(__file__).parent / "fixtures"


def test_backend_ids_table_has_champion_default() -> None:
    """The LB.2 champion config must remain a registered backend id."""
    assert "esm2_t36_3B" in BACKEND_IDS
    spec = BACKEND_IDS["esm2_t36_3B"]
    assert spec.plugin_name == "esm"
    assert spec.model_name == "facebook/esm2_t36_3B_UR50D"
    assert spec.pip_extra == "esm"


def test_resolve_backend_unknown_lists_known() -> None:
    """The error message guides the user back to the accepted ids."""
    with pytest.raises(ValueError, match="esm2_t36_3B"):
        resolve_backend("bogus_backend")


def test_resolve_backend_mock_returns_spec() -> None:
    spec = resolve_backend("mock_constant")
    assert isinstance(spec, BackendSpec)
    assert spec.plugin_name == "__mock__"


def test_embed_fasta_mock_backend_round_trip(tmp_path: Path) -> None:
    """Mock backend returns a deterministic float16 matrix per sequence."""
    fasta = FIXTURES / "tiny.fasta"
    table = embed_fasta(
        fasta, backend_id="mock_constant", cache_dir=tmp_path / "c1",
    )
    assert sorted(table.keys()) == ["P00001", "P00002", "P00003"]
    for vec in table.values():
        assert vec.dtype == np.float16
        assert vec.shape == (8,)


def test_embed_fasta_cache_hit_skips_recompute(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The second call must hit the cache and skip the backend code path."""
    fasta = FIXTURES / "tiny.fasta"
    cache_dir = tmp_path / "cache"
    first = embed_fasta(fasta, backend_id="mock_constant", cache_dir=cache_dir)

    # Sabotage the mock embedder; if the cache layer truly short-circuits the
    # call site, we should still get the *first-run* vectors back.
    def _explode(*_args: Any, **_kwargs: Any) -> np.ndarray:
        raise AssertionError("embedding compute should have been skipped")

    monkeypatch.setattr("protea_method.embed.backend._mock_embed", _explode)
    second = embed_fasta(fasta, backend_id="mock_constant", cache_dir=cache_dir)
    for acc in first:
        np.testing.assert_array_equal(first[acc], second[acc])


def test_embedding_cache_isolates_per_backend(tmp_path: Path) -> None:
    """Different backend ids must not collide on a shared cache root."""
    cache = EmbeddingCache(tmp_path)
    embeddings = {
        "P00001": np.array([1.0, 0.0], dtype=np.float16),
        "P00002": np.array([0.0, 1.0], dtype=np.float16),
    }
    cache.store("esm2_t33_650M", "deadbeef", embeddings)
    cache.store("prost_t5_xl_uniref50", "deadbeef", embeddings)

    assert cache.has("esm2_t33_650M", "deadbeef")
    assert cache.has("prost_t5_xl_uniref50", "deadbeef")
    assert not cache.has("esm2_t36_3B", "deadbeef")  # different backend

    loaded = cache.load("esm2_t33_650M", "deadbeef")
    np.testing.assert_array_equal(loaded["P00001"], embeddings["P00001"])


def test_embedding_cache_load_missing_raises(tmp_path: Path) -> None:
    cache = EmbeddingCache(tmp_path)
    assert not cache.has("esm2_t36_3B", "missing")
    with pytest.raises(FileNotFoundError):
        cache.load("esm2_t36_3B", "missing")


def test_resolve_cache_dir_env_fallback(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    """``LAFA_EMBED_CACHE`` wins when no explicit dir is supplied."""
    monkeypatch.setenv("LAFA_EMBED_CACHE", str(tmp_path))
    assert resolve_cache_dir(None) == tmp_path
    monkeypatch.delenv("LAFA_EMBED_CACHE")
    assert resolve_cache_dir(None) is None


def test_hash_fasta_normalises_newlines(tmp_path: Path) -> None:
    """CRLF and LF copies of the same FASTA must hash to the same digest."""
    lf = tmp_path / "lf.fasta"
    crlf = tmp_path / "crlf.fasta"
    lf.write_bytes(b">a\nMKV\n")
    crlf.write_bytes(b">a\r\nMKV\r\n")
    assert hash_fasta(lf) == hash_fasta(crlf)


def test_embed_fasta_rejects_unknown_backend(tmp_path: Path) -> None:
    fasta = FIXTURES / "tiny.fasta"
    with pytest.raises(ValueError, match="unknown backend_id"):
        embed_fasta(fasta, backend_id="not_a_real_backend", cache_dir=tmp_path)


@pytest.mark.requires_esm2
def test_embed_fasta_real_esm2_smoke(tmp_path: Path) -> None:
    """End-to-end smoke against the real ESM-2 backend.

    Gated on ``PROTEA_HAS_ESM2=1`` so CI does not download multi-GB
    weights. Run locally with::

        PROTEA_HAS_ESM2=1 poetry run pytest -m requires_esm2

    Uses the lighter ``esm2_t33_650M`` to keep the run under five
    minutes on a workstation CPU.
    """
    if os.environ.get("PROTEA_HAS_ESM2") != "1":
        pytest.skip("set PROTEA_HAS_ESM2=1 to run the real ESM-2 smoke")
    table = embed_fasta(
        FIXTURES / "tiny.fasta",
        backend_id="esm2_t33_650M",
        cache_dir=tmp_path,
        device="cpu",
        batch_size=2,
    )
    assert sorted(table.keys()) == ["P00001", "P00002", "P00003"]
    for vec in table.values():
        assert vec.dtype == np.float16
        # ESM-2 650M hidden dim is 1280.
        assert vec.shape == (1280,)


def test_embedding_cache_store_swallows_permission_error(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A read-only filesystem must not crash inference."""
    cache = EmbeddingCache(tmp_path / "ro")

    def _refuse(*_args: Any, **_kwargs: Any) -> None:
        raise PermissionError("read-only fs")

    monkeypatch.setattr(Path, "mkdir", _refuse)
    cache.store(
        "esm2_t36_3B",
        "deadbeef",
        {"P00001": np.zeros(2, dtype=np.float16)},
    )
    # Cache write should have been skipped silently.
    assert not (tmp_path / "ro" / "esm2_t36_3B").exists()
