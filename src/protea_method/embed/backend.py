"""Backend resolver and ``embed_fasta`` orchestrator.

This module is the thin glue between :mod:`protea_method.io` (FASTA
loader), :mod:`protea_backends` (the plugin package that ships ESM /
ProstT5 / Ankh / ESM3-C), and :mod:`protea_method.embed.cache` (the
disk cache).

Backend identifiers
-------------------

The CLI exposes **friendly ids** rather than raw HuggingFace paths so
the LAFA container surface stays stable across model-card renames. The
canonical map is :data:`BACKEND_IDS`. Today:

* ``esm2_t36_3B``: ESM-2 3B, ``facebook/esm2_t36_3B_UR50D``. Champion
  config of the LB.2 v226 leakage-fixed reranker (selective avg
  cafaeval Fmax 0.6215 on v226).
* ``esm2_t33_650M``: ESM-2 650M, ``facebook/esm2_t33_650M_UR50D``.
  Lighter alternative (~2.5 GB weights, ~5x faster) for smoke runs
  and CI integration tests.
* ``prost_t5_xl_uniref50``: ProstT5, ``Rostlab/ProstT5``. Optional
  cross-check backend, served by the ``t5`` plugin.
* ``mock_constant``: deterministic in-process backend used by unit
  tests; returns a constant float16 vector per sequence so the cache
  and the entrypoint can be exercised without downloading any
  weights.

Anything else raises :class:`ValueError` with a friendly list of the
ids accepted today.

Heavy imports
-------------

``protea_backends`` itself is import-cheap, but the resolved plugin
imports ``torch`` and ``transformers`` lazily inside ``load_model``.
The mock backend imports nothing heavy. ``resolve_backend`` therefore
costs essentially nothing until the caller actually asks for a real
PLM.
"""

from __future__ import annotations

import sys
from dataclasses import dataclass
from importlib.metadata import entry_points
from pathlib import Path
from typing import Any

import numpy as np

from protea_method.embed.cache import EmbeddingCache, hash_fasta, resolve_cache_dir
from protea_method.io import read_fasta


@dataclass(frozen=True)
class BackendSpec:
    """Static description of a backend id.

    ``plugin_name`` is the ``protea.backends`` entry-point key
    (``"esm"``, ``"t5"``, ``"ankh"``, ``"esm3c"``). ``model_name`` is
    the HuggingFace identifier passed verbatim to ``plugin.load_model``.
    ``pip_extra`` is the optional extras group the user must install
    (``protea-method[esm]`` etc.) for the runtime to be available.
    """

    backend_id: str
    plugin_name: str
    model_name: str
    pip_extra: str
    description: str


#: Friendly id -> :class:`BackendSpec` table. Keep alphabetical so the
#: error message produced by :func:`resolve_backend` is stable.
BACKEND_IDS: dict[str, BackendSpec] = {
    "esm2_t33_650M": BackendSpec(
        backend_id="esm2_t33_650M",
        plugin_name="esm",
        model_name="facebook/esm2_t33_650M_UR50D",
        pip_extra="esm",
        description="ESM-2 650M (smaller, faster; recommended for smoke runs)",
    ),
    "esm2_t36_3B": BackendSpec(
        backend_id="esm2_t36_3B",
        plugin_name="esm",
        model_name="facebook/esm2_t36_3B_UR50D",
        pip_extra="esm",
        description="ESM-2 3B (LB.2 champion config; ~12 GB weights)",
    ),
    "mock_constant": BackendSpec(
        backend_id="mock_constant",
        plugin_name="__mock__",
        model_name="",
        pip_extra="",
        description="Deterministic in-process backend for unit tests.",
    ),
    "prost_t5_xl_uniref50": BackendSpec(
        backend_id="prost_t5_xl_uniref50",
        plugin_name="t5",
        model_name="Rostlab/ProstT5",
        pip_extra="t5",
        description="ProstT5 (cross-check backend; ~5.5 GB weights)",
    ),
}


def resolve_backend(backend_id: str) -> BackendSpec:
    """Look up ``backend_id`` in :data:`BACKEND_IDS` or raise ``ValueError``.

    The error message lists all accepted ids and points at the
    optional-extras install recipe so a user who picked the wrong id
    can self-correct without reading the source.
    """
    spec = BACKEND_IDS.get(backend_id)
    if spec is None:
        known = ", ".join(sorted(BACKEND_IDS))
        raise ValueError(
            f"unknown backend_id {backend_id!r}; accepted: {known}. "
            "Real PLM backends require the matching extras install "
            "(e.g. `pip install \"protea-method[esm]\"`)."
        )
    return spec


def _mock_embed(sequences: list[str], dim: int = 8) -> np.ndarray:
    """Return a deterministic float16 matrix for the mock backend.

    Row ``i`` encodes ``len(sequences[i])`` plus an offset so equal-
    length sequences are still distinguishable. The vector is then
    L2-normalised so KNN behaviour matches a real backend's
    post-normalisation output.
    """
    rows = []
    for idx, seq in enumerate(sequences):
        base = np.zeros(dim, dtype=np.float32)
        base[idx % dim] = float(len(seq))
        base[(idx + 1) % dim] = float(idx + 1)
        norm = float(np.linalg.norm(base)) or 1.0
        rows.append((base / norm).astype(np.float16))
    return np.stack(rows)


def _emit_noop(*_args: Any, **_kwargs: Any) -> None:
    """Drop-in for the structured-event callback the plugins expect."""
    return None


def _load_plugin(spec: BackendSpec) -> Any:
    """Resolve the ``protea.backends`` entry point for ``spec``.

    Imports happen lazily so the default ``pip install protea-method``
    keeps working without ``protea-backends`` installed at all; the
    error message points at the right extras group to fix the
    environment.
    """
    try:
        eps = entry_points(group="protea.backends")
    except Exception as exc:
        raise RuntimeError(
            f"protea-backends entry points unavailable: {exc}. "
            f"Install `protea-method[{spec.pip_extra}]` to enable "
            f"backend {spec.backend_id!r}."
        ) from exc

    by_name: dict[str, Any] = {ep.name: ep for ep in eps}
    ep = by_name.get(spec.plugin_name)
    if ep is None:
        raise RuntimeError(
            f"backend plugin {spec.plugin_name!r} not registered. "
            f"Install `protea-method[{spec.pip_extra}]` and retry."
        )
    return ep.load()


def _compute_with_backend(
    spec: BackendSpec,
    sequences: list[str],
    *,
    device: str,
    batch_size: int,
) -> np.ndarray:
    """Run the real PLM plugin in chunks and return a ``(N, D)`` matrix.

    Chunking keeps the GPU memory footprint bounded even when the
    caller hands in tens of thousands of sequences. ``device="auto"``
    picks ``cuda`` when a CUDA build of torch is present, ``cpu``
    otherwise.
    """
    plugin = _load_plugin(spec)
    resolved_device = _resolve_device(device)
    sys.stderr.write(
        f"[embed] loading {spec.plugin_name}:{spec.model_name} on {resolved_device}\n"
    )
    model, tokenizer = plugin.load_model(spec.model_name, resolved_device, _emit_noop)

    out: list[np.ndarray] = []
    for start in range(0, len(sequences), batch_size):
        batch = sequences[start : start + batch_size]
        sys.stderr.write(
            f"[embed] batch {start}-{start + len(batch)}/{len(sequences)}\n"
        )
        out.append(
            plugin.embed_batch(
                model, tokenizer, batch, emit=_emit_noop,
            ).astype(np.float16)
        )
    return np.concatenate(out, axis=0)


def _resolve_device(device: str) -> str:
    """Resolve ``"auto"`` to ``"cuda"`` if available else ``"cpu"``."""
    if device != "auto":
        return device
    try:
        import torch

        if torch.cuda.is_available():
            return "cuda"
    except ImportError:
        pass
    return "cpu"


def _compute_matrix(
    spec: BackendSpec,
    seqs: list[str],
    *,
    device: str,
    batch_size: int,
) -> np.ndarray:
    """Branch between the mock and the real backend code paths."""
    if spec.backend_id == "mock_constant":
        return _mock_embed(seqs)
    return _compute_with_backend(spec, seqs, device=device, batch_size=batch_size)


def _emit_cache_hit(
    backend_id: str,
    path: Path,
    fasta_hash: str,
    cache: EmbeddingCache,
    sequences: dict[str, str],
) -> dict[str, np.ndarray]:
    """Format the cache-hit log line and read the entry back to a dict."""
    sys.stderr.write(
        f"[embed] cache-hit {backend_id} {path.name} {fasta_hash[:12]}\n"
    )
    loaded = cache.load(backend_id, fasta_hash)
    # Reorder to match the FASTA emission so downstream KNN sees a stable
    # row layout even if the cache file was written by an earlier run.
    return {acc: loaded[acc] for acc in sequences if acc in loaded}


def embed_fasta(
    fasta_path: Path | str,
    *,
    backend_id: str,
    cache_dir: Path | None = None,
    device: str = "auto",
    batch_size: int = 8,
) -> dict[str, np.ndarray]:
    """Compute (or load from cache) embeddings for every sequence in ``fasta_path``.

    ``backend_id`` is resolved via :func:`resolve_backend` (raises
    :class:`ValueError` on unknown). ``cache_dir=None`` consults the
    ``LAFA_EMBED_CACHE`` env var; an unset env disables caching.
    ``device="auto"`` picks CUDA when available, CPU otherwise.

    Returns an ``accession -> float16 ndarray`` dict in FASTA order.
    """
    path = Path(fasta_path)
    spec = resolve_backend(backend_id)
    sequences = read_fasta(path)
    if not sequences:
        return {}

    cache_root = resolve_cache_dir(cache_dir)
    cache = EmbeddingCache(cache_root) if cache_root else None
    fasta_hash = hash_fasta(path)

    if cache is not None and cache.has(backend_id, fasta_hash):
        return _emit_cache_hit(backend_id, path, fasta_hash, cache, sequences)

    sys.stderr.write(
        f"[embed] cache-miss, computing {len(sequences)} embeddings via "
        f"{backend_id}\n"
    )
    accs = list(sequences.keys())
    seqs = [sequences[a] for a in accs]
    matrix = _compute_matrix(spec, seqs, device=device, batch_size=batch_size)
    result = {acc: matrix[i] for i, acc in enumerate(accs)}
    if cache is not None:
        cache.store(backend_id, fasta_hash, result)
    return result


__all__ = [
    "BACKEND_IDS",
    "BackendSpec",
    "embed_fasta",
    "resolve_backend",
]
