"""Disk-backed embedding cache keyed by ``(backend_id, fasta_sha256)``.

The cache exists so a re-run of the LAFA container on the same FASTA
inputs skips the multi-hour ESM-2 forward pass. Layout on disk:

.. code-block:: text

    <cache_dir>/
      <backend_id>/<fasta_sha256>.npz
      <backend_id>/<fasta_sha256>.acc.txt

The ``.npz`` carries a single ``embeddings`` ``(N, D)`` ``float16``
array; the ``.acc.txt`` carries one accession per line, in the order
the columns of ``embeddings`` were written. Loading reads both back and
reconstructs the ``accession -> ndarray`` dict the orchestrator
expects.

Hashing is over the **FASTA content** (after newline normalisation) so
that two paths to the same file map to the same cache key, and a file
edit invalidates the cache automatically.
"""

from __future__ import annotations

import hashlib
import os
import sys
from pathlib import Path

import numpy as np


def hash_fasta(path: Path) -> str:
    """Return the SHA-256 hex digest of a FASTA file's normalised content.

    Newlines are collapsed to ``\\n`` so a file copied across
    platforms hashes to the same value. The whole file is streamed in
    64 KB chunks: even a 280 MB SwissProt FASTA hashes in under a
    second on a typical disk.
    """
    sha = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(65536), b""):
            sha.update(chunk.replace(b"\r\n", b"\n").replace(b"\r", b"\n"))
    return sha.hexdigest()


class EmbeddingCache:
    """Filesystem cache for ``(backend_id, fasta_hash) -> embeddings``.

    The cache is opt-in: pass ``cache_dir=None`` to
    :func:`protea_method.embed.embed_fasta` to disable it entirely.
    When enabled, the directory is created on first write. A
    :class:`PermissionError` on write is downgraded to a warning so a
    read-only container layout does not crash inference; the embeddings
    are still computed and returned, just not persisted.
    """

    def __init__(self, root: Path) -> None:
        self.root = root

    def _paths(self, backend_id: str, fasta_hash: str) -> tuple[Path, Path]:
        """Return the ``(npz, acc)`` paths for a cache entry."""
        sub = self.root / backend_id
        return sub / f"{fasta_hash}.npz", sub / f"{fasta_hash}.acc.txt"

    def has(self, backend_id: str, fasta_hash: str) -> bool:
        """Return ``True`` iff both cache files exist for the key."""
        npz, acc = self._paths(backend_id, fasta_hash)
        return npz.exists() and acc.exists()

    def load(self, backend_id: str, fasta_hash: str) -> dict[str, np.ndarray]:
        """Read the cache entry back into an ``accession -> ndarray`` dict.

        Raises :class:`FileNotFoundError` if the entry is missing
        (caller is expected to check :meth:`has` first; this is a hard
        error rather than a silent miss so a partially-deleted cache
        does not silently slip into a recompute).
        """
        npz, acc = self._paths(backend_id, fasta_hash)
        accessions = acc.read_text(encoding="utf-8").splitlines()
        with np.load(npz) as data:
            arr = data["embeddings"]
        if arr.shape[0] != len(accessions):
            raise ValueError(
                f"cache entry {npz} mismatched: {arr.shape[0]} rows vs "
                f"{len(accessions)} accessions"
            )
        return {a: arr[i] for i, a in enumerate(accessions)}

    def store(
        self,
        backend_id: str,
        fasta_hash: str,
        embeddings: dict[str, np.ndarray],
    ) -> None:
        """Persist embeddings to disk. Silent no-op on PermissionError.

        Accession ordering is taken from the dict's insertion order
        (Python 3.7+ guarantee). The ``.npz`` saves a single ``float16``
        matrix; a sibling ``.acc.txt`` records the row order so
        :meth:`load` can reconstruct the dict deterministically.
        """
        npz, acc = self._paths(backend_id, fasta_hash)
        try:
            npz.parent.mkdir(parents=True, exist_ok=True)
            accessions = list(embeddings.keys())
            matrix = np.stack(
                [np.asarray(embeddings[a], dtype=np.float16) for a in accessions]
            )
            np.savez_compressed(npz, embeddings=matrix)
            acc.write_text("\n".join(accessions) + "\n", encoding="utf-8")
        except (PermissionError, OSError) as exc:
            sys.stderr.write(
                f"[embed-cache] warning: could not persist to {self.root}: {exc}\n"
            )


def resolve_cache_dir(explicit: Path | None) -> Path | None:
    """Resolve the cache directory from arg or ``LAFA_EMBED_CACHE`` env.

    Precedence is ``explicit`` arg, then the ``LAFA_EMBED_CACHE``
    environment variable, then ``None`` (cache disabled). The container
    sets the env to ``/app/output/.embed_cache`` so a bind-mounted
    ``./output`` directory persists embeddings across invocations
    without any extra mount.
    """
    if explicit is not None:
        return explicit
    env = os.environ.get("LAFA_EMBED_CACHE")
    if env:
        return Path(env)
    return None


__all__ = [
    "EmbeddingCache",
    "hash_fasta",
    "resolve_cache_dir",
]
