"""Anc2Vec GO-term embedding index.

Loads a pre-trained Anc2Vec dictionary (200-dim GO-term vectors) from a
compact ``.npz`` artifact and exposes a zero-copy lookup by ``GO:`` id
plus a batched variant that returns an ``(N, D)`` matrix. Missing rows
fill with the zero vector by default so downstream cosine operations
degrade to 0 rather than NaN.

Pure numpy + filesystem; no DB, no FastAPI, no protea-core. The default
artifact path is taken from the ``PROTEA_METHOD_ANC2VEC_PATH``
environment variable when set, falling back to a sibling ``artifacts``
directory under the current working directory. The LAFA bind-mount
container will set the env var to its mounted location.

Original file: PROTEA's ``protea/core/anc2vec_embeddings.py``. The
embeddings shipped by https://github.com/aedera/anc2vec (GO release
2020-10-06) are the canonical source.
"""

from __future__ import annotations

import os
from functools import lru_cache
from pathlib import Path

import numpy as np

_DEFAULT_PATH = Path(
    os.environ.get(
        "PROTEA_METHOD_ANC2VEC_PATH",
        str(Path.cwd() / "artifacts" / "anc2vec" / "anc2vec_2020-10.npz"),
    )
)


class Anc2VecIndex:
    """In-memory lookup table for Anc2Vec GO-term embeddings."""

    __slots__ = ("_idx", "dim", "embeddings", "go_ids", "release")

    def __init__(self, path: str | Path | None = None) -> None:
        src = Path(path) if path else _DEFAULT_PATH
        data = np.load(src, allow_pickle=True)
        self.embeddings = np.ascontiguousarray(data["embeddings"], dtype=np.float32)
        self.go_ids = [str(g) for g in data["go_ids"]]
        self._idx = {g: i for i, g in enumerate(self.go_ids)}
        self.dim = int(self.embeddings.shape[1])
        self.release = str(data["ontology_release"]) if "ontology_release" in data.files else ""

    def __len__(self) -> int:
        return len(self.go_ids)

    def __contains__(self, go_id: str) -> bool:
        return go_id in self._idx

    def vec(self, go_id: str) -> np.ndarray | None:
        """Return the embedding for ``go_id`` or ``None`` when unknown."""
        i = self._idx.get(go_id)
        return self.embeddings[i] if i is not None else None

    def batch(self, go_ids: list[str], *, zero_if_missing: bool = True) -> np.ndarray:
        """Return an ``(N, dim)`` matrix; missing rows zero (or NaN if disabled)."""
        fill = 0.0 if zero_if_missing else np.nan
        out = np.full((len(go_ids), self.dim), fill, dtype=np.float32)
        for row, g in enumerate(go_ids):
            i = self._idx.get(g)
            if i is not None:
                out[row] = self.embeddings[i]
        return out


@lru_cache(maxsize=2)
def get_index(path: str | None = None) -> Anc2VecIndex:
    """Return a process-wide singleton index (keyed by path)."""
    return Anc2VecIndex(path)


__all__ = ["Anc2VecIndex", "get_index"]
