"""Embedding computation for the LAFA self-contained container.

This sub-package is the optional path that lets the standalone
``method_main.py`` entrypoint run end-to-end from FASTA inputs alone,
without bind-mounting pre-computed embedding parquet files.

The orchestration entrypoint is :func:`embed_fasta`. It resolves a
backend id (e.g. ``esm2_t36_3B`` for the LB.2 champion config) to a
:class:`protea_backends` plugin via the ``protea.backends``
``entry_points`` group, computes mean-pooled embeddings for every
sequence in the FASTA, and caches the result to disk keyed by
``(backend_id, fasta_sha256, model_revision)``.

Heavy ML dependencies (``torch``, ``transformers``, ``protea-backends``)
are imported lazily inside :mod:`protea_method.embed.backend`. The
default ``pip install protea-method`` stays slim; users opt in with
``pip install "protea-method[esm]"`` to bring in the ESM-2 runtime.

A ``mock_constant`` backend id is shipped for unit tests: it returns a
deterministic constant vector per sequence so the cache layer and the
``method_main`` end-to-end path can be exercised without downloading any
PLM weights.
"""

from protea_method.embed.backend import (
    BACKEND_IDS,
    BackendSpec,
    embed_fasta,
    resolve_backend,
)
from protea_method.embed.cache import EmbeddingCache

__all__ = [
    "BACKEND_IDS",
    "BackendSpec",
    "EmbeddingCache",
    "embed_fasta",
    "resolve_backend",
]
