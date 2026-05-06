"""On-disk PCA state cache for the inference pipeline.

The PCA projection of reference embeddings into 16 dims is a feature
input for the re-ranker. Fitting it on the full reference pool is
expensive (~50k samples × ~1280 dims) and the result is deterministic
for a given embedding configuration, so we materialise
``(mean, components)`` into a single ``.npz`` artifact and reuse it
across all callers (workers, batch jobs, container runs) that share
the same configuration id.

Artifact layout: one file per configuration id:
``{artifacts_dir}/{config_id}.npz`` with two arrays ``mean`` (D,)
float32 and ``components`` (16, D) float32.
"""

from __future__ import annotations

import os
import uuid
from pathlib import Path

import numpy as np

from protea_method.reranker import EMBEDDING_PCA_DIM, fit_embedding_pca

_DEFAULT_ARTIFACTS_DIR = Path(
    os.environ.get(
        "PROTEA_PCA_ARTIFACTS_DIR",
        str(Path.cwd() / "artifacts" / "pca"),
    )
)


def _resolve_dir(artifacts_dir: Path | None) -> Path:
    return artifacts_dir if artifacts_dir is not None else _DEFAULT_ARTIFACTS_DIR


def _pca_state_path(config_id: uuid.UUID, artifacts_dir: Path | None = None) -> Path:
    return _resolve_dir(artifacts_dir) / f"{config_id}.npz"


def _load_pca_state(
    config_id: uuid.UUID,
    artifacts_dir: Path | None = None,
) -> tuple[np.ndarray, np.ndarray] | None:
    path = _pca_state_path(config_id, artifacts_dir)
    if not path.exists():
        return None
    try:
        data = np.load(path)
        return (
            np.ascontiguousarray(data["mean"], dtype=np.float32),
            np.ascontiguousarray(data["components"], dtype=np.float32),
        )
    except Exception:
        return None


def _save_pca_state(
    config_id: uuid.UUID,
    mean: np.ndarray,
    components: np.ndarray,
    artifacts_dir: Path | None = None,
) -> None:
    path = _pca_state_path(config_id, artifacts_dir)
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(path, mean=mean, components=components)


def load_or_fit_pca_state(
    config_id: uuid.UUID,
    unified_embeddings_f32: np.ndarray,
    *,
    artifacts_dir: Path | None = None,
) -> tuple[np.ndarray, np.ndarray] | None:
    """Load PCA state from disk or fit on the reference pool.

    Returns ``None`` when the reference pool is empty (no projection
    possible). The artifact is shared across all callers and every
    prediction run that uses this configuration id; fit once, reuse
    forever.
    """
    cached = _load_pca_state(config_id, artifacts_dir)
    if cached is not None:
        return cached
    if unified_embeddings_f32.size == 0:
        return None
    mean, components = fit_embedding_pca(unified_embeddings_f32, EMBEDDING_PCA_DIM)
    _save_pca_state(config_id, mean, components, artifacts_dir)
    return mean, components


__all__ = [
    "load_or_fit_pca_state",
]
