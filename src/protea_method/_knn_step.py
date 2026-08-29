"""The two shapes the neighbour search takes, side by side.

One search over the whole bank, or three over aspect-filtered slices. They
are the same step and the self exclusion has to happen in both, so they sit
together: two functions that look alike are harder to change apart than one
function and one block inlined into the orchestrator, which is what these
were when the exclusion was added to only one of them.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import numpy as np

from protea_method._self_by_sequence import Queries, drop_own_sequence, self_margin
from protea_method.feature_enricher import ASPECT_CODES
from protea_method.knn_search import search_knn

if TYPE_CHECKING:
    from protea_method.pipeline import PredictConfig

__all__ = ("Queries", "aspect_separated_knn", "unified_knn")


def _build_go_map(
    neighbors_per_query: list[list[tuple[str, float]]],
    annotations: dict[str, list[dict[str, Any]]],
) -> dict[str, list[dict[str, Any]]]:
    """Collect annotations of every neighbour seen in the KNN result."""
    go_map: dict[str, list[dict[str, Any]]] = {}
    for hits in neighbors_per_query:
        for ref_acc, _ in hits:
            if ref_acc not in go_map:
                go_map[ref_acc] = list(annotations.get(ref_acc, []))
    return go_map


def _partition_refs_by_aspect(
    reference_accessions: list[str],
    reference_embeddings: np.ndarray,
    annotations: dict[str, list[dict[str, Any]]],
    go_aspect_map: dict[int, str],
) -> dict[str, tuple[list[str], np.ndarray]]:
    """Group reference proteins by the GO aspects of their annotations.

    A reference belongs to aspect ``a`` iff at least one of its
    annotations resolves to aspect ``a`` via ``go_aspect_map``. The
    returned mapping has one entry per ``ASPECT_CODES`` letter, each
    pointing at the filtered ``(accessions, embeddings)`` pair.
    """
    per_aspect_idx: dict[str, list[int]] = {a: [] for a in ASPECT_CODES}
    for ref_idx, ref_acc in enumerate(reference_accessions):
        seen: set[str] = set()
        for ann in annotations.get(ref_acc, []):
            asp = go_aspect_map.get(int(ann["go_term_id"]), "")
            if asp in per_aspect_idx and asp not in seen:
                per_aspect_idx[asp].append(ref_idx)
                seen.add(asp)
    out: dict[str, tuple[list[str], np.ndarray]] = {}
    for asp, idx_list in per_aspect_idx.items():
        if not idx_list:
            out[asp] = ([], np.zeros((0, reference_embeddings.shape[1]), dtype=np.float32))
            continue
        idx_array = np.asarray(idx_list, dtype=np.int64)
        out[asp] = (
            [reference_accessions[i] for i in idx_list],
            reference_embeddings[idx_array],
        )
    return out


def unified_knn(
    *,
    queries: Queries,
    reference_accessions: list[str],
    reference_embeddings: np.ndarray,
    annotations: dict[str, list[dict[str, Any]]],
    cfg: PredictConfig,
) -> tuple[
    dict[str, list[list[tuple[str, float]]]],
    dict[str, dict[str, list[dict[str, Any]]]],
]:
    """One KNN search over the whole bank, keyed under the empty aspect.

    A function rather than a block inside ``predict`` so it is symmetric
    with ``_aspect_separated_knn``: the two are the same step and the self
    exclusion has to happen in both, which is easier to see and harder to
    forget when they look alike.
    """
    hits = search_knn(
        queries.embeddings,
        reference_embeddings,
        reference_accessions,
        k=cfg.k + self_margin(cfg, queries, reference_accessions),
        distance_threshold=cfg.distance_threshold,
        backend=cfg.backend,
        metric=cfg.metric,
        pre_normalized=cfg.pre_normalized,
    )
    hits = drop_own_sequence(cfg, hits, queries)
    return {"": hits}, {"": _build_go_map(hits, annotations)}


def aspect_separated_knn(
    *,
    queries: Queries,
    reference_accessions: list[str],
    reference_embeddings: np.ndarray,
    annotations: dict[str, list[dict[str, Any]]],
    go_aspect_map: dict[int, str],
    cfg: PredictConfig,
) -> tuple[
    dict[str, list[list[tuple[str, float]]]],
    dict[str, dict[str, list[dict[str, Any]]]],
]:
    """Three KNN searches, one per GO aspect, over aspect-filtered refs."""
    partitioned = _partition_refs_by_aspect(
        reference_accessions, reference_embeddings, annotations, go_aspect_map,
    )
    neighbors_by_aspect: dict[str, list[list[tuple[str, float]]]] = {}
    go_map_by_aspect: dict[str, dict[str, list[dict[str, Any]]]] = {}
    n_queries = queries.embeddings.shape[0]
    for aspect, (acc_subset, emb_subset) in partitioned.items():
        if not acc_subset:
            neighbors_by_aspect[aspect] = [[] for _ in range(n_queries)]
            go_map_by_aspect[aspect] = {}
            continue
        margin = self_margin(cfg, queries, acc_subset)
        hits = search_knn(
            queries.embeddings,
            emb_subset,
            acc_subset,
            k=cfg.k + margin,
            distance_threshold=cfg.distance_threshold,
            backend=cfg.backend,
            metric=cfg.metric,
            pre_normalized=cfg.pre_normalized,
        )
        hits = drop_own_sequence(cfg, hits, queries)
        neighbors_by_aspect[aspect] = hits
        go_map_by_aspect[aspect] = _build_go_map(hits, annotations)
    return neighbors_by_aspect, go_map_by_aspect
