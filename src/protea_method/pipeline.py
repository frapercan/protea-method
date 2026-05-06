"""End-to-end inference orchestrator.

Wires the F2C.1 reranker, F2C.2 feature enricher, and F2C.3 KNN
backend into the single ``predict`` function that LAFA-style
containers call. Pure: takes already-loaded inputs (query embeddings,
reference embeddings, annotations table, GO maps) and returns
prediction dicts. The platform (protea-core) or container caller is
responsible for materialising the inputs from a DB or bind-mounted
parquet files.

Two modes are supported via ``PredictConfig.aspect_separated``:

* unified KNN (default): a single KNN index across all reference
  embeddings, one search per query.
* aspect-separated KNN: three independent KNN indices (one per GO
  aspect P / F / C), each restricted to references that have at
  least one annotation in that aspect; results from all three are
  merged. This guarantees BPO / MFO / CCO candidates per query even
  when the globally-nearest neighbours happen to carry annotations
  in only one or two aspects (which is the dominant cause of the
  BPO recall ceiling on a unified index).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import lightgbm as lgb
import numpy as np

from protea_method.anc2vec import Anc2VecIndex
from protea_method.feature_enricher import ASPECT_CODES, enrich_v6_features
from protea_method.knn_search import search_knn
from protea_method.reranker import apply_reranker


@dataclass(frozen=True)
class PredictConfig:
    """Configuration for the ``predict`` orchestrator.

    All fields default to values that match PROTEA's production KNN
    behaviour so the LAFA-side caller can use the same defaults
    without re-specifying each field.

    Attributes
    ----------
    k:
        Maximum number of nearest neighbours to retrieve per query.
    metric:
        ``"cosine"`` or ``"l2"``. Distances are returned in the
        convention used by ``protea_method.knn_search`` (lower is
        more similar).
    backend:
        ``"numpy"`` or ``"faiss"``.
    distance_threshold:
        If set, drop neighbours with distance > threshold before
        accumulating votes.
    aspect_separated:
        Run one KNN per GO aspect (P, F, C). Reserved for a follow-up
        slice; the current implementation only supports the unified
        single-KNN path (``False``).
    compute_v6_features:
        Run the v6 feature enrichment pass (Anc2Vec centroids, PCA,
        tax voters). Disable to skip when a downstream consumer does
        not need them.
    compute_taxonomy:
        Forwarded to ``enrich_v6_features`` for the tax-voters
        family. Requires ``pair_features`` to be populated.
    pre_normalized:
        Reference embeddings are already L2-normalised. Skips the
        per-call normalisation in ``search_knn`` (cosine only).
    """

    k: int = 5
    metric: str = "cosine"
    backend: str = "numpy"
    distance_threshold: float | None = None
    aspect_separated: bool = False
    compute_v6_features: bool = True
    compute_taxonomy: bool = False
    pre_normalized: bool = False
    extra: dict[str, Any] = field(default_factory=dict)


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


def _aspect_separated_knn(
    *,
    query_embeddings: np.ndarray,
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
    n_queries = query_embeddings.shape[0]
    for aspect, (acc_subset, emb_subset) in partitioned.items():
        if not acc_subset:
            neighbors_by_aspect[aspect] = [[] for _ in range(n_queries)]
            go_map_by_aspect[aspect] = {}
            continue
        hits = search_knn(
            query_embeddings,
            emb_subset,
            acc_subset,
            k=cfg.k,
            distance_threshold=cfg.distance_threshold,
            backend=cfg.backend,
            metric=cfg.metric,
            pre_normalized=cfg.pre_normalized,
        )
        neighbors_by_aspect[aspect] = hits
        go_map_by_aspect[aspect] = _build_go_map(hits, annotations)
    return neighbors_by_aspect, go_map_by_aspect


def _accumulate_votes(
    *,
    query_accessions: list[str],
    neighbors_by_aspect: dict[str, list[list[tuple[str, float]]]],
    annotations: dict[str, list[dict[str, Any]]],
    go_aspect_map: dict[int, str],
    aspect_separated: bool,
) -> list[dict[str, Any]]:
    """Build base prediction dicts (protein, go_term, vote_count, distances).

    In aspect-separated mode, only annotations matching the current
    KNN's aspect contribute to that aspect's votes; this prevents an
    aspect-F neighbour's P-annotations from leaking into the F vote
    pool.
    """
    predictions: list[dict[str, Any]] = []
    for q_idx, q_acc in enumerate(query_accessions):
        votes: dict[int, dict[str, float]] = {}
        for aspect_key, neighbors_per_query in neighbors_by_aspect.items():
            if q_idx >= len(neighbors_per_query):
                continue
            for ref_acc, distance in neighbors_per_query[q_idx]:
                for ann in annotations.get(ref_acc, []):
                    gtid = int(ann["go_term_id"])
                    if aspect_separated:
                        ann_aspect = go_aspect_map.get(gtid, "")
                        if ann_aspect != aspect_key:
                            continue
                    stat = votes.setdefault(
                        gtid,
                        {"vote_count": 0.0, "sum_d": 0.0, "min_d": float("inf")},
                    )
                    stat["vote_count"] += 1.0
                    stat["sum_d"] += float(distance)
                    if float(distance) < stat["min_d"]:
                        stat["min_d"] = float(distance)
        for gtid, stat in votes.items():
            predictions.append(
                {
                    "protein_accession": q_acc,
                    "go_term_id": gtid,
                    "vote_count": stat["vote_count"],
                    "min_distance": stat["min_d"],
                    "mean_distance": stat["sum_d"] / stat["vote_count"],
                    "distance": stat["min_d"],
                    "aspect": go_aspect_map.get(gtid, ""),
                },
            )
    return predictions


def predict(
    *,
    query_accessions: list[str],
    query_embeddings: np.ndarray,
    reference_accessions: list[str],
    reference_embeddings: np.ndarray,
    annotations: dict[str, list[dict[str, Any]]],
    go_id_map: dict[int, str],
    go_aspect_map: dict[int, str],
    config: PredictConfig | None = None,
    pca_state: tuple[np.ndarray, np.ndarray] | None = None,
    pair_features: dict[tuple[str, str], dict[str, Any]] | None = None,
    booster: lgb.Booster | None = None,
    reranker_feature_cols: list[str] | None = None,
    anc_idx: Anc2VecIndex | None = None,
) -> list[dict[str, Any]]:
    """End-to-end inference: KNN, base features, v6 features, optional reranker.

    Returns a flat list of prediction dicts. Each dict has at least:

    - ``protein_accession`` (query)
    - ``go_term_id`` (the candidate GO term id, integer)
    - ``vote_count`` (number of neighbours that voted for this term)
    - ``min_distance`` (smallest distance among voting neighbours)
    - ``mean_distance`` (mean distance among voting neighbours)
    - the v6 feature columns from ``enrich_v6_features`` when
      ``config.compute_v6_features`` is true
    - ``reranker_score`` when a ``booster`` is provided (in the
      [0, 1] range)

    ``annotations`` maps each reference accession to its list of GO
    annotations. Each annotation dict must contain ``go_term_id`` and
    may carry additional metadata that flows through to the v6
    enricher.

    ``go_id_map`` and ``go_aspect_map`` are the metadata maps loaded
    by the caller (DB query in PROTEA, bind-mounted parquet in the
    LAFA container).
    """
    cfg = config or PredictConfig()

    if not query_accessions or query_embeddings.size == 0:
        return []
    if reference_embeddings.size == 0:
        return []

    if cfg.aspect_separated:
        neighbors_by_aspect, go_map_by_aspect = _aspect_separated_knn(
            query_embeddings=query_embeddings,
            reference_accessions=reference_accessions,
            reference_embeddings=reference_embeddings,
            annotations=annotations,
            go_aspect_map=go_aspect_map,
            cfg=cfg,
        )
    else:
        neighbors_per_query = search_knn(
            query_embeddings,
            reference_embeddings,
            reference_accessions,
            k=cfg.k,
            distance_threshold=cfg.distance_threshold,
            backend=cfg.backend,
            metric=cfg.metric,
            pre_normalized=cfg.pre_normalized,
        )
        neighbors_by_aspect = {"": neighbors_per_query}
        go_map_by_aspect = {"": _build_go_map(neighbors_per_query, annotations)}

    predictions = _accumulate_votes(
        query_accessions=query_accessions,
        neighbors_by_aspect=neighbors_by_aspect,
        annotations=annotations,
        go_aspect_map=go_aspect_map,
        aspect_separated=cfg.aspect_separated,
    )

    if cfg.compute_v6_features and predictions:
        enrich_v6_features(
            predictions,
            go_id_map=go_id_map,
            go_aspect_map=go_aspect_map,
            valid_accessions=query_accessions,
            query_embeddings=query_embeddings,
            neighbors_by_aspect=neighbors_by_aspect,
            go_map_by_aspect=go_map_by_aspect,
            pair_features=pair_features or {},
            pca_state=pca_state,
            compute_taxonomy=cfg.compute_taxonomy,
            anc_idx=anc_idx,
        )

    if booster is not None and predictions:
        import pandas as pd

        df = pd.DataFrame(predictions)
        scores = apply_reranker(df, booster, feature_cols=reranker_feature_cols)
        for pred, score in zip(predictions, scores, strict=True):
            pred["reranker_score"] = float(score)

    return predictions


__all__ = ["PredictConfig", "predict"]
