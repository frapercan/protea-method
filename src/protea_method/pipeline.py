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
    prediction_set_id:
        Free-form provenance string PROTEA forwards from the
        ``PredictionSet`` row id. When given it is copied onto every
        emitted row so the lab dump and the live pipeline produce
        identical schemas.
    """

    k: int = 5
    metric: str = "cosine"
    backend: str = "numpy"
    distance_threshold: float | None = None
    aspect_separated: bool = False
    compute_v6_features: bool = True
    compute_taxonomy: bool = False
    pre_normalized: bool = False
    prediction_set_id: str | None = None
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


def _annotation_aggregates(
    annotations: dict[str, list[dict[str, Any]]],
) -> tuple[dict[int, int], dict[str, int]]:
    """Pre-compute ``(go_term_frequency, ref_annotation_density)``.

    Both are dataset-wide aggregates independent of the query batch,
    so they are computed once before the per-query loop.
    """
    go_term_freq: dict[int, int] = {}
    ref_ann_density: dict[str, int] = {}
    for ref_acc, anns in annotations.items():
        if not anns:
            continue
        ref_ann_density[ref_acc] = len(anns)
        for ann in anns:
            gtid = int(ann["go_term_id"])
            go_term_freq[gtid] = go_term_freq.get(gtid, 0) + 1
    return go_term_freq, ref_ann_density


def _collect_query_distances(
    q_idx: int,
    neighbors_by_aspect: dict[str, list[list[tuple[str, float]]]],
) -> list[float]:
    """Flatten the KNN distances of one query across all aspect indices."""
    out: list[float] = []
    for neighbors_per_query in neighbors_by_aspect.values():
        if q_idx < len(neighbors_per_query):
            out.extend(float(d) for _, d in neighbors_per_query[q_idx])
    return out


def _tally_query_votes(
    *,
    q_idx: int,
    neighbors_by_aspect: dict[str, list[list[tuple[str, float]]]],
    annotations: dict[str, list[dict[str, Any]]],
    go_aspect_map: dict[int, str],
    aspect_separated: bool,
) -> dict[int, dict[str, Any]]:
    """Run the vote-tally for one query and return per-(go_term) stats."""
    votes: dict[int, dict[str, Any]] = {}
    for aspect_key, neighbors_per_query in neighbors_by_aspect.items():
        if q_idx >= len(neighbors_per_query):
            continue
        for k_pos, (ref_acc, distance) in enumerate(
            neighbors_per_query[q_idx], start=1,
        ):
            d = float(distance)
            for ann in annotations.get(ref_acc, []):
                gtid = int(ann["go_term_id"])
                if aspect_separated:
                    if go_aspect_map.get(gtid, "") != aspect_key:
                        continue
                stat = votes.get(gtid)
                if stat is None:
                    stat = {
                        "vote_count": 0,
                        "sum_d": 0.0,
                        "min_d": d,
                        "donor_ref": ref_acc,
                        "donor_ann": ann,
                        "k_position": k_pos,
                    }
                    votes[gtid] = stat
                stat["vote_count"] += 1
                stat["sum_d"] += d
                if d < stat["min_d"]:
                    stat["min_d"] = d
                    stat["donor_ref"] = ref_acc
                    stat["donor_ann"] = ann
    return votes


@dataclass(frozen=True)
class _RowContext:
    """Static state shared by every ``(query, go_term)`` row of one batch."""

    go_id_map: dict[int, str]
    go_aspect_map: dict[int, str]
    go_term_freq: dict[int, int]
    ref_ann_density: dict[str, int]
    pair_features: dict[tuple[str, str], dict[str, Any]]
    k_div: float
    prediction_set_id: str | None


def _make_row(
    q_acc: str,
    gtid: int,
    stat: dict[str, Any],
    distance_std: float,
    ctx: _RowContext,
) -> dict[str, Any]:
    """Build one PROTEA-shaped prediction row from a tally stat dict."""
    vote_count = int(stat["vote_count"])
    mean_d = stat["sum_d"] / vote_count
    donor_ref = str(stat["donor_ref"])
    donor_ann = stat["donor_ann"]
    row: dict[str, Any] = {
        "protein_accession": q_acc,
        "go_term_id": gtid,
        "vote_count": vote_count,
        "min_distance": stat["min_d"],
        "mean_distance": mean_d,
        "distance": stat["min_d"],
        "aspect": ctx.go_aspect_map.get(gtid, ""),
        "ref_protein_accession": donor_ref,
        "qualifier": donor_ann.get("qualifier") or "",
        "evidence_code": donor_ann.get("evidence_code") or "",
        "k_position": int(stat["k_position"]),
        "go_term_frequency": ctx.go_term_freq.get(gtid, 0),
        "ref_annotation_density": ctx.ref_ann_density.get(donor_ref, 0),
        "neighbor_distance_std": distance_std,
        "neighbor_vote_fraction": vote_count / ctx.k_div,
        "neighbor_min_distance": stat["min_d"],
        "neighbor_mean_distance": mean_d,
    }
    go_id = ctx.go_id_map.get(gtid)
    if go_id is not None:
        row["go_id"] = go_id
    if ctx.prediction_set_id is not None:
        row["prediction_set_id"] = ctx.prediction_set_id
    pf = ctx.pair_features.get((q_acc, donor_ref), {})
    if pf:
        _propagate_pair_features(row, pf)
    return row


def _accumulate_votes(
    *,
    query_accessions: list[str],
    neighbors_by_aspect: dict[str, list[list[tuple[str, float]]]],
    annotations: dict[str, list[dict[str, Any]]],
    ctx: _RowContext,
    aspect_separated: bool,
) -> list[dict[str, Any]]:
    """Build PROTEA-compatible prediction dicts with reranker aggregates.

    See ``predict`` for the row shape. The function delegates the
    per-query vote tally to ``_tally_query_votes`` and the row
    materialisation to ``_make_row``; this orchestrator just walks
    queries.
    """
    predictions: list[dict[str, Any]] = []
    for q_idx, q_acc in enumerate(query_accessions):
        dists = _collect_query_distances(q_idx, neighbors_by_aspect)
        distance_std = float(np.std(dists)) if len(dists) > 1 else 0.0
        votes = _tally_query_votes(
            q_idx=q_idx,
            neighbors_by_aspect=neighbors_by_aspect,
            annotations=annotations,
            go_aspect_map=ctx.go_aspect_map,
            aspect_separated=aspect_separated,
        )
        for gtid, stat in votes.items():
            predictions.append(_make_row(q_acc, gtid, stat, distance_std, ctx))
    return predictions


def _build_row_context(
    *,
    cfg: PredictConfig,
    annotations: dict[str, list[dict[str, Any]]],
    go_id_map: dict[int, str],
    go_aspect_map: dict[int, str],
    pair_features: dict[tuple[str, str], dict[str, Any]] | None,
) -> _RowContext:
    """Assemble the static row context for one ``predict`` invocation."""
    go_term_freq, ref_ann_density = _annotation_aggregates(annotations)
    return _RowContext(
        go_id_map=go_id_map,
        go_aspect_map=go_aspect_map,
        go_term_freq=go_term_freq,
        ref_ann_density=ref_ann_density,
        pair_features=pair_features or {},
        k_div=float(max(1, cfg.k)),
        prediction_set_id=cfg.prediction_set_id,
    )


_PAIR_FEATURE_KEYS: tuple[str, ...] = (
    "identity_nw",
    "similarity_nw",
    "alignment_score_nw",
    "gaps_pct_nw",
    "alignment_length_nw",
    "identity_sw",
    "similarity_sw",
    "alignment_score_sw",
    "gaps_pct_sw",
    "alignment_length_sw",
    "length_query",
    "length_ref",
    "taxonomic_distance",
    "taxonomic_common_ancestors",
    "taxonomic_relation",
)


def _propagate_pair_features(
    row: dict[str, Any], pf: dict[str, Any],
) -> None:
    """Copy the alignment / taxonomy fields the lab schema expects."""
    for key in _PAIR_FEATURE_KEYS:
        if key in pf:
            row[key] = pf[key]


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
    boosters_by_aspect: dict[str, lgb.Booster] | None = None,
    reranker_feature_cols: list[str] | None = None,
    anc_idx: Anc2VecIndex | None = None,
) -> list[dict[str, Any]]:
    """End-to-end inference. See module docstring for the row shape.

    Returns PROTEA-compatible prediction dicts. Each row carries
    identity (``protein_accession``, ``go_term_id``, ``go_id``,
    ``aspect``, ``ref_protein_accession``, donor ``qualifier`` /
    ``evidence_code``, optional ``prediction_set_id``) and the
    reranker-feature aggregates ``vote_count``, ``k_position``,
    ``go_term_frequency``, ``ref_annotation_density``,
    ``neighbor_distance_std``, ``neighbor_vote_fraction``,
    ``neighbor_min_distance``, ``neighbor_mean_distance``. Legacy
    aliases ``min_distance`` / ``mean_distance`` / ``distance`` are
    preserved. Alignment / taxonomy fields are merged from
    ``pair_features[(query, donor_ref)]``. v6 features and
    ``reranker_score`` are appended when their respective inputs are
    provided.

    Reranker routing: ``boosters_by_aspect`` (per-aspect models) wins
    over ``booster`` (single model) when both are given; aspects
    without an entry stay unscored.
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

    row_ctx = _build_row_context(
        cfg=cfg,
        annotations=annotations,
        go_id_map=go_id_map,
        go_aspect_map=go_aspect_map,
        pair_features=pair_features,
    )
    predictions = _accumulate_votes(
        query_accessions=query_accessions,
        neighbors_by_aspect=neighbors_by_aspect,
        annotations=annotations,
        ctx=row_ctx,
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

    if predictions:
        if boosters_by_aspect:
            _score_per_aspect(
                predictions, boosters_by_aspect, reranker_feature_cols,
            )
        elif booster is not None:
            _score_single(predictions, booster, reranker_feature_cols)

    return predictions


def _score_single(
    predictions: list[dict[str, Any]],
    booster: lgb.Booster,
    feature_cols: list[str] | None,
) -> None:
    """Score every prediction with a single booster (legacy path)."""
    import pandas as pd

    df = pd.DataFrame(predictions)
    scores = apply_reranker(df, booster, feature_cols=feature_cols)
    for pred, score in zip(predictions, scores, strict=True):
        pred["reranker_score"] = float(score)


def _score_per_aspect(
    predictions: list[dict[str, Any]],
    boosters: dict[str, lgb.Booster],
    feature_cols: list[str] | None,
) -> None:
    """Score predictions by aspect-specific boosters.

    Predictions whose ``aspect`` field has no entry in ``boosters``
    are left without a ``reranker_score`` (caller falls back to
    distance-based ordering for those rows).
    """
    import pandas as pd

    by_aspect: dict[str, list[int]] = {}
    for idx, pred in enumerate(predictions):
        aspect = str(pred.get("aspect", ""))
        if aspect in boosters:
            by_aspect.setdefault(aspect, []).append(idx)

    for aspect, indices in by_aspect.items():
        subset = [predictions[i] for i in indices]
        df = pd.DataFrame(subset)
        scores = apply_reranker(df, boosters[aspect], feature_cols=feature_cols)
        for i, score in zip(indices, scores, strict=True):
            predictions[i]["reranker_score"] = float(score)


__all__ = ["PredictConfig", "predict"]
