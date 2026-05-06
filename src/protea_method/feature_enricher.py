"""V6 feature enrichment for GO predictions (pure-compute path).

The "v6" feature family is the set of 25 columns that the post-2026
LightGBM re-ranker consumes on top of the base KNN features. Computing
them requires three independent intermediates over the prediction
batch:

* an Anc2Vec embedding pool covering every GO term seen as either a
  candidate or a voting-neighbor annotation,
* per-(query, aspect) neighbor centroids derived from that pool,
* per-(query, candidate) tax-voter counters built from the pair
  features produced by the alignment / taxonomy pipeline.

A fourth intermediate, the PCA projection of query embeddings, is
optional. It is present only when the caller passes a fitted
``pca_state``.

The orchestrator :func:`enrich_v6_features` runs the four stages,
then walks the prediction list and merges each candidate's features
in place. It takes ``go_id_map`` and ``go_aspect_map`` as parameters,
both already loaded by the caller (from a DB session, a parquet
file, etc.). This is the boundary that keeps the module pure: no
SQLAlchemy imports, no protea-core dependency.

Original file: PROTEA's ``protea/core/feature_enricher.py``. The
module-private helpers are unchanged in shape; only ``enrich_v6_features``
has a different signature (Session is gone).
"""

from __future__ import annotations

from typing import Any

import numpy as np

from protea_method.anc2vec import Anc2VecIndex
from protea_method.anc2vec import get_index as get_anc2vec_index
from protea_method.reranker import EMBEDDING_PCA_DIM

#: Aspect single-letter codes (F, P, C). Inlined from PROTEA's
#: ``protea.core.domain.aspect`` so the module stays free of the Aspect
#: enum and the larger domain layer.
ASPECT_CODES: tuple[str, str, str] = ("F", "P", "C")

_TAX_CLOSE_RELATIONS = frozenset(
    {"same", "ancestor", "descendant", "child", "parent", "close"}
)

#: Feature columns the v6 enricher writes into each prediction dict.
NEW_V6_FEATURE_KEYS: tuple[str, ...] = (
    "anc2vec_neighbor_cos",
    "anc2vec_neighbor_maxcos",
    "anc2vec_has_emb",
    "anc2vec_query_known_cos",
    "anc2vec_query_known_maxcos",
    "anc2vec_query_known_count",
    "tax_voters_same_frac",
    "tax_voters_close_frac",
    "tax_voters_mean_common_ancestors",
    *(f"emb_pca_query_{i}" for i in range(EMBEDDING_PCA_DIM)),
)


def _collect_gtids_in_play(
    predictions: list[dict[str, Any]],
    go_map_by_aspect: dict[str, dict[str, list[dict[str, Any]]]],
) -> set[int]:
    """Every go_term_id that appears either as a candidate in
    predictions or as an annotation of a voting neighbor; both are
    needed to compute neighbor Anc2Vec centroids.
    """
    gtids: set[int] = {int(pred["go_term_id"]) for pred in predictions}
    for go_map in go_map_by_aspect.values():
        for anns in go_map.values():
            for ann in anns:
                gtids.add(int(ann["go_term_id"]))
    return gtids


def _build_anc2vec_pool(
    gtids_in_play: set[int],
    go_id_map: dict[int, str],
    *,
    anc_idx: Anc2VecIndex | None = None,
) -> tuple[dict[str, int], np.ndarray, np.ndarray]:
    """Materialise the Anc2Vec embedding matrix for every GO id seen.

    Returns ``(idx_of_go, all_norm, has_emb_mask)`` where ``all_norm``
    is the L2-normalised projection of every GO id (rows lacking an
    Anc2Vec vector are zeroed and flagged in ``has_emb_mask``).

    ``anc_idx`` is injectable for testing; defaults to the
    process-wide singleton from ``protea_method.anc2vec.get_index``.
    """
    idx = anc_idx if anc_idx is not None else get_anc2vec_index()
    all_go_id_strs = {go_id_map[gid] for gid in gtids_in_play if gid in go_id_map}
    all_go_id_list = sorted(all_go_id_strs)
    idx_of_go: dict[str, int] = {g: i for i, g in enumerate(all_go_id_list)}
    all_emb = idx.batch(all_go_id_list)
    raw_norms = np.linalg.norm(all_emb, axis=1)
    has_emb_mask = raw_norms > 0.0
    safe_norms = np.where(has_emb_mask, raw_norms, 1.0)[:, None]
    all_norm = (all_emb / safe_norms).astype(np.float32)
    all_norm[~has_emb_mask] = 0.0
    return idx_of_go, all_norm, has_emb_mask


def _compute_neighbor_centroids(
    *,
    valid_accessions: list[str],
    neighbors_by_aspect: dict[str, list[list[tuple[str, float]]]],
    go_map_by_aspect: dict[str, dict[str, list[dict[str, Any]]]],
    go_id_map: dict[int, str],
    go_aspect_map: dict[int, str],
    idx_of_go: dict[str, int],
    all_norm: np.ndarray,
    has_emb_mask: np.ndarray,
) -> dict[tuple[str, str], tuple[np.ndarray | None, np.ndarray | None]]:
    """Per ``(q_acc, aspect)`` pair, compute the Anc2Vec centroid of the
    neighbor-side annotations. Returns ``(centroid_unit, nmat)`` where
    ``nmat`` is the matrix of contributing neighbor vectors and
    ``centroid_unit`` is its L2-normalised mean.

    The aspect partition matches training: each centroid only mixes
    neighbor terms from the same GO aspect as the candidate.
    """
    info: dict[tuple[str, str], tuple[np.ndarray | None, np.ndarray | None]] = {}
    for aspect_key, nbs_all in neighbors_by_aspect.items():
        go_map = go_map_by_aspect.get(aspect_key, {})
        for q_idx, q_acc in enumerate(valid_accessions):
            if q_idx >= len(nbs_all):
                continue
            per_asp_rows: dict[str, list[int]] = {a: [] for a in ASPECT_CODES}
            per_asp_seen: dict[str, set[str]] = {a: set() for a in ASPECT_CODES}
            for ref_acc, _ in nbs_all[q_idx]:
                for ann in go_map.get(ref_acc, []):
                    gtid = int(ann["go_term_id"])
                    gid_str = go_id_map.get(gtid)
                    asp = go_aspect_map.get(gtid, "")
                    if not gid_str or asp not in per_asp_rows:
                        continue
                    if gid_str in per_asp_seen[asp]:
                        continue
                    per_asp_seen[asp].add(gid_str)
                    i = idx_of_go.get(gid_str)
                    if i is not None and has_emb_mask[i]:
                        per_asp_rows[asp].append(i)
            for asp, rows in per_asp_rows.items():
                if not rows:
                    info.setdefault((q_acc, asp), (None, None))
                    continue
                nmat = all_norm[rows]
                centroid = nmat.mean(axis=0)
                cn = float(np.linalg.norm(centroid))
                centroid_unit = (
                    (centroid / cn).astype(np.float32) if cn > 0.0 else None
                )
                prev = info.get((q_acc, asp))
                if prev is None or prev == (None, None):
                    info[(q_acc, asp)] = (centroid_unit, nmat)
    return info


def _compute_tax_voter_counters(
    *,
    valid_accessions: list[str],
    neighbors_by_aspect: dict[str, list[list[tuple[str, float]]]],
    go_map_by_aspect: dict[str, dict[str, list[dict[str, Any]]]],
    pair_features: dict[tuple[str, str], dict[str, Any]],
    compute_taxonomy: bool,
) -> tuple[
    dict[str, dict[int, int]],
    dict[str, dict[int, int]],
    dict[str, dict[int, float]],
    dict[str, dict[int, int]],
    dict[str, dict[int, int]],
]:
    """Build the five per-(q_acc, gtid) counters that feed the
    tax_voters_* feature family.

    Returns ``(same_cnt, close_cnt, ca_sum, ca_n, vote_count_div)``.
    When ``compute_taxonomy`` is False, every dict comes back empty
    (the merge loop emits NaN for the corresponding columns).
    """
    same_cnt: dict[str, dict[int, int]] = {}
    close_cnt: dict[str, dict[int, int]] = {}
    ca_sum: dict[str, dict[int, float]] = {}
    ca_n: dict[str, dict[int, int]] = {}
    vc_div: dict[str, dict[int, int]] = {}
    if not compute_taxonomy:
        return same_cnt, close_cnt, ca_sum, ca_n, vc_div

    for aspect_key, nbs_all in neighbors_by_aspect.items():
        go_map = go_map_by_aspect.get(aspect_key, {})
        for q_idx, q_acc in enumerate(valid_accessions):
            if q_idx >= len(nbs_all):
                continue
            same_d = same_cnt.setdefault(q_acc, {})
            close_d = close_cnt.setdefault(q_acc, {})
            sum_d = ca_sum.setdefault(q_acc, {})
            n_d = ca_n.setdefault(q_acc, {})
            vc_d = vc_div.setdefault(q_acc, {})
            for ref_acc, _ in nbs_all[q_idx]:
                pf = pair_features.get((q_acc, ref_acc), {})
                rel = pf.get("taxonomic_relation") or ""
                ca = pf.get("taxonomic_common_ancestors")
                is_same = rel == "same"
                is_close = rel in _TAX_CLOSE_RELATIONS
                for ann in go_map.get(ref_acc, []):
                    gtid = int(ann["go_term_id"])
                    vc_d[gtid] = vc_d.get(gtid, 0) + 1
                    if is_same:
                        same_d[gtid] = same_d.get(gtid, 0) + 1
                    if is_close:
                        close_d[gtid] = close_d.get(gtid, 0) + 1
                    if isinstance(ca, int | float) and ca is not None:
                        sum_d[gtid] = sum_d.get(gtid, 0.0) + float(ca)
                        n_d[gtid] = n_d.get(gtid, 0) + 1
    return same_cnt, close_cnt, ca_sum, ca_n, vc_div


def _compute_pca_projection(
    query_embeddings: np.ndarray,
    pca_state: tuple[np.ndarray, np.ndarray] | None,
) -> np.ndarray | None:
    """Project query embeddings into the PCA reference subspace.

    Returns ``None`` when the projection cannot be computed (no PCA
    state or no query embeddings); the merge loop emits NaN for the
    ``emb_pca_query_*`` columns in that case.
    """
    if pca_state is None or not query_embeddings.size:
        return None
    pca_mean, pca_components = pca_state
    proj = (query_embeddings.astype(np.float32) - pca_mean) @ pca_components.T
    return np.asarray(proj, dtype=np.float32)


def enrich_v6_features(
    predictions: list[dict[str, Any]],
    *,
    go_id_map: dict[int, str],
    go_aspect_map: dict[int, str],
    valid_accessions: list[str],
    query_embeddings: np.ndarray,
    neighbors_by_aspect: dict[str, list[list[tuple[str, float]]]],
    go_map_by_aspect: dict[str, dict[str, list[dict[str, Any]]]],
    pair_features: dict[tuple[str, str], dict[str, Any]],
    pca_state: tuple[np.ndarray, np.ndarray] | None,
    compute_taxonomy: bool,
    anc_idx: Anc2VecIndex | None = None,
) -> None:
    """Compute the 25 v6 features and merge them into each ``pred``
    dict in place.

    ``go_id_map`` and ``go_aspect_map`` are the two metadata maps the
    caller pre-loaded (from a DB query in PROTEA, or from a parquet
    table in the LAFA container). The library does not own the
    transport.

    ``neighbors_by_aspect`` / ``go_map_by_aspect`` may contain a
    single aspect key (``""``) in unified-KNN mode; the per-aspect
    groupings of candidate GO terms are resolved via the GO term
    aspect map.

    ``pair_features`` carries the ``taxonomic_relation`` and
    ``taxonomic_common_ancestors`` keys populated by
    ``compute_taxonomy``. When ``compute_taxonomy`` is ``False`` the
    tax_voters family stays NaN.

    Query-known Anc2Vec features are emitted as NaN / 0 at predict
    time; ``anc2vec_query_known_*`` is resolved at eval time from
    the split's pre-cutoff annotations (LK / PK only).

    The body is a five-stage pipeline (Anc2Vec pool, neighbor
    centroids, tax-voter counters, PCA projection, merge); each
    stage is its own private helper.

    ``anc_idx`` is injectable for testing; defaults to the
    process-wide singleton.
    """
    if not predictions:
        return

    acc_to_idx = {acc: i for i, acc in enumerate(valid_accessions)}

    gtids_in_play = _collect_gtids_in_play(predictions, go_map_by_aspect)

    idx_of_go, all_norm, has_emb_mask = _build_anc2vec_pool(
        gtids_in_play, go_id_map, anc_idx=anc_idx,
    )

    neighbor_info = _compute_neighbor_centroids(
        valid_accessions=valid_accessions,
        neighbors_by_aspect=neighbors_by_aspect,
        go_map_by_aspect=go_map_by_aspect,
        go_id_map=go_id_map,
        go_aspect_map=go_aspect_map,
        idx_of_go=idx_of_go,
        all_norm=all_norm,
        has_emb_mask=has_emb_mask,
    )

    tax_same_cnt, tax_close_cnt, tax_ca_sum, tax_ca_n, vote_count_div = (
        _compute_tax_voter_counters(
            valid_accessions=valid_accessions,
            neighbors_by_aspect=neighbors_by_aspect,
            go_map_by_aspect=go_map_by_aspect,
            pair_features=pair_features,
            compute_taxonomy=compute_taxonomy,
        )
    )

    pca_query_proj = _compute_pca_projection(query_embeddings, pca_state)
    nan_pca = [float("nan")] * EMBEDDING_PCA_DIM

    for pred in predictions:
        q_acc = pred["protein_accession"]
        gtid = int(pred["go_term_id"])
        go_id = go_id_map.get(gtid)
        aspect = go_aspect_map.get(gtid, "")

        cand_i = idx_of_go.get(go_id, -1) if go_id else -1
        if cand_i >= 0 and has_emb_mask[cand_i]:
            cand_vec = all_norm[cand_i]
            centroid_unit, nmat = neighbor_info.get((q_acc, aspect), (None, None))
            anc_cos = (
                float(cand_vec @ centroid_unit)
                if centroid_unit is not None
                else float("nan")
            )
            anc_maxcos = (
                float((nmat @ cand_vec).max()) if nmat is not None else float("nan")
            )
            anc_has = 1.0
        else:
            anc_cos = float("nan")
            anc_maxcos = float("nan")
            anc_has = 0.0

        pred["anc2vec_neighbor_cos"] = anc_cos
        pred["anc2vec_neighbor_maxcos"] = anc_maxcos
        pred["anc2vec_has_emb"] = anc_has
        pred["anc2vec_query_known_cos"] = float("nan")
        pred["anc2vec_query_known_maxcos"] = float("nan")
        pred["anc2vec_query_known_count"] = 0.0

        if compute_taxonomy:
            vc_total = max(1, vote_count_div.get(q_acc, {}).get(gtid, 1))
            pred["tax_voters_same_frac"] = (
                tax_same_cnt.get(q_acc, {}).get(gtid, 0) / vc_total
            )
            pred["tax_voters_close_frac"] = (
                tax_close_cnt.get(q_acc, {}).get(gtid, 0) / vc_total
            )
            ca_n = tax_ca_n.get(q_acc, {}).get(gtid, 0)
            pred["tax_voters_mean_common_ancestors"] = (
                tax_ca_sum.get(q_acc, {}).get(gtid, 0.0) / max(1, ca_n)
                if ca_n > 0
                else float("nan")
            )
        else:
            pred["tax_voters_same_frac"] = float("nan")
            pred["tax_voters_close_frac"] = float("nan")
            pred["tax_voters_mean_common_ancestors"] = float("nan")

        q_idx = acc_to_idx.get(q_acc, -1)
        if pca_query_proj is not None and 0 <= q_idx < pca_query_proj.shape[0]:
            row = pca_query_proj[q_idx]
            for i in range(EMBEDDING_PCA_DIM):
                pred[f"emb_pca_query_{i}"] = float(row[i])
        else:
            for i in range(EMBEDDING_PCA_DIM):
                pred[f"emb_pca_query_{i}"] = nan_pca[i]


__all__ = [
    "ASPECT_CODES",
    "NEW_V6_FEATURE_KEYS",
    "enrich_v6_features",
]
