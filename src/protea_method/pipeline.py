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

from collections.abc import Mapping
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal, overload

import lightgbm as lgb
import numpy as np

from protea_method._donor_ledger import DonorLedger
from protea_method._knn_step import (
    Queries,
    aspect_separated_knn,
    unified_knn,
)
from protea_method._sequence_depth import dense_sequence_ranks
from protea_method.anc2vec import Anc2VecIndex
from protea_method.feature_enricher import ASPECT_CODES, enrich_v6_features
from protea_method.reranker import apply_reranker

# Long-form aspect names accepted by ``load_boosters_by_aspect``. Maps to
# the short ``F`` / ``P`` / ``C`` codes the rest of the pipeline uses.
_ASPECT_ALIASES: dict[str, str] = {
    "F": "F",
    "P": "P",
    "C": "C",
    "mfo": "F",
    "bpo": "P",
    "cco": "C",
    "MFO": "F",
    "BPO": "P",
    "CCO": "C",
}


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
        Run one KNN per GO aspect (``F``, ``P``, ``C``) over
        aspect-filtered reference pools and union the per-aspect
        candidate sets before scoring. Mirrors the per-aspect search
        the lab champion config (``bench-v1-K5-v226-lineage``,
        selective avg 0.6215 on v226) trained against; pair with
        ``boosters_by_aspect`` to reproduce the selective-rerank
        path. When ``False`` the orchestrator runs a single unified
        KNN across all reference embeddings.
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
    exclude_self_neighbour:
        Drop from each query's neighbourhood every reference carrying the
        query's OWN sequence. By sequence and not by accession: 38,694
        sequences in this bank belong to more than one protein, so an
        accession filter removes the query's identity and leaves its
        similarity, a copy of it at distance zero under another name.
        Needs ``sequence_keys`` to cover the queries as well as the bank,
        and raises rather than passing a query it cannot recognise. The
        search asks for extra neighbours first, measured from the bank,
        so the drop does not cost depth.
    sequence_keys:
        Accession to sequence identity for the reference bank. When
        given, every row carries a ``sequence_rank`` numbering its
        query's neighbour list by distinct sequence rather than by
        protein, so a cut at depth d admits d sequences. Forwarded
        here rather than as a kwarg, as ``prediction_set_id`` is, to
        keep ``predict``'s signature from growing further. A neighbour
        absent from the map raises; the whole map absent leaves the
        column empty on every row.
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
    sequence_keys: Mapping[str, str] | None = None
    exclude_self_neighbour: bool = False
    extra: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class PredictDiagnostics:
    """Intermediate KNN state exposed alongside ``predict()`` output.

    Returned only when callers pass ``return_diagnostics=True`` to
    :func:`predict`. Lets callers layer their own per-(query, gtid)
    derived features (rank, neighbor distance distribution, per-ref
    annotation density, …) on top of the base prediction shape
    without re-running the KNN.

    Attributes
    ----------
    neighbors_by_aspect:
        Per-aspect neighbour lists. Single-key ``{"": [...]}`` when
        ``PredictConfig.aspect_separated=False``; per-aspect keys
        ``"P" / "F" / "C"`` otherwise. Each value is a list of
        ``[(ref_acc, distance), ...]`` per query, in rank order.
    go_map_by_aspect:
        Per-aspect ``ref_acc -> annotation list`` map for every
        neighbour seen in ``neighbors_by_aspect``. Same key shape.
    pair_feature_misses:
        Every ``(query, donor)`` pair that a row asked ``pair_features``
        for and did not find. Empty when the caller passed no
        ``pair_features`` at all, because then nothing was asked.

        This is not bookkeeping. A caller that computes ``pair_features``
        from its own neighbour search, and lets ``predict`` run its own,
        is relying on the two agreeing. They agree almost always and not
        exactly: two searches over differently sliced float arrays
        disagree at the 1e-7 level, so wherever the k-th distance is a
        tie they can keep different donors. The rows for those donors
        came out with every pair-feature column NULL and said nothing,
        which is how 76 rows of a 2.4 million row run went unexplained
        for a day.
    """

    neighbors_by_aspect: dict[str, list[list[tuple[str, float]]]]
    go_map_by_aspect: dict[str, dict[str, list[dict[str, Any]]]]
    pair_feature_misses: frozenset[tuple[str, str]] = frozenset()


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


def _fresh_stat(
    ref_acc: str,
    ann: dict[str, Any],
    k_pos: int,
    seq_rank: int | None,
    d: float,
) -> dict[str, Any]:
    """The stat a term gets on its first sighting, from its shallowest donor.

    Everything here is either that donor's identity or an accumulator
    seeded from it. ``k_position`` and ``sequence_rank`` stay as they are
    set here: a term's depth is where it first became reachable, and a
    later, further donor does not move that.
    """
    return {
        "ledger": DonorLedger(),
        "vote_count": 0,
        "sum_d": 0.0,
        "min_d": d,
        "donor_ref": ref_acc,
        "donor_ann": ann,
        "k_position": k_pos,
        "sequence_rank": seq_rank,
    }


def _tally_query_votes(
    *,
    q_idx: int,
    neighbors_by_aspect: dict[str, list[list[tuple[str, float]]]],
    annotations: dict[str, list[dict[str, Any]]],
    go_aspect_map: dict[int, str],
    aspect_separated: bool,
    sequence_keys: Mapping[str, str] | None = None,
) -> dict[int, dict[str, Any]]:
    """Run the vote-tally for one query and return per-(go_term) stats.

    ``k_position`` numbers the neighbour list by protein. When
    ``sequence_keys`` is given, ``sequence_rank`` numbers the same list
    by distinct sequence, so a later cut at a depth admits that many
    sequences rather than that many rows. Both are taken at the term's
    first appearance, which is its shallowest donor.
    """
    votes: dict[int, dict[str, Any]] = {}
    for aspect_key, neighbors_per_query in neighbors_by_aspect.items():
        if q_idx >= len(neighbors_per_query):
            continue
        top = neighbors_per_query[q_idx]
        seq_ranks = (
            dense_sequence_ranks(top, sequence_keys)
            if sequence_keys is not None
            else None
        )
        for k_pos, (ref_acc, distance) in enumerate(top, start=1):
            d = float(distance)
            for ann in annotations.get(ref_acc, []):
                gtid = int(ann["go_term_id"])
                if aspect_separated:
                    if go_aspect_map.get(gtid, "") != aspect_key:
                        continue
                seq_rank = seq_ranks[k_pos - 1] if seq_ranks is not None else None
                stat = votes.get(gtid)
                if stat is None:
                    stat = _fresh_stat(ref_acc, ann, k_pos, seq_rank, d)
                    votes[gtid] = stat
                stat["ledger"].record(ref_acc, k_pos, seq_rank, d)
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
    sequence_keys: Mapping[str, str] | None = None
    #: Every ``(query, donor)`` a row asked for and did not find. A missing key
    #: and a computed-empty result used to be the same thing here, so a row
    #: emitted with fifteen NULL pair-feature columns was indistinguishable
    #: from a row whose features were genuinely uncomputable, and the caller
    #: was told nothing either way. Recording them costs one set.
    pair_feature_misses: set[tuple[str, str]] = field(default_factory=set)


def _make_row(
    q_acc: str,
    gtid: int,
    stat: dict[str, Any],
    distance_std: float,
    ctx: _RowContext,
) -> dict[str, Any]:
    """Build one PROTEA-shaped prediction row from a tally stat dict."""
    vote_count = int(stat["vote_count"])
    ledger: DonorLedger = stat["ledger"]
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
        "donor_accessions": list(ledger.accessions),
        "donor_k_positions": list(ledger.k_positions),
        "donor_sequence_ranks": ledger.sequence_ranks_or_none(),
        "donor_distances": list(ledger.distances),
        "donor_count": len(ledger),
        "sequence_rank": (
            None if stat.get("sequence_rank") is None else int(stat["sequence_rank"])
        ),
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
    key = (q_acc, donor_ref)
    pf = ctx.pair_features.get(key)
    if pf:
        propagate_pair_features(row, pf)
    elif pf is None and ctx.pair_features:
        # Absent, not empty. An empty dict is a computed answer ("this pair has
        # no features"), a missing key is a question nobody asked, and the two
        # produce the same row. Only the second is recorded.
        ctx.pair_feature_misses.add(key)
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
            sequence_keys=ctx.sequence_keys,
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
        sequence_keys=cfg.sequence_keys,
    )


PAIR_FEATURE_KEYS: tuple[str, ...] = (
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


def propagate_pair_features(
    row: dict[str, Any], pf: dict[str, Any],
) -> None:
    """Copy the alignment / taxonomy fields the lab schema expects.

    Public because a caller whose own neighbour search disagreed with this
    one has to be able to repair those rows afterwards, and a caller that
    reimplemented this would carry a second copy of PAIR_FEATURE_KEYS. Two
    copies of a field list is how a field comes to be added to one of them.
    """
    for key in PAIR_FEATURE_KEYS:
        if key in pf:
            row[key] = pf[key]


#: Kept so nothing importing the private spelling breaks on this release.
_propagate_pair_features = propagate_pair_features


def load_boosters_by_aspect(directory: str | Path) -> dict[str, lgb.Booster]:
    """Load three per-aspect LightGBM artefacts from a directory.

    Expects one artefact per GO aspect, named either ``F.txt`` /
    ``P.txt`` / ``C.txt`` (short codes) or ``mfo.txt`` / ``bpo.txt`` /
    ``cco.txt`` (LAFA-style long codes). Other extensions (``.lgb``,
    ``.bin``, ``.model``) are also accepted; LightGBM's text format is
    extension-agnostic.

    Returns a mapping ``{short_code: Booster}`` keyed on ``F``, ``P``,
    and ``C``. Raises ``FileNotFoundError`` if the directory is missing
    and ``ValueError`` if no aspect-named artefacts are found or two
    files target the same aspect (e.g. both ``F.txt`` and ``mfo.txt``
    present). Partial coverage (one or two aspects) is allowed; the
    selective-rerank path handles missing aspects by leaving those
    predictions unscored.
    """
    root = Path(directory)
    if not root.is_dir():
        raise FileNotFoundError(f"booster directory not found: {root}")
    out: dict[str, lgb.Booster] = {}
    seen_sources: dict[str, str] = {}
    for path in sorted(root.iterdir()):
        if not path.is_file():
            continue
        alias_key = path.stem
        if alias_key not in _ASPECT_ALIASES:
            continue
        aspect = _ASPECT_ALIASES[alias_key]
        if aspect in out:
            raise ValueError(
                f"multiple artefacts target aspect {aspect!r}: "
                f"{seen_sources[aspect]} and {path.name}",
            )
        out[aspect] = lgb.Booster(model_file=str(path))
        seen_sources[aspect] = path.name
    if not out:
        raise ValueError(
            f"no per-aspect booster artefacts found in {root}; "
            "expected files named F/P/C or mfo/bpo/cco",
        )
    return out


def _validate_aspect_boosters(boosters: dict[str, lgb.Booster]) -> None:
    """Refuse boosters keyed by anything other than ``F`` / ``P`` / ``C``.

    The selective-rerank path keys on the row ``aspect`` field which
    the rest of the pipeline writes as a single-letter short code. A
    long-code key (``mfo`` / ``bpo`` / ``cco``) silently routes zero
    rows; surface that misconfiguration as a ``ValueError`` instead.
    """
    if not boosters:
        return
    allowed = set(ASPECT_CODES)
    bad = sorted(k for k in boosters if k not in allowed)
    if bad:
        raise ValueError(
            f"boosters_by_aspect keys must be a subset of {sorted(allowed)}; "
            f"got disallowed keys {bad}. Use ``load_boosters_by_aspect`` to "
            "normalise long-form names (mfo/bpo/cco) into F/P/C.",
        )


@overload
def predict(
    *,
    query_accessions: list[str],
    query_embeddings: np.ndarray,
    reference_accessions: list[str],
    reference_embeddings: np.ndarray,
    annotations: dict[str, list[dict[str, Any]]],
    go_id_map: dict[int, str],
    go_aspect_map: dict[int, str],
    config: PredictConfig | None = ...,
    pca_state: tuple[np.ndarray, np.ndarray] | None = ...,
    pair_features: dict[tuple[str, str], dict[str, Any]] | None = ...,
    booster: lgb.Booster | None = ...,
    boosters_by_aspect: dict[str, lgb.Booster] | None = ...,
    reranker_feature_cols: list[str] | None = ...,
    anc_idx: Anc2VecIndex | None = ...,
    return_diagnostics: Literal[False] = False,
) -> list[dict[str, Any]]: ...


@overload
def predict(
    *,
    query_accessions: list[str],
    query_embeddings: np.ndarray,
    reference_accessions: list[str],
    reference_embeddings: np.ndarray,
    annotations: dict[str, list[dict[str, Any]]],
    go_id_map: dict[int, str],
    go_aspect_map: dict[int, str],
    config: PredictConfig | None = ...,
    pca_state: tuple[np.ndarray, np.ndarray] | None = ...,
    pair_features: dict[tuple[str, str], dict[str, Any]] | None = ...,
    booster: lgb.Booster | None = ...,
    boosters_by_aspect: dict[str, lgb.Booster] | None = ...,
    reranker_feature_cols: list[str] | None = ...,
    anc_idx: Anc2VecIndex | None = ...,
    return_diagnostics: Literal[True],
) -> tuple[list[dict[str, Any]], PredictDiagnostics]: ...


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
    return_diagnostics: bool = False,
) -> list[dict[str, Any]] | tuple[list[dict[str, Any]], PredictDiagnostics]:
    """End-to-end inference. See module docstring for the row shape.

    Pass ``return_diagnostics=True`` to receive a
    :class:`PredictDiagnostics` alongside the rows, carrying the KNN
    state the rows were derived from so a caller can layer its own
    features on top without re-running the search.

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

    if boosters_by_aspect:
        _validate_aspect_boosters(boosters_by_aspect)

    empty_diag = PredictDiagnostics(neighbors_by_aspect={}, go_map_by_aspect={})
    if not query_accessions or query_embeddings.size == 0:
        return ([], empty_diag) if return_diagnostics else []
    if reference_embeddings.size == 0:
        return ([], empty_diag) if return_diagnostics else []

    if cfg.aspect_separated:
        neighbors_by_aspect, go_map_by_aspect = aspect_separated_knn(
            queries=Queries(query_accessions, query_embeddings),
            reference_accessions=reference_accessions,
            reference_embeddings=reference_embeddings,
            annotations=annotations,
            go_aspect_map=go_aspect_map,
            cfg=cfg,
        )
    else:
        neighbors_by_aspect, go_map_by_aspect = unified_knn(
            queries=Queries(query_accessions, query_embeddings),
            reference_accessions=reference_accessions,
            reference_embeddings=reference_embeddings,
            annotations=annotations,
            cfg=cfg,
        )

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

    if return_diagnostics:
        diagnostics = PredictDiagnostics(
            neighbors_by_aspect=neighbors_by_aspect,
            go_map_by_aspect=go_map_by_aspect,
            pair_feature_misses=frozenset(row_ctx.pair_feature_misses),
        )
        return predictions, diagnostics
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


__all__ = ["PredictConfig", "load_boosters_by_aspect", "predict"]
