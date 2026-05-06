"""Coverage tests for ``protea_method.feature_enricher``."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pytest

from protea_method.anc2vec import Anc2VecIndex
from protea_method.feature_enricher import (
    ASPECT_CODES,
    NEW_V6_FEATURE_KEYS,
    _build_anc2vec_pool,
    _collect_gtids_in_play,
    _compute_neighbor_centroids,
    _compute_pca_projection,
    _compute_tax_voter_counters,
    enrich_v6_features,
)


@pytest.fixture
def anc_idx(tmp_path: Path) -> Anc2VecIndex:
    artifact = tmp_path / "anc.npz"
    artifact.parent.mkdir(parents=True, exist_ok=True)
    go_ids = ["GO:0000001", "GO:0000002", "GO:0000003"]
    embeddings = np.eye(3, 4, dtype=np.float32)
    np.savez(
        artifact,
        embeddings=embeddings,
        go_ids=np.array(go_ids, dtype=object),
        ontology_release="test",
    )
    return Anc2VecIndex(artifact)


def test_collect_gtids_includes_predictions_and_neighbor_annotations() -> None:
    predictions = [{"go_term_id": 10}, {"go_term_id": 20}]
    go_map_by_aspect = {
        "F": {"P00001": [{"go_term_id": 30}, {"go_term_id": 40}]},
        "P": {"P00002": [{"go_term_id": 50}]},
    }
    gtids = _collect_gtids_in_play(predictions, go_map_by_aspect)
    assert gtids == {10, 20, 30, 40, 50}


def test_build_anc2vec_pool_shapes(anc_idx: Anc2VecIndex) -> None:
    gtids = {1, 2, 99}
    go_id_map = {1: "GO:0000001", 2: "GO:0000002", 99: "GO:9999999"}
    idx_of_go, all_norm, mask = _build_anc2vec_pool(gtids, go_id_map, anc_idx=anc_idx)
    assert sorted(idx_of_go.keys()) == ["GO:0000001", "GO:0000002", "GO:9999999"]
    assert all_norm.shape == (3, 4)
    assert mask.tolist() == [True, True, False]
    assert all_norm.dtype == np.float32


def test_compute_neighbor_centroids(anc_idx: Anc2VecIndex) -> None:
    go_id_map = {1: "GO:0000001", 2: "GO:0000002"}
    go_aspect_map = {1: "F", 2: "F"}
    idx_of_go, all_norm, mask = _build_anc2vec_pool(
        {1, 2}, go_id_map, anc_idx=anc_idx,
    )
    info = _compute_neighbor_centroids(
        valid_accessions=["Q1"],
        neighbors_by_aspect={"F": [[("R1", 0.1)]]},
        go_map_by_aspect={
            "F": {"R1": [{"go_term_id": 1}, {"go_term_id": 2}]},
        },
        go_id_map=go_id_map,
        go_aspect_map=go_aspect_map,
        idx_of_go=idx_of_go,
        all_norm=all_norm,
        has_emb_mask=mask,
    )
    centroid, nmat = info[("Q1", "F")]
    assert centroid is not None
    assert nmat is not None
    assert nmat.shape == (2, 4)


def test_compute_neighbor_centroids_empty_yields_none(anc_idx: Anc2VecIndex) -> None:
    info = _compute_neighbor_centroids(
        valid_accessions=["Q1"],
        neighbors_by_aspect={"F": [[]]},
        go_map_by_aspect={"F": {}},
        go_id_map={},
        go_aspect_map={},
        idx_of_go={},
        all_norm=np.zeros((0, 4), dtype=np.float32),
        has_emb_mask=np.zeros((0,), dtype=bool),
    )
    assert info[("Q1", "F")] == (None, None)


def test_tax_voter_counters_off() -> None:
    same, close, ca_sum, ca_n, vc = _compute_tax_voter_counters(
        valid_accessions=["Q1"],
        neighbors_by_aspect={"F": [[("R1", 0.1)]]},
        go_map_by_aspect={"F": {"R1": [{"go_term_id": 1}]}},
        pair_features={("Q1", "R1"): {"taxonomic_relation": "same"}},
        compute_taxonomy=False,
    )
    assert same == close == ca_sum == ca_n == vc == {}


def test_tax_voter_counters_on() -> None:
    same, close, ca_sum, ca_n, vc = _compute_tax_voter_counters(
        valid_accessions=["Q1"],
        neighbors_by_aspect={"F": [[("R1", 0.1)]]},
        go_map_by_aspect={"F": {"R1": [{"go_term_id": 1}, {"go_term_id": 2}]}},
        pair_features={
            ("Q1", "R1"): {
                "taxonomic_relation": "same",
                "taxonomic_common_ancestors": 5,
            },
        },
        compute_taxonomy=True,
    )
    assert same["Q1"][1] == 1
    assert close["Q1"][1] == 1
    assert ca_sum["Q1"][1] == 5.0
    assert ca_n["Q1"][1] == 1
    assert vc["Q1"][1] == 1


def test_compute_pca_projection_returns_none_for_empty() -> None:
    assert _compute_pca_projection(np.zeros((0, 4), dtype=np.float32), None) is None
    pca_state = (np.zeros(4, dtype=np.float32), np.eye(2, 4, dtype=np.float32))
    assert _compute_pca_projection(np.zeros((0, 4), dtype=np.float32), pca_state) is None


def test_compute_pca_projection_shapes() -> None:
    rng = np.random.default_rng(0)
    queries = rng.standard_normal(size=(5, 4)).astype(np.float32)
    pca_mean = queries.mean(axis=0)
    pca_components = np.eye(2, 4, dtype=np.float32)
    proj = _compute_pca_projection(queries, (pca_mean, pca_components))
    assert proj is not None
    assert proj.shape == (5, 2)
    assert proj.dtype == np.float32


def test_enrich_v6_features_writes_all_keys(anc_idx: Anc2VecIndex) -> None:
    predictions: list[dict[str, Any]] = [{"protein_accession": "Q1", "go_term_id": 1}]
    go_id_map = {1: "GO:0000001"}
    go_aspect_map = {1: "F"}
    queries = np.array([[1.0, 0.0, 0.0, 0.0]], dtype=np.float32)
    enrich_v6_features(
        predictions,
        go_id_map=go_id_map,
        go_aspect_map=go_aspect_map,
        valid_accessions=["Q1"],
        query_embeddings=queries,
        neighbors_by_aspect={"F": [[("R1", 0.1)]]},
        go_map_by_aspect={"F": {"R1": [{"go_term_id": 1}]}},
        pair_features={("Q1", "R1"): {"taxonomic_relation": "same"}},
        pca_state=None,
        compute_taxonomy=True,
        anc_idx=anc_idx,
    )
    pred = predictions[0]
    for key in NEW_V6_FEATURE_KEYS:
        assert key in pred
    assert pred["anc2vec_has_emb"] == 1.0
    assert pred["tax_voters_same_frac"] == 1.0


def test_enrich_v6_features_short_circuits_on_empty(anc_idx: Anc2VecIndex) -> None:
    predictions: list[dict[str, Any]] = []
    enrich_v6_features(
        predictions,
        go_id_map={},
        go_aspect_map={},
        valid_accessions=[],
        query_embeddings=np.zeros((0, 4), dtype=np.float32),
        neighbors_by_aspect={},
        go_map_by_aspect={},
        pair_features={},
        pca_state=None,
        compute_taxonomy=False,
        anc_idx=anc_idx,
    )
    assert predictions == []


def test_enrich_v6_features_pca_path(anc_idx: Anc2VecIndex) -> None:
    """When pca_state is provided, the merge loop fills emb_pca_query_*
    columns from the projection rather than NaN.
    """
    predictions: list[dict[str, Any]] = [{"protein_accession": "Q1", "go_term_id": 1}]
    queries = np.eye(1, 4, dtype=np.float32)
    pca_mean = np.zeros(4, dtype=np.float32)
    # PCA components must have exactly EMBEDDING_PCA_DIM rows so the
    # merge loop's f"emb_pca_query_{i}" indices stay in bounds.
    from protea_method.reranker import EMBEDDING_PCA_DIM as _DIM
    pca_components = np.zeros((_DIM, 4), dtype=np.float32)
    pca_components[0, 0] = 1.0  # project axis 0 onto component 0
    enrich_v6_features(
        predictions,
        go_id_map={1: "GO:0000001"},
        go_aspect_map={1: "F"},
        valid_accessions=["Q1"],
        query_embeddings=queries,
        neighbors_by_aspect={},
        go_map_by_aspect={},
        pair_features={},
        pca_state=(pca_mean, pca_components),
        compute_taxonomy=False,
        anc_idx=anc_idx,
    )
    pred = predictions[0]
    assert pred["emb_pca_query_0"] == pytest.approx(1.0)
    assert pred["emb_pca_query_1"] == pytest.approx(0.0)


def test_aspect_codes_constant() -> None:
    assert ASPECT_CODES == ("F", "P", "C")
