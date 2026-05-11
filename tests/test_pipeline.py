"""End-to-end tests for ``protea_method.pipeline``."""

from __future__ import annotations

from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd

from protea_method.anc2vec import Anc2VecIndex
from protea_method.pipeline import PredictConfig, predict
from protea_method.reranker import ALL_FEATURES, LABEL_COLUMN, prepare_dataset


def _make_anc_idx(tmp_path: Path) -> Anc2VecIndex:
    artifact = tmp_path / "anc.npz"
    np.savez(
        artifact,
        embeddings=np.eye(3, 4, dtype=np.float32),
        go_ids=np.array(["GO:0000001", "GO:0000002", "GO:0000003"], dtype=object),
        ontology_release="test",
    )
    return Anc2VecIndex(artifact)


def _toy_corpus() -> tuple[
    list[str], np.ndarray, list[str], np.ndarray, dict[str, list[dict[str, object]]]
]:
    rng = np.random.default_rng(0)
    query_accessions = ["Q1", "Q2"]
    reference_accessions = [f"R{i:02d}" for i in range(8)]
    query_embeddings = rng.standard_normal(size=(2, 4)).astype(np.float32)
    reference_embeddings = rng.standard_normal(size=(8, 4)).astype(np.float32)
    annotations: dict[str, list[dict[str, object]]] = {
        "R00": [{"go_term_id": 1}, {"go_term_id": 2}],
        "R01": [{"go_term_id": 1}],
        "R02": [{"go_term_id": 3}],
        "R03": [],
        "R04": [{"go_term_id": 2}],
        "R05": [{"go_term_id": 1}, {"go_term_id": 3}],
        "R06": [{"go_term_id": 2}],
        "R07": [{"go_term_id": 3}],
    }
    return (
        query_accessions,
        query_embeddings,
        reference_accessions,
        reference_embeddings,
        annotations,
    )


def test_predict_returns_base_features(tmp_path: Path) -> None:
    qa, qe, ra, re_, anns = _toy_corpus()
    go_id_map = {1: "GO:0000001", 2: "GO:0000002", 3: "GO:0000003"}
    go_aspect_map = {1: "F", 2: "P", 3: "C"}
    anc_idx = _make_anc_idx(tmp_path)
    preds = predict(
        query_accessions=qa,
        query_embeddings=qe,
        reference_accessions=ra,
        reference_embeddings=re_,
        annotations=anns,
        go_id_map=go_id_map,
        go_aspect_map=go_aspect_map,
        config=PredictConfig(k=3, compute_v6_features=True),
        anc_idx=anc_idx,
    )
    assert preds, "expected at least one prediction"
    for p in preds:
        assert p["protein_accession"] in {"Q1", "Q2"}
        assert isinstance(p["go_term_id"], int)
        assert p["vote_count"] >= 1
        assert p["min_distance"] >= 0.0
        assert p["min_distance"] <= p["mean_distance"]
        assert "anc2vec_neighbor_cos" in p


def test_predict_emits_protea_row_shape(tmp_path: Path) -> None:
    """Every row carries the PROTEA-compatible identity + reranker
    aggregate fields the LightGBM lab booster trained on.
    """
    qa, qe, ra, re_, anns = _toy_corpus()
    anns_with_meta: dict[str, list[dict[str, object]]] = {
        ref: [
            {**ann, "qualifier": "enables", "evidence_code": "IEA"}
            for ann in ann_list
        ]
        for ref, ann_list in anns.items()
    }
    go_id_map = {1: "GO:0000001", 2: "GO:0000002", 3: "GO:0000003"}
    go_aspect_map = {1: "F", 2: "P", 3: "C"}
    preds = predict(
        query_accessions=qa,
        query_embeddings=qe,
        reference_accessions=ra,
        reference_embeddings=re_,
        annotations=anns_with_meta,
        go_id_map=go_id_map,
        go_aspect_map=go_aspect_map,
        config=PredictConfig(
            k=3, compute_v6_features=False, prediction_set_id="pset-42",
        ),
    )
    assert preds
    required = {
        "prediction_set_id",
        "ref_protein_accession",
        "qualifier",
        "evidence_code",
        "vote_count",
        "k_position",
        "go_term_frequency",
        "ref_annotation_density",
        "neighbor_distance_std",
        "neighbor_vote_fraction",
        "neighbor_min_distance",
        "neighbor_mean_distance",
        "go_id",
        "aspect",
    }
    for p in preds:
        missing = required - p.keys()
        assert not missing, f"missing fields {missing} on row {p}"
        assert p["prediction_set_id"] == "pset-42"
        assert p["ref_protein_accession"] in ra
        assert p["qualifier"] == "enables"
        assert p["evidence_code"] == "IEA"
        assert p["vote_count"] >= 1
        assert 1 <= p["k_position"] <= 3
        assert p["go_term_frequency"] >= 1
        assert p["ref_annotation_density"] >= 1
        assert p["neighbor_distance_std"] >= 0.0
        assert 0.0 <= p["neighbor_vote_fraction"] <= 1.0
        assert p["neighbor_min_distance"] <= p["neighbor_mean_distance"]
        assert p["go_id"] == go_id_map[p["go_term_id"]]
        assert p["aspect"] == go_aspect_map[p["go_term_id"]]


def test_predict_propagates_pair_features_from_donor(tmp_path: Path) -> None:
    """Alignment / taxonomy fields from the donor neighbour's
    ``pair_features`` entry are merged into the row.
    """
    qa, qe, ra, re_, anns = _toy_corpus()
    pair_features: dict[tuple[str, str], dict[str, object]] = {
        (q, r): {
            "identity_nw": 0.42,
            "alignment_score_nw": 123.0,
            "length_query": 100,
            "length_ref": 110,
            "taxonomic_distance": 7,
            "taxonomic_common_ancestors": 3,
            "taxonomic_relation": "close",
        }
        for q in qa
        for r in ra
    }
    preds = predict(
        query_accessions=qa,
        query_embeddings=qe,
        reference_accessions=ra,
        reference_embeddings=re_,
        annotations=anns,
        go_id_map={1: "GO:0000001", 2: "GO:0000002", 3: "GO:0000003"},
        go_aspect_map={1: "F", 2: "P", 3: "C"},
        config=PredictConfig(k=3, compute_v6_features=False),
        pair_features=pair_features,
    )
    assert preds
    for p in preds:
        assert p["identity_nw"] == 0.42
        assert p["alignment_score_nw"] == 123.0
        assert p["length_query"] == 100
        assert p["length_ref"] == 110
        assert p["taxonomic_distance"] == 7
        assert p["taxonomic_common_ancestors"] == 3
        assert p["taxonomic_relation"] == "close"


def test_predict_neighbor_vote_fraction_matches_k(tmp_path: Path) -> None:
    """vote_count / k_limit equals ``neighbor_vote_fraction``."""
    qa, qe, ra, re_, anns = _toy_corpus()
    preds = predict(
        query_accessions=qa,
        query_embeddings=qe,
        reference_accessions=ra,
        reference_embeddings=re_,
        annotations=anns,
        go_id_map={1: "GO:0000001", 2: "GO:0000002", 3: "GO:0000003"},
        go_aspect_map={1: "F", 2: "P", 3: "C"},
        config=PredictConfig(k=4, compute_v6_features=False),
    )
    for p in preds:
        assert p["neighbor_vote_fraction"] == p["vote_count"] / 4.0


def test_predict_aspect_separated_rows_carry_protea_fields(tmp_path: Path) -> None:
    """Per-aspect KNN still produces the full PROTEA row shape."""
    qa, qe, ra, re_, anns = _toy_corpus()
    preds = predict(
        query_accessions=qa,
        query_embeddings=qe,
        reference_accessions=ra,
        reference_embeddings=re_,
        annotations=anns,
        go_id_map={1: "GO:0000001", 2: "GO:0000002", 3: "GO:0000003"},
        go_aspect_map={1: "F", 2: "P", 3: "C"},
        config=PredictConfig(
            k=2,
            aspect_separated=True,
            compute_v6_features=False,
            prediction_set_id="pset-asp",
        ),
    )
    assert preds
    for p in preds:
        assert p["prediction_set_id"] == "pset-asp"
        assert "ref_protein_accession" in p
        assert "k_position" in p
        assert "go_term_frequency" in p
        assert "neighbor_vote_fraction" in p
        assert 0.0 <= p["neighbor_vote_fraction"] <= 1.0


def test_predict_distance_std_zero_when_single_distance(tmp_path: Path) -> None:
    """One total neighbour across all aspects produces std = 0."""
    qa = ["Q1"]
    qe = np.array([[1.0, 0.0, 0.0, 0.0]], dtype=np.float32)
    ra = ["R0"]
    re_ = np.array([[1.0, 0.0, 0.0, 0.0]], dtype=np.float32)
    anns: dict[str, list[dict[str, object]]] = {"R0": [{"go_term_id": 1}]}
    preds = predict(
        query_accessions=qa,
        query_embeddings=qe,
        reference_accessions=ra,
        reference_embeddings=re_,
        annotations=anns,
        go_id_map={1: "GO:0000001"},
        go_aspect_map={1: "F"},
        config=PredictConfig(k=1, compute_v6_features=False),
    )
    assert preds
    for p in preds:
        assert p["neighbor_distance_std"] == 0.0
        assert p["k_position"] == 1
        assert p["vote_count"] == 1
        assert p["ref_protein_accession"] == "R0"


def test_predict_short_circuits_on_empty_query(tmp_path: Path) -> None:
    _, _, ra, re_, anns = _toy_corpus()
    preds = predict(
        query_accessions=[],
        query_embeddings=np.zeros((0, 4), dtype=np.float32),
        reference_accessions=ra,
        reference_embeddings=re_,
        annotations=anns,
        go_id_map={},
        go_aspect_map={},
        anc_idx=_make_anc_idx(tmp_path),
    )
    assert preds == []


def test_predict_short_circuits_on_empty_reference(tmp_path: Path) -> None:
    qa, qe, _, _, _ = _toy_corpus()
    preds = predict(
        query_accessions=qa,
        query_embeddings=qe,
        reference_accessions=[],
        reference_embeddings=np.zeros((0, 4), dtype=np.float32),
        annotations={},
        go_id_map={},
        go_aspect_map={},
        anc_idx=_make_anc_idx(tmp_path),
    )
    assert preds == []


def test_predict_aspect_separated_runs_three_knns(tmp_path: Path) -> None:
    """Aspect-separated mode produces predictions tagged by aspect, with
    each aspect's votes restricted to that aspect's annotations.
    """
    qa, qe, ra, re_, anns = _toy_corpus()
    go_id_map = {1: "GO:0000001", 2: "GO:0000002", 3: "GO:0000003"}
    go_aspect_map = {1: "F", 2: "P", 3: "C"}
    preds = predict(
        query_accessions=qa,
        query_embeddings=qe,
        reference_accessions=ra,
        reference_embeddings=re_,
        annotations=anns,
        go_id_map=go_id_map,
        go_aspect_map=go_aspect_map,
        config=PredictConfig(k=2, aspect_separated=True, compute_v6_features=False),
        anc_idx=_make_anc_idx(tmp_path),
    )
    aspects = {p["aspect"] for p in preds}
    assert {"F", "P", "C"}.issubset(aspects | {""})
    for p in preds:
        gtid = p["go_term_id"]
        assert p["aspect"] == go_aspect_map[gtid]


def test_partition_refs_by_aspect_filters_correctly(tmp_path: Path) -> None:
    """A reference with only F annotations belongs only to the F bank."""
    from protea_method.pipeline import _partition_refs_by_aspect

    refs = ["A", "B", "C"]
    embeddings = np.eye(3, 4, dtype=np.float32)
    annotations: dict[str, list[dict[str, object]]] = {
        "A": [{"go_term_id": 1}],  # F only
        "B": [{"go_term_id": 2}, {"go_term_id": 3}],  # P + C
        "C": [],
    }
    go_aspect_map = {1: "F", 2: "P", 3: "C"}
    out = _partition_refs_by_aspect(refs, embeddings, annotations, go_aspect_map)
    assert out["F"][0] == ["A"]
    assert out["P"][0] == ["B"]
    assert out["C"][0] == ["B"]


def test_predict_skips_v6_when_disabled(tmp_path: Path) -> None:
    qa, qe, ra, re_, anns = _toy_corpus()
    preds = predict(
        query_accessions=qa,
        query_embeddings=qe,
        reference_accessions=ra,
        reference_embeddings=re_,
        annotations=anns,
        go_id_map={1: "GO:0000001", 2: "GO:0000002", 3: "GO:0000003"},
        go_aspect_map={1: "F", 2: "P", 3: "C"},
        config=PredictConfig(k=3, compute_v6_features=False),
        anc_idx=_make_anc_idx(tmp_path),
    )
    assert preds
    for p in preds:
        assert "anc2vec_neighbor_cos" not in p


def test_predict_with_reranker_scores(tmp_path: Path) -> None:
    """When a booster is provided, every prediction gets a reranker_score in [0, 1]."""
    qa, qe, ra, re_, anns = _toy_corpus()

    rng = np.random.default_rng(1)
    n = 200
    train_rows: dict[str, list] = {
        "protein_accession": [f"P{i}" for i in range(n)],
        "go_id": [f"GO:{i}" for i in range(n)],
        "aspect": rng.choice(["F", "P", "C"], n).tolist(),
        LABEL_COLUMN: rng.integers(0, 2, n).tolist(),
    }
    for col in ALL_FEATURES:
        train_rows.setdefault(col, rng.random(n).tolist())
    train_df = pd.DataFrame(train_rows)
    X, y = prepare_dataset(train_df)
    booster = lgb.train(
        {"objective": "binary", "verbose": -1, "num_leaves": 7, "learning_rate": 0.1},
        lgb.Dataset(X, label=y),
        num_boost_round=5,
    )

    preds = predict(
        query_accessions=qa,
        query_embeddings=qe,
        reference_accessions=ra,
        reference_embeddings=re_,
        annotations=anns,
        go_id_map={1: "GO:0000001", 2: "GO:0000002", 3: "GO:0000003"},
        go_aspect_map={1: "F", 2: "P", 3: "C"},
        config=PredictConfig(k=3, compute_v6_features=True),
        booster=booster,
        anc_idx=_make_anc_idx(tmp_path),
    )
    assert preds
    for p in preds:
        score = p["reranker_score"]
        assert 0.0 <= score <= 1.0


def _train_aspect_booster(rng_seed: int) -> lgb.Booster:
    rng = np.random.default_rng(rng_seed)
    n = 200
    train_rows: dict[str, list] = {
        "protein_accession": [f"P{i}" for i in range(n)],
        "go_id": [f"GO:{i}" for i in range(n)],
        "aspect": rng.choice(["F", "P", "C"], n).tolist(),
        LABEL_COLUMN: rng.integers(0, 2, n).tolist(),
    }
    for col in ALL_FEATURES:
        train_rows.setdefault(col, rng.random(n).tolist())
    train_df = pd.DataFrame(train_rows)
    X, y = prepare_dataset(train_df)
    return lgb.train(
        {"objective": "binary", "verbose": -1, "num_leaves": 7, "learning_rate": 0.1},
        lgb.Dataset(X, label=y),
        num_boost_round=5,
    )


def test_predict_with_boosters_by_aspect_scores_only_covered_aspects(
    tmp_path: Path,
) -> None:
    """Per-aspect boosters score predictions of their aspect only."""
    qa, qe, ra, re_, anns = _toy_corpus()

    boosters_by_aspect = {
        "F": _train_aspect_booster(1),
        "C": _train_aspect_booster(2),
        # No booster for aspect P → those predictions stay unscored.
    }

    preds = predict(
        query_accessions=qa,
        query_embeddings=qe,
        reference_accessions=ra,
        reference_embeddings=re_,
        annotations=anns,
        go_id_map={1: "GO:0000001", 2: "GO:0000002", 3: "GO:0000003"},
        go_aspect_map={1: "F", 2: "P", 3: "C"},
        config=PredictConfig(k=3, compute_v6_features=True),
        boosters_by_aspect=boosters_by_aspect,
        anc_idx=_make_anc_idx(tmp_path),
    )
    assert preds
    f_or_c = [p for p in preds if p["aspect"] in {"F", "C"}]
    p_only = [p for p in preds if p["aspect"] == "P"]
    assert f_or_c, "expected predictions for aspects F/C in toy corpus"
    for p in f_or_c:
        assert "reranker_score" in p
        assert 0.0 <= p["reranker_score"] <= 1.0
    for p in p_only:
        assert "reranker_score" not in p


def test_predict_boosters_by_aspect_takes_precedence_over_single(
    tmp_path: Path,
) -> None:
    """When both ``boosters_by_aspect`` and ``booster`` are given, the
    per-aspect mapping wins; aspects without an entry are left
    unscored even if a single ``booster`` is also provided.

    The toy corpus only produces predictions with aspects ``P`` and
    ``C`` under ``k=3``; we route only ``C`` so non-``C`` rows are
    proof that the single booster does not bleed in.
    """
    qa, qe, ra, re_, anns = _toy_corpus()
    single = _train_aspect_booster(3)
    boosters_by_aspect = {"C": _train_aspect_booster(4)}

    preds = predict(
        query_accessions=qa,
        query_embeddings=qe,
        reference_accessions=ra,
        reference_embeddings=re_,
        annotations=anns,
        go_id_map={1: "GO:0000001", 2: "GO:0000002", 3: "GO:0000003"},
        go_aspect_map={1: "F", 2: "P", 3: "C"},
        config=PredictConfig(k=3, compute_v6_features=True),
        booster=single,
        boosters_by_aspect=boosters_by_aspect,
        anc_idx=_make_anc_idx(tmp_path),
    )
    c_preds = [p for p in preds if p["aspect"] == "C"]
    non_c = [p for p in preds if p["aspect"] != "C"]
    assert c_preds, "expected at least one C prediction in the toy corpus"
    assert non_c, "expected at least one non-C prediction in the toy corpus"
    for p in c_preds:
        assert "reranker_score" in p
    for p in non_c:
        # boosters_by_aspect wins → single booster ignored on non-C aspects.
        assert "reranker_score" not in p


# ---------------------------------------------------------------------------
# return_diagnostics — opt-in intermediate state for downstream callers
# ---------------------------------------------------------------------------


def test_predict_default_returns_predictions_only(tmp_path: Path) -> None:
    """Backwards compatibility: callers that don't pass ``return_diagnostics``
    keep receiving a plain list, not a tuple."""
    qa, qe, ra, re_, anns = _toy_corpus()
    out = predict(
        query_accessions=qa,
        query_embeddings=qe,
        reference_accessions=ra,
        reference_embeddings=re_,
        annotations=anns,
        go_id_map={1: "GO:0000001", 2: "GO:0000002", 3: "GO:0000003"},
        go_aspect_map={1: "F", 2: "P", 3: "C"},
        config=PredictConfig(k=2, compute_v6_features=False),
        anc_idx=_make_anc_idx(tmp_path),
    )
    assert isinstance(out, list)


def test_predict_return_diagnostics_unified(tmp_path: Path) -> None:
    """Diagnostics expose neighbours + go_map for the unified KNN branch."""
    from protea_method.pipeline import PredictDiagnostics

    qa, qe, ra, re_, anns = _toy_corpus()
    out = predict(
        query_accessions=qa,
        query_embeddings=qe,
        reference_accessions=ra,
        reference_embeddings=re_,
        annotations=anns,
        go_id_map={1: "GO:0000001", 2: "GO:0000002", 3: "GO:0000003"},
        go_aspect_map={1: "F", 2: "P", 3: "C"},
        config=PredictConfig(k=2, compute_v6_features=False),
        anc_idx=_make_anc_idx(tmp_path),
        return_diagnostics=True,
    )
    assert isinstance(out, tuple)
    _preds, diag = out
    assert isinstance(diag, PredictDiagnostics)
    # Unified mode collapses to a single ``""`` aspect key.
    assert set(diag.neighbors_by_aspect.keys()) == {""}
    assert set(diag.go_map_by_aspect.keys()) == {""}
    # One neighbour list per query.
    assert len(diag.neighbors_by_aspect[""]) == len(qa)
    # go_map covers every neighbour seen.
    seen_refs: set[str] = set()
    for hits in diag.neighbors_by_aspect[""]:
        for ref_acc, _ in hits:
            seen_refs.add(ref_acc)
    assert seen_refs.issubset(diag.go_map_by_aspect[""].keys())


def test_predict_return_diagnostics_aspect_separated(tmp_path: Path) -> None:
    """Diagnostics expose one neighbours list + go_map per GO aspect."""
    from protea_method.pipeline import PredictDiagnostics

    qa, qe, ra, re_, anns = _toy_corpus()
    out = predict(
        query_accessions=qa,
        query_embeddings=qe,
        reference_accessions=ra,
        reference_embeddings=re_,
        annotations=anns,
        go_id_map={1: "GO:0000001", 2: "GO:0000002", 3: "GO:0000003"},
        go_aspect_map={1: "F", 2: "P", 3: "C"},
        config=PredictConfig(k=2, aspect_separated=True, compute_v6_features=False),
        anc_idx=_make_anc_idx(tmp_path),
        return_diagnostics=True,
    )
    assert isinstance(out, tuple)
    _preds, diag = out
    assert isinstance(diag, PredictDiagnostics)
    # Aspect-separated mode: one bucket per ASPECT_CODES letter.
    assert set(diag.neighbors_by_aspect.keys()) == {"P", "F", "C"}
    assert set(diag.go_map_by_aspect.keys()) == {"P", "F", "C"}


def test_predict_return_diagnostics_empty_query(tmp_path: Path) -> None:
    """Empty-query short-circuit still returns a (preds, diag) tuple with
    empty diagnostics so callers don't have to special-case the path."""
    from protea_method.pipeline import PredictDiagnostics

    _, _, ra, re_, anns = _toy_corpus()
    out = predict(
        query_accessions=[],
        query_embeddings=np.zeros((0, 4), dtype=np.float32),
        reference_accessions=ra,
        reference_embeddings=re_,
        annotations=anns,
        go_id_map={},
        go_aspect_map={},
        anc_idx=_make_anc_idx(tmp_path),
        return_diagnostics=True,
    )
    preds, diag = out
    assert preds == []
    assert isinstance(diag, PredictDiagnostics)
    assert diag.neighbors_by_aspect == {}
    assert diag.go_map_by_aspect == {}


def test_predict_lineage_features_off_by_default(tmp_path: Path) -> None:
    """Bit-exact baseline: lineage columns are absent when the flag is off."""
    qa, qe, ra, re_, anns = _toy_corpus()
    go_id_map = {1: "GO:0000001", 2: "GO:0000002", 3: "GO:0000003"}
    go_aspect_map = {1: "F", 2: "P", 3: "C"}
    preds = predict(
        query_accessions=qa,
        query_embeddings=qe,
        reference_accessions=ra,
        reference_embeddings=re_,
        annotations=anns,
        go_id_map=go_id_map,
        go_aspect_map=go_aspect_map,
        config=PredictConfig(k=3, compute_v6_features=False),
    )
    assert preds
    for p in preds:
        assert "lineage_is_ancestor_of_known" not in p
        assert "lineage_is_descendant_of_known" not in p


def test_predict_lineage_features_on_layers_columns(tmp_path: Path) -> None:
    """When the flag is on, the 4 lineage columns land on every row."""
    qa, qe, ra, re_, anns = _toy_corpus()
    go_id_map = {1: "GO:0000001", 2: "GO:0000002", 3: "GO:0000003"}
    go_aspect_map = {1: "F", 2: "P", 3: "C"}
    # GO:0000002 is_a GO:0000001; GO:0000003 is_a GO:0000002.
    parents = {
        "GO:0000002": ["GO:0000001"],
        "GO:0000003": ["GO:0000002"],
    }
    known_by_protein = {"Q1": {"GO:0000003"}, "Q2": set()}
    preds = predict(
        query_accessions=qa,
        query_embeddings=qe,
        reference_accessions=ra,
        reference_embeddings=re_,
        annotations=anns,
        go_id_map=go_id_map,
        go_aspect_map=go_aspect_map,
        config=PredictConfig(
            k=3,
            compute_v6_features=False,
            compute_lineage_features=True,
        ),
        parents=parents,
        known_by_protein=known_by_protein,
    )
    assert preds
    for p in preds:
        for key in (
            "lineage_is_ancestor_of_known",
            "lineage_is_descendant_of_known",
            "lineage_ancestor_of_count",
            "lineage_descendant_of_count",
        ):
            assert key in p, f"missing {key} on {p}"
    # Q1 knows GO:0000003 (a descendant of GO:0000001 and GO:0000002), so
    # any candidate equal to GO:0000001 or GO:0000002 must flip the
    # ancestor-of-known flag on for Q1; Q2 has no known annotations so
    # every flag stays zero.
    q1_rows = [p for p in preds if p["protein_accession"] == "Q1"]
    q2_rows = [p for p in preds if p["protein_accession"] == "Q2"]
    q1_ancestor_hits = [
        p for p in q1_rows
        if p.get("go_id") in {"GO:0000001", "GO:0000002"}
    ]
    if q1_ancestor_hits:
        assert any(
            p["lineage_is_ancestor_of_known"] == 1.0 for p in q1_ancestor_hits
        )
    for p in q2_rows:
        assert p["lineage_is_ancestor_of_known"] == 0.0
        assert p["lineage_is_descendant_of_known"] == 0.0


def test_predict_lineage_features_requires_parents_and_known(tmp_path: Path) -> None:
    """compute_lineage_features=True without inputs is a hard error."""
    qa, qe, ra, re_, anns = _toy_corpus()
    go_id_map = {1: "GO:0000001", 2: "GO:0000002", 3: "GO:0000003"}
    go_aspect_map = {1: "F", 2: "P", 3: "C"}
    import pytest

    with pytest.raises(ValueError, match="compute_lineage_features"):
        predict(
            query_accessions=qa,
            query_embeddings=qe,
            reference_accessions=ra,
            reference_embeddings=re_,
            annotations=anns,
            go_id_map=go_id_map,
            go_aspect_map=go_aspect_map,
            config=PredictConfig(
                k=3,
                compute_v6_features=False,
                compute_lineage_features=True,
            ),
        )
