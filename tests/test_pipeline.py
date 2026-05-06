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
