"""``return_diagnostics`` is part of the published surface, on both branches.

This capability shipped on ``main`` in 0.3.0 (#8, #10, May 2026) and was never
carried onto ``develop``. PROTEA imports ``PredictDiagnostics`` at module level
and passes ``return_diagnostics=True``, so for three months the trunk could not
have been promoted to ``main`` without breaking that consumer at import time.

The tests exist so the gap cannot reopen quietly. The failure they guard is not
a wrong answer, it is a release that looks fine here and fails in the consumer.
"""

from __future__ import annotations

from typing import Any

import numpy as np

import protea_method
from protea_method.pipeline import PredictConfig, PredictDiagnostics, predict


def _toy_corpus() -> tuple[
    list[str], np.ndarray, list[str], np.ndarray, dict[str, list[dict[str, object]]]
]:
    rng = np.random.default_rng(0)
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
        ["Q1", "Q2"],
        rng.standard_normal(size=(2, 4)).astype(np.float32),
        [f"R{i:02d}" for i in range(8)],
        rng.standard_normal(size=(8, 4)).astype(np.float32),
        annotations,
    )


GO_ID_MAP = {1: "GO:0000001", 2: "GO:0000002", 3: "GO:0000003"}
GO_ASPECT_MAP = {1: "F", 2: "P", 3: "C"}


def _config(aspect_separated: bool = False) -> PredictConfig:
    """v6 features off: they load an anc2vec artifact that is not a test
    fixture, and PROTEA's adapter builds the config the same way."""
    return PredictConfig(
        k=3, aspect_separated=aspect_separated, compute_v6_features=False
    )


def _rows(aspect_separated: bool = False) -> list[dict[str, Any]]:
    """The plain call, spelled the way an existing caller spells it."""
    qa, qe, ra, re_, anns = _toy_corpus()
    return predict(
        query_accessions=qa,
        query_embeddings=qe,
        reference_accessions=ra,
        reference_embeddings=re_,
        annotations=anns,
        go_id_map=GO_ID_MAP,
        go_aspect_map=GO_ASPECT_MAP,
        config=_config(aspect_separated),
    )


def _rows_and_diag(
    aspect_separated: bool = False,
) -> tuple[list[dict[str, Any]], PredictDiagnostics]:
    """The diagnostics call. Written out rather than routed through kwargs so
    the overload actually has a literal to select on, which is the half of
    this feature a type checker sees."""
    qa, qe, ra, re_, anns = _toy_corpus()
    return predict(
        query_accessions=qa,
        query_embeddings=qe,
        reference_accessions=ra,
        reference_embeddings=re_,
        annotations=anns,
        go_id_map=GO_ID_MAP,
        go_aspect_map=GO_ASPECT_MAP,
        config=_config(aspect_separated),
        return_diagnostics=True,
    )


class TestTheDefaultShapeIsUnchanged:
    def test_without_the_flag_a_bare_list_comes_back(self) -> None:
        """Existing callers must not have to change anything."""
        assert isinstance(_rows(), list)

    def test_explicitly_false_is_also_a_bare_list(self) -> None:
        qa, qe, ra, re_, anns = _toy_corpus()
        out = predict(
            query_accessions=qa, query_embeddings=qe,
            reference_accessions=ra, reference_embeddings=re_,
            annotations=anns, go_id_map=GO_ID_MAP, go_aspect_map=GO_ASPECT_MAP,
            config=_config(), return_diagnostics=False,
        )
        assert isinstance(out, list)


class TestTheDiagnosticsCarryTheKnnStateTheRowsCameFrom:
    def test_the_flag_returns_a_pair(self) -> None:
        rows, diag = _rows_and_diag()
        assert isinstance(rows, list)
        assert isinstance(diag, PredictDiagnostics)

    def test_the_rows_are_the_same_rows(self) -> None:
        """The flag exposes state; it must not change what is predicted."""
        plain = _rows()
        rows, _ = _rows_and_diag()
        assert rows == plain

    def test_the_unified_path_uses_the_single_empty_key(self) -> None:
        _, diag = _rows_and_diag()
        assert set(diag.neighbors_by_aspect) == {""}
        assert set(diag.go_map_by_aspect) == {""}

    def test_there_is_one_neighbour_list_per_query(self) -> None:
        _, diag = _rows_and_diag()
        assert len(diag.neighbors_by_aspect[""]) == 2

    def test_the_neighbours_are_accession_distance_pairs_in_rank_order(self) -> None:
        _, diag = _rows_and_diag()
        for hits in diag.neighbors_by_aspect[""]:
            assert all(isinstance(a, str) and isinstance(d, float) for a, d in hits)
            assert [d for _, d in hits] == sorted(d for _, d in hits)

    def test_every_neighbour_is_in_the_go_map(self) -> None:
        """Otherwise a caller deriving features would hit a KeyError."""
        _, diag = _rows_and_diag()
        mapped = diag.go_map_by_aspect[""]
        for hits in diag.neighbors_by_aspect[""]:
            for acc, _ in hits:
                assert acc in mapped

    def test_the_aspect_separated_path_keys_by_aspect(self) -> None:
        _, diag = _rows_and_diag(aspect_separated=True)
        assert set(diag.neighbors_by_aspect) <= {"P", "F", "C"}
        assert set(diag.neighbors_by_aspect) == set(diag.go_map_by_aspect)


class TestTheEmptyCasesStillHonourTheContract:
    def test_no_queries_still_returns_a_pair(self) -> None:
        """A caller that unpacks two values must not crash on an empty batch."""
        _, _, ra, re_, anns = _toy_corpus()
        rows, diag = predict(
            query_accessions=[],
            query_embeddings=np.zeros((0, 4), dtype=np.float32),
            reference_accessions=ra,
            reference_embeddings=re_,
            annotations=anns,
            go_id_map=GO_ID_MAP,
            go_aspect_map=GO_ASPECT_MAP,
            config=PredictConfig(k=3, compute_v6_features=False),
            return_diagnostics=True,
        )
        assert rows == []
        assert diag.neighbors_by_aspect == {}

    def test_no_references_still_returns_a_pair(self) -> None:
        qa, qe, _, _, anns = _toy_corpus()
        rows, diag = predict(
            query_accessions=qa,
            query_embeddings=qe,
            reference_accessions=[],
            reference_embeddings=np.zeros((0, 4), dtype=np.float32),
            annotations=anns,
            go_id_map=GO_ID_MAP,
            go_aspect_map=GO_ASPECT_MAP,
            config=PredictConfig(k=3, compute_v6_features=False),
            return_diagnostics=True,
        )
        assert rows == []
        assert diag.go_map_by_aspect == {}


class TestThePackageRootExportsWhatConsumersImport:
    def test_predict_diagnostics_is_importable_from_the_root(self) -> None:
        """PROTEA does ``from protea_method.pipeline import PredictDiagnostics``.

        The root export is the weaker of the two paths, so it is the one worth
        pinning: a refactor that moves the class would break it first.
        """
        assert protea_method.PredictDiagnostics is PredictDiagnostics
        assert "PredictDiagnostics" in protea_method.__all__

    def test_the_aspect_booster_loader_survived_the_merge(self) -> None:
        """develop added this; restoring the diagnostics must not drop it."""
        assert "load_boosters_by_aspect" in protea_method.__all__
