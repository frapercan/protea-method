"""Unit tests for ``protea_method.lineage``.

The DAG used in most tests is::

    GO:1 (root)
    ├── GO:2
    │   └── GO:4 (leaf)
    └── GO:3
        └── GO:5 (leaf, also part_of GO:2)

Encoded via the ``parents`` dict where each entry lists the
immediate parents (mix of ``is_a`` and ``relationship: part_of``).
"""

from __future__ import annotations

from typing import Any

import pytest

from protea_method.lineage import LINEAGE_FEATURE_KEYS, compute_lineage_features


def _parents() -> dict[str, list[str]]:
    return {
        "GO:1": [],
        "GO:2": ["GO:1"],
        "GO:3": ["GO:1"],
        "GO:4": ["GO:2"],
        "GO:5": ["GO:3", "GO:2"],
    }


def _row(prot: str, go: str) -> dict[str, Any]:
    """Build a prediction row typed as ``dict[str, Any]`` so mypy
    stops narrowing the value type to ``str`` after the helper adds
    float fields."""
    return {"protein_accession": prot, "go_id": go}


def test_feature_keys_are_exported() -> None:
    assert set(LINEAGE_FEATURE_KEYS) == {
        "lineage_is_ancestor_of_known",
        "lineage_is_descendant_of_known",
        "lineage_ancestor_of_count",
        "lineage_descendant_of_count",
    }


def test_empty_predictions_is_noop() -> None:
    preds: list[dict[str, Any]] = []
    compute_lineage_features(preds, parents=_parents(), known_by_protein={"P1": {"GO:1"}})
    assert preds == []


def test_protein_with_no_known_gets_zero_features() -> None:
    preds = [_row("P1", "GO:4")]
    compute_lineage_features(preds, parents=_parents(), known_by_protein={})
    assert preds[0]["lineage_is_ancestor_of_known"] == 0.0
    assert preds[0]["lineage_is_descendant_of_known"] == 0.0
    assert preds[0]["lineage_ancestor_of_count"] == 0.0
    assert preds[0]["lineage_descendant_of_count"] == 0.0


def test_candidate_is_ancestor_of_one_known() -> None:
    # known = {GO:4}, candidate = GO:2 → GO:2 is an ancestor of GO:4.
    preds = [_row("P1", "GO:2")]
    compute_lineage_features(
        preds,
        parents=_parents(),
        known_by_protein={"P1": {"GO:4"}},
    )
    assert preds[0]["lineage_is_ancestor_of_known"] == 1.0
    assert preds[0]["lineage_is_descendant_of_known"] == 0.0
    assert preds[0]["lineage_ancestor_of_count"] == 1.0


def test_candidate_is_descendant_of_one_known() -> None:
    preds = [_row("P1", "GO:4")]
    compute_lineage_features(
        preds,
        parents=_parents(),
        known_by_protein={"P1": {"GO:1"}},
    )
    assert preds[0]["lineage_is_ancestor_of_known"] == 0.0
    assert preds[0]["lineage_is_descendant_of_known"] == 1.0
    assert preds[0]["lineage_descendant_of_count"] == 1.0


def test_candidate_unrelated_to_known() -> None:
    preds = [_row("P1", "GO:5")]
    compute_lineage_features(
        preds,
        parents=_parents(),
        known_by_protein={"P1": {"GO:4"}},
    )
    assert preds[0]["lineage_is_ancestor_of_known"] == 0.0
    assert preds[0]["lineage_is_descendant_of_known"] == 0.0


def test_candidate_equals_known_does_not_self_count() -> None:
    preds = [_row("P1", "GO:4")]
    compute_lineage_features(
        preds,
        parents=_parents(),
        known_by_protein={"P1": {"GO:4"}},
    )
    assert preds[0]["lineage_ancestor_of_count"] == 0.0
    assert preds[0]["lineage_descendant_of_count"] == 0.0
    assert preds[0]["lineage_is_ancestor_of_known"] == 0.0
    assert preds[0]["lineage_is_descendant_of_known"] == 0.0


def test_count_aggregates_over_multiple_known_terms() -> None:
    preds = [_row("P1", "GO:1")]
    compute_lineage_features(
        preds,
        parents=_parents(),
        known_by_protein={"P1": {"GO:4", "GO:5"}},
    )
    assert preds[0]["lineage_ancestor_of_count"] == 2.0
    assert preds[0]["lineage_is_ancestor_of_known"] == 1.0


def test_proteins_isolated_from_each_other() -> None:
    preds = [_row("P1", "GO:2"), _row("P2", "GO:2")]
    compute_lineage_features(
        preds,
        parents=_parents(),
        known_by_protein={"P1": {"GO:4"}},
    )
    assert preds[0]["lineage_ancestor_of_count"] == 1.0
    assert preds[1]["lineage_ancestor_of_count"] == 0.0


def test_part_of_edge_traversed_for_descendant() -> None:
    preds = [_row("P1", "GO:5")]
    compute_lineage_features(
        preds,
        parents=_parents(),
        known_by_protein={"P1": {"GO:2"}},
    )
    assert preds[0]["lineage_is_descendant_of_known"] == 1.0


def test_unknown_go_id_in_predictions_handled() -> None:
    preds = [_row("P1", "GO:99999")]
    compute_lineage_features(
        preds,
        parents=_parents(),
        known_by_protein={"P1": {"GO:4"}},
    )
    assert preds[0]["lineage_is_ancestor_of_known"] == 0.0
    assert preds[0]["lineage_is_descendant_of_known"] == 0.0


def test_blank_go_id_or_protein_accession_zero_features() -> None:
    preds = [_row("", "GO:2"), _row("P1", "")]
    compute_lineage_features(
        preds,
        parents=_parents(),
        known_by_protein={"P1": {"GO:4"}},
    )
    for p in preds:
        assert p["lineage_is_ancestor_of_known"] == 0.0
        assert p["lineage_is_descendant_of_known"] == 0.0


@pytest.mark.parametrize(
    ("cand", "known", "expect_anc", "expect_desc"),
    [
        ("GO:1", {"GO:4"}, 1, 0),
        ("GO:4", {"GO:1"}, 0, 1),
        ("GO:2", {"GO:1"}, 0, 1),
        ("GO:1", {"GO:1"}, 0, 0),
    ],
)
def test_param_sanity(
    cand: str, known: set[str], expect_anc: int, expect_desc: int
) -> None:
    preds = [_row("P1", cand)]
    compute_lineage_features(
        preds, parents=_parents(), known_by_protein={"P1": known}
    )
    assert preds[0]["lineage_ancestor_of_count"] == float(expect_anc)
    assert preds[0]["lineage_descendant_of_count"] == float(expect_desc)
