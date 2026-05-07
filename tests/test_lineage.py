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


def test_feature_keys_are_exported():
    assert set(LINEAGE_FEATURE_KEYS) == {
        "lineage_is_ancestor_of_known",
        "lineage_is_descendant_of_known",
        "lineage_ancestor_of_count",
        "lineage_descendant_of_count",
    }


def test_empty_predictions_is_noop():
    preds: list[dict] = []
    compute_lineage_features(preds, parents=_parents(), known_by_protein={"P1": {"GO:1"}})
    assert preds == []


def test_protein_with_no_known_gets_zero_features():
    preds = [{"protein_accession": "P1", "go_id": "GO:4"}]
    compute_lineage_features(preds, parents=_parents(), known_by_protein={})
    assert preds[0]["lineage_is_ancestor_of_known"] == 0.0
    assert preds[0]["lineage_is_descendant_of_known"] == 0.0
    assert preds[0]["lineage_ancestor_of_count"] == 0.0
    assert preds[0]["lineage_descendant_of_count"] == 0.0


def test_candidate_is_ancestor_of_one_known():
    # known = {GO:4}, candidate = GO:2 → GO:2 is an ancestor of GO:4.
    preds = [{"protein_accession": "P1", "go_id": "GO:2"}]
    compute_lineage_features(
        preds,
        parents=_parents(),
        known_by_protein={"P1": {"GO:4"}},
    )
    assert preds[0]["lineage_is_ancestor_of_known"] == 1.0
    assert preds[0]["lineage_is_descendant_of_known"] == 0.0
    assert preds[0]["lineage_ancestor_of_count"] == 1.0


def test_candidate_is_descendant_of_one_known():
    # known = {GO:1}, candidate = GO:4 → GO:1 is in ancestors(GO:4).
    preds = [{"protein_accession": "P1", "go_id": "GO:4"}]
    compute_lineage_features(
        preds,
        parents=_parents(),
        known_by_protein={"P1": {"GO:1"}},
    )
    assert preds[0]["lineage_is_ancestor_of_known"] == 0.0
    assert preds[0]["lineage_is_descendant_of_known"] == 1.0
    assert preds[0]["lineage_descendant_of_count"] == 1.0


def test_candidate_unrelated_to_known():
    # GO:5 is on a different branch from GO:4 (until they share GO:1
    # via the part_of edge).
    # known = {GO:4} only; candidate GO:5 is neither ancestor nor
    # descendant of GO:4 (they are siblings via the DAG).
    preds = [{"protein_accession": "P1", "go_id": "GO:5"}]
    compute_lineage_features(
        preds,
        parents=_parents(),
        known_by_protein={"P1": {"GO:4"}},
    )
    assert preds[0]["lineage_is_ancestor_of_known"] == 0.0
    assert preds[0]["lineage_is_descendant_of_known"] == 0.0


def test_candidate_equals_known_does_not_self_count():
    # candidate = GO:4, known = {GO:4} → must NOT report itself as
    # ancestor or descendant.
    preds = [{"protein_accession": "P1", "go_id": "GO:4"}]
    compute_lineage_features(
        preds,
        parents=_parents(),
        known_by_protein={"P1": {"GO:4"}},
    )
    assert preds[0]["lineage_ancestor_of_count"] == 0.0
    assert preds[0]["lineage_descendant_of_count"] == 0.0
    assert preds[0]["lineage_is_ancestor_of_known"] == 0.0
    assert preds[0]["lineage_is_descendant_of_known"] == 0.0


def test_count_aggregates_over_multiple_known_terms():
    # known = {GO:4, GO:5}; candidate = GO:1 sits above both.
    preds = [{"protein_accession": "P1", "go_id": "GO:1"}]
    compute_lineage_features(
        preds,
        parents=_parents(),
        known_by_protein={"P1": {"GO:4", "GO:5"}},
    )
    assert preds[0]["lineage_ancestor_of_count"] == 2.0
    assert preds[0]["lineage_is_ancestor_of_known"] == 1.0


def test_proteins_isolated_from_each_other():
    # P1 has known {GO:4}; P2 has none. Same candidate GO:2.
    preds = [
        {"protein_accession": "P1", "go_id": "GO:2"},
        {"protein_accession": "P2", "go_id": "GO:2"},
    ]
    compute_lineage_features(
        preds,
        parents=_parents(),
        known_by_protein={"P1": {"GO:4"}},
    )
    assert preds[0]["lineage_ancestor_of_count"] == 1.0
    assert preds[1]["lineage_ancestor_of_count"] == 0.0


def test_part_of_edge_traversed_for_descendant():
    # GO:5 has parents {GO:3, GO:2}. So ancestors(GO:5) = {GO:1, GO:2, GO:3, GO:5}.
    # candidate=GO:5, known={GO:2} → GO:2 ∈ ancestors(GO:5) → descendant=true.
    preds = [{"protein_accession": "P1", "go_id": "GO:5"}]
    compute_lineage_features(
        preds,
        parents=_parents(),
        known_by_protein={"P1": {"GO:2"}},
    )
    assert preds[0]["lineage_is_descendant_of_known"] == 1.0


def test_unknown_go_id_in_predictions_handled():
    # Candidate id not in the parents map: ancestors closure is just itself.
    # known = {GO:4} → no relation expected.
    preds = [{"protein_accession": "P1", "go_id": "GO:99999"}]
    compute_lineage_features(
        preds,
        parents=_parents(),
        known_by_protein={"P1": {"GO:4"}},
    )
    assert preds[0]["lineage_is_ancestor_of_known"] == 0.0
    assert preds[0]["lineage_is_descendant_of_known"] == 0.0


def test_blank_go_id_or_protein_accession_zero_features():
    preds = [
        {"protein_accession": "", "go_id": "GO:2"},
        {"protein_accession": "P1", "go_id": ""},
    ]
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
        ("GO:1", {"GO:4"}, 1, 0),  # root above leaf
        ("GO:4", {"GO:1"}, 0, 1),  # leaf below root
        ("GO:2", {"GO:1"}, 0, 1),  # mid below root
        ("GO:1", {"GO:1"}, 0, 0),  # self does not count
    ],
)
def test_param_sanity(cand, known, expect_anc, expect_desc):
    preds = [{"protein_accession": "P1", "go_id": cand}]
    compute_lineage_features(
        preds, parents=_parents(), known_by_protein={"P1": known}
    )
    assert preds[0]["lineage_ancestor_of_count"] == float(expect_anc)
    assert preds[0]["lineage_descendant_of_count"] == float(expect_desc)
