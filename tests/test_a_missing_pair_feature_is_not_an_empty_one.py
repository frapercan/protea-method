"""A pair whose features were never computed says so.

WHY THIS TEST EXISTS. ``pair_features`` is built by the caller from the
caller's own neighbour search, and ``predict`` then runs its own. The two agree
almost always and not exactly: two searches over differently sliced float
arrays disagree at the 1e-7 level, so wherever the k-th distance is a tie they
can keep different donors. A row for a donor the caller never asked about used
to be emitted with all fifteen pair-feature columns NULL and no signal of any
kind, which made it identical to a row whose features were genuinely
uncomputable.

That is what happened on 2026-08-29: 76 rows of a 2,441,584 row run, all at
k_position 28 to 30, with donors enriched 43-fold in shared sequences (a shared
sequence is an identical embedding, hence an exact tie). 33 of the 37 pairs
already had their alignment computed and stored. The alignment was there for
the taking and the run wrote NULL, and nothing in the output said so.

The distinction this asserts is between two things that produce the same row:
an empty dict, which is a computed answer, and a missing key, which is a
question nobody asked. Only the second is a miss.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from protea_method.pipeline import PredictConfig, PredictDiagnostics, predict

GO_ID_MAP = {1: "GO:0000001", 2: "GO:0000002"}
GO_ASPECT_MAP = {1: "F", 2: "P"}


def _corpus() -> tuple[list[str], np.ndarray, list[str], np.ndarray, dict[str, Any]]:
    rng = np.random.default_rng(7)
    annotations = {
        "R00": [{"go_term_id": 1}],
        "R01": [{"go_term_id": 2}],
        "R02": [{"go_term_id": 1}],
        "R03": [{"go_term_id": 2}],
    }
    return (
        ["Q1"],
        rng.standard_normal(size=(1, 4)).astype(np.float32),
        [f"R{i:02d}" for i in range(4)],
        rng.standard_normal(size=(4, 4)).astype(np.float32),
        annotations,
    )


def _run(
    pair_features: dict[tuple[str, str], dict[str, Any]] | None,
) -> tuple[list[dict[str, Any]], PredictDiagnostics]:
    qa, qe, ra, re_, anns = _corpus()
    return predict(
        query_accessions=qa,
        query_embeddings=qe,
        reference_accessions=ra,
        reference_embeddings=re_,
        annotations=anns,
        go_id_map=GO_ID_MAP,
        go_aspect_map=GO_ASPECT_MAP,
        config=PredictConfig(k=3, compute_v6_features=False),
        pair_features=pair_features,
        return_diagnostics=True,
    )


def _donors(rows: list[dict[str, Any]]) -> set[tuple[str, str]]:
    return {(r["protein_accession"], r["ref_protein_accession"]) for r in rows}


def test_a_donor_nobody_asked_about_is_reported() -> None:
    rows, _ = _run(None)
    asked = _donors(rows)
    assert len(asked) >= 2, "the corpus has to produce at least two donors to drop one"

    kept, dropped = sorted(asked)[0], sorted(asked)[-1]
    rows, diag = _run({kept: {"identity_nw": 0.5}})

    assert dropped in diag.pair_feature_misses
    assert kept not in diag.pair_feature_misses
    # The row is still emitted. Refusing to emit it would lose the donor's
    # vote, which is correct and present; what is missing is its columns.
    assert dropped in _donors(rows)


def test_an_empty_dict_is_an_answer_and_not_a_miss() -> None:
    """The distinction the whole change rests on.

    A caller that computed a pair's features and found none has answered the
    question. Counting that as a miss would make the number unusable, because
    the common case would drown the rare one.
    """
    rows, _ = _run(None)
    every = _donors(rows)
    _, diag = _run({pair: {} for pair in every})
    assert diag.pair_feature_misses == frozenset()


def test_asking_nothing_reports_nothing() -> None:
    """A caller that passes no pair_features asked no questions.

    Reporting every donor as a miss would fire on every existing caller that
    does not use the feature at all.
    """
    _, diag = _run(None)
    assert diag.pair_feature_misses == frozenset()


def test_a_complete_map_reports_no_miss() -> None:
    rows, _ = _run(None)
    complete = {pair: {"identity_nw": 0.9} for pair in _donors(rows)}
    rows, diag = _run(complete)
    assert diag.pair_feature_misses == frozenset()
    assert all(r.get("identity_nw") == 0.9 for r in rows)


def test_the_field_list_carries_the_taxonomy_a_caller_computes() -> None:
    """Three fields were computed, paid for, and dropped in silence.

    PAIR_FEATURE_KEYS is the only gate between what a caller computes and what
    reaches its database, and it omitted taxonomic_lca and the two taxonomy ids
    that PROTEA's adapter sets beside them. The three sibling taxonomy columns
    were filled on the same rows, so nothing looked wrong: 0 non-null values in
    7,082,480 rows across every prediction set that exists.

    Named one by one rather than counted, because a count passes when a field
    is swapped for another.
    """
    from protea_method.pipeline import PAIR_FEATURE_KEYS

    for field in (
        "taxonomic_lca",
        "taxonomic_distance",
        "taxonomic_common_ancestors",
        "taxonomic_relation",
        "query_taxonomy_id",
        "ref_taxonomy_id",
    ):
        assert field in PAIR_FEATURE_KEYS, field


def test_the_list_has_no_duplicates() -> None:
    """A duplicate would hide a rename: the old and new name both present."""
    from protea_method.pipeline import PAIR_FEATURE_KEYS

    assert len(PAIR_FEATURE_KEYS) == len(set(PAIR_FEATURE_KEYS))


def test_a_field_in_the_list_reaches_the_row() -> None:
    """The list is only a promise until something copies by it."""
    from protea_method.pipeline import PAIR_FEATURE_KEYS, propagate_pair_features

    row: dict[str, Any] = {}
    propagate_pair_features(row, {k: i for i, k in enumerate(PAIR_FEATURE_KEYS)})
    assert set(row) == set(PAIR_FEATURE_KEYS)
