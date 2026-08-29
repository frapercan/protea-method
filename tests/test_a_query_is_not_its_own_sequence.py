"""Excluding by accession removes an identity and leaves a similarity.

In this bank 38,694 sequences belong to more than one protein, and 784 of
the 14,032 queries in the evaluation window share their sequence with
another protein. For those, dropping the query's own accession leaves a
copy of it at distance zero under a different name, so the flag would
report an exclusion it did not perform.

The discriminating case here is exactly that one: a query, its twin, and
a genuine neighbour. Accession-based exclusion keeps the twin. Sequence
based exclusion does not.
"""

from __future__ import annotations

import numpy as np
import pytest

from protea_method._self_by_sequence import (
    Queries,
    drop_own_sequence,
    extra_neighbours_for,
    without_own_sequence,
)
from protea_method._sequence_depth import SequenceIdentityMissingError
from protea_method.pipeline import PredictConfig, predict

#: Q1 and its twin T1 are one sequence under two names. R2 is somebody else.
_KEYS = {"Q1": "s1", "T1": "s1", "R2": "s2", "R3": "s3"}


class TestTheMarginIsMeasuredNotGuessed:
    def test_a_query_with_no_twin_costs_nothing(self) -> None:
        assert extra_neighbours_for(["Q1"], ["R2", "R3"], _KEYS) == 0

    def test_a_query_in_the_bank_asks_for_one_more(self) -> None:
        assert extra_neighbours_for(["Q1"], ["Q1", "R2"], _KEYS) == 1

    def test_a_query_with_a_twin_asks_for_two(self) -> None:
        """Q1 and T1 are the same sequence, so both have to be droppable."""
        assert extra_neighbours_for(["Q1"], ["Q1", "T1", "R2"], _KEYS) == 2

    def test_the_margin_is_the_worst_query_not_the_average(self) -> None:
        """One query with many twins must not come back short while the
        others are fine, because a shortfall on one is invisible in an
        aggregate."""
        keys = {"A": "s1", "B": "s2", **{f"T{i}": "s2" for i in range(5)}}
        bank = ["A", "B", *[f"T{i}" for i in range(5)]]
        assert extra_neighbours_for(["A", "B"], bank, keys) == 6

    def test_a_query_with_no_sequence_is_refused(self) -> None:
        with pytest.raises(SequenceIdentityMissingError, match="cannot be excluded"):
            extra_neighbours_for(["UNKNOWN"], ["R2"], _KEYS)


class TestTheTwinGoesToo:
    def test_the_twin_is_dropped_and_the_neighbour_is_not(self) -> None:
        """The case accession-based exclusion gets wrong."""
        got = without_own_sequence(
            [[("Q1", 0.0), ("T1", 0.0), ("R2", 0.4)]], ["Q1"], 2, _KEYS
        )
        assert got == [[("R2", 0.4)]]

    def test_a_query_absent_from_the_bank_keeps_everything(self) -> None:
        got = without_own_sequence([[("R2", 0.1), ("R3", 0.2)]], ["Q1"], 2, _KEYS)
        assert got == [[("R2", 0.1), ("R3", 0.2)]]

    def test_the_cut_to_k_happens_after_the_drop(self) -> None:
        """Otherwise asking for k and dropping two scores k minus two."""
        got = without_own_sequence(
            [[("Q1", 0.0), ("T1", 0.0), ("R2", 0.4), ("R3", 0.5)]], ["Q1"], 2, _KEYS
        )
        assert got == [[("R2", 0.4), ("R3", 0.5)]]

    def test_an_unequal_pairing_is_refused(self) -> None:
        with pytest.raises(ValueError, match="somebody else's neighbourhood"):
            without_own_sequence([[("R2", 0.1)]], ["Q1", "Q2"], 2, _KEYS)


class TestTheFlagGovernsIt:
    def test_off_by_default_so_stored_runs_keep_their_meaning(self) -> None:
        cfg = PredictConfig(k=2, sequence_keys=_KEYS)
        assert cfg.exclude_self_neighbour is False
        rows = [[("Q1", 0.0), ("R2", 0.4)]]
        assert drop_own_sequence(cfg, rows, Queries(["Q1"], np.zeros((1, 2)))) == rows

    def test_without_a_sequence_map_it_does_nothing_rather_than_guess(self) -> None:
        cfg = PredictConfig(k=2, exclude_self_neighbour=True)
        rows = [[("Q1", 0.0), ("R2", 0.4)]]
        assert drop_own_sequence(cfg, rows, Queries(["Q1"], np.zeros((1, 2)))) == rows


def test_the_flag_reaches_the_row() -> None:
    """End to end: the query is in the bank, and must not donate to itself."""
    refs = np.array([[1.0, 0.0], [1.0, 0.0], [0.0, 1.0]], dtype=np.float32)
    ann = {
        acc: [{"go_term_id": 7, "qualifier": "enables", "evidence_code": "EXP"}]
        for acc in ("Q1", "T1", "R2")
    }
    out = predict(
        query_accessions=["Q1"],
        query_embeddings=np.array([[1.0, 0.0]], dtype=np.float32),
        reference_accessions=["Q1", "T1", "R2"],
        reference_embeddings=refs,
        annotations=ann,
        go_id_map={7: "GO:0000007"},
        go_aspect_map={7: "F"},
        config=PredictConfig(
            k=1, compute_v6_features=False,
            exclude_self_neighbour=True, sequence_keys=_KEYS,
        ),
    )
    assert isinstance(out, list)
    assert out, "the query lost every donor, so the margin was too small"
    donors = {row["ref_protein_accession"] for row in out}
    assert donors == {"R2"}, f"Q1 or its twin T1 still donated: {donors}"
