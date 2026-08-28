"""Depth counted in sequences, from the ranking rule up to the row.

The unit that matters is not the position of a neighbour but the
position of its sequence, because the bank holds 38,694 sequences that
belong to more than one protein. These tests pin the rule, its refusal
to guess, and the fact that the number survives the trip through
``predict`` onto the row PROTEA stores.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pytest

from protea_method._sequence_depth import (
    SequenceIdentityMissingError,
    dense_sequence_ranks,
)
from protea_method.pipeline import PredictConfig, predict

_N: list[tuple[str, float]] = [("A", 0.1), ("B", 0.2), ("C", 0.3), ("D", 0.4)]


def test_a_repeated_sequence_does_not_consume_a_rank() -> None:
    """B repeats A's sequence, so C is the second sequence, not the third."""
    keys = {"A": "s1", "B": "s1", "C": "s2", "D": "s3"}
    assert dense_sequence_ranks(_N, keys) == [1, 1, 2, 3]


def test_ranks_and_positions_agree_when_no_sequence_repeats() -> None:
    keys = {a: f"s{i}" for i, (a, _) in enumerate(_N)}
    assert dense_sequence_ranks(_N, keys) == [1, 2, 3, 4]


def test_one_sequence_held_by_every_neighbour_is_depth_one() -> None:
    """The pathological bank: four proteins, one sequence, one sequence deep."""
    assert dense_sequence_ranks(_N, dict.fromkeys("ABCD", "s1")) == [1, 1, 1, 1]


def test_an_unmapped_neighbour_is_refused_rather_than_numbered() -> None:
    """A guessed rank would be a number nothing produced."""
    with pytest.raises(SequenceIdentityMissingError) as caught:
        dense_sequence_ranks(_N, {"A": "s1"})
    message = str(caught.value)
    assert "3 of 4" in message
    assert "'B'" in message


def test_an_empty_list_ranks_to_nothing() -> None:
    assert dense_sequence_ranks([], {}) == []


def _rows(sequence_keys: dict[str, str] | None) -> list[dict[str, Any]]:
    """Two references that annotate the same term, ranked from one query."""
    refs = np.array([[1.0, 0.0], [0.9, 0.1], [0.0, 1.0]], dtype=np.float32)
    annotations: dict[str, list[dict[str, Any]]] = {
        acc: [{"go_term_id": 7, "qualifier": "enables", "evidence_code": "EXP"}]
        for acc in ("R1", "R2", "R3")
    }
    # Only the third neighbour carries term 9, and it is the second
    # sequence. Position and rank disagree there, which is the whole point.
    annotations["R3"].append({"go_term_id": 9, "qualifier": "enables", "evidence_code": "EXP"})
    out = predict(
        query_accessions=["Q1"],
        query_embeddings=np.array([[1.0, 0.0]], dtype=np.float32),
        reference_accessions=["R1", "R2", "R3"],
        reference_embeddings=refs,
        annotations=annotations,
        go_id_map={7: "GO:0000007", 9: "GO:0000009"},
        go_aspect_map={7: "F", 9: "F"},
        config=PredictConfig(k=3, compute_v6_features=False),
        sequence_keys=sequence_keys,
    )
    assert isinstance(out, list)
    return out


def test_the_row_carries_the_sequence_rank_of_its_shallowest_donor() -> None:
    by_term = {r["go_term_id"]: r for r in _rows({"R1": "s1", "R2": "s1", "R3": "s2"})}
    assert by_term[7]["k_position"] == 1
    assert by_term[7]["sequence_rank"] == 1
    # R2 repeats R1's sequence, so R3 sits at protein position 3 and
    # sequence rank 2. A cut at depth 2 keeps this term; a cut at
    # position 2 would have dropped it.
    assert by_term[9]["k_position"] == 3
    assert by_term[9]["sequence_rank"] == 2


def test_the_column_is_present_and_empty_when_no_map_is_supplied() -> None:
    """Absent is a state the row states, not a state it hides."""
    rows = _rows(None)
    assert rows
    assert all("sequence_rank" in row for row in rows)
    assert all(row["sequence_rank"] is None for row in rows)
