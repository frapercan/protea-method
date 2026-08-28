"""A row carries the donors that voted for it, so a cut can recount them.

Two things are pinned here. That the ledger holds donors rather than
annotation rows, which is what makes a count taken from it a count of
voters. And that it reaches the row, so a later depth cut has the detail
it needs instead of inheriting an aggregate measured over a wider
neighbourhood.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from protea_method._donor_ledger import DonorLedger
from protea_method.pipeline import PredictConfig, predict

_TERM = 7


class TestTheLedgerCountsDonorsNotPaperwork:
    def test_a_donor_seen_twice_is_one_entry(self) -> None:
        """37.6% of pairs in this corpus carry more than one annotation row."""
        ledger = DonorLedger()
        ledger.record("R1", 1, 1, 0.10)
        ledger.record("R1", 1, 1, 0.10)
        ledger.record("R2", 2, 2, 0.20)
        assert len(ledger) == 2
        assert ledger.accessions == ["R1", "R2"]

    def test_a_repeat_keeps_the_shallowest_sighting(self) -> None:
        """A cut admits a donor by where it first appeared, not where it last did."""
        ledger = DonorLedger()
        ledger.record("R1", 2, 1, 0.20)
        ledger.record("R1", 9, 5, 0.90)
        assert ledger.k_positions == [2]
        assert ledger.distances == [0.20]

    def test_the_entries_stay_parallel(self) -> None:
        ledger = DonorLedger()
        for i, acc in enumerate(("R1", "R2", "R3"), start=1):
            ledger.record(acc, i, i, i / 10)
        assert len(ledger.k_positions) == len(ledger)
        assert len(ledger.distances) == len(ledger)
        assert ledger.sequence_ranks_or_none() == [1, 2, 3]

    def test_a_partly_ranked_ledger_reports_nothing_rather_than_half(self) -> None:
        """Half a column reads as "the rest are missing", which is not the case."""
        ledger = DonorLedger()
        ledger.record("R1", 1, 1, 0.1)
        ledger.record("R2", 2, None, 0.2)
        assert ledger.sequence_ranks_or_none() is None

    def test_an_empty_ledger_is_ranked_by_nothing_rather_than_unranked(self) -> None:
        assert DonorLedger().sequence_ranks_or_none() == []


def _rows(*, annotations_per_donor: int) -> list[dict[str, Any]]:
    """One query, three references, each carrying the term N times over."""
    refs = np.array([[1.0, 0.0], [0.9, 0.1], [0.0, 1.0]], dtype=np.float32)
    ann = [
        {"go_term_id": _TERM, "qualifier": "enables", "evidence_code": code}
        for code in ("EXP", "IDA", "IMP")[:annotations_per_donor]
    ]
    out = predict(
        query_accessions=["Q1"],
        query_embeddings=np.array([[1.0, 0.0]], dtype=np.float32),
        reference_accessions=["R1", "R2", "R3"],
        reference_embeddings=refs,
        annotations={acc: list(ann) for acc in ("R1", "R2", "R3")},
        go_id_map={_TERM: "GO:0000007"},
        go_aspect_map={_TERM: "F"},
        config=PredictConfig(
            k=3,
            compute_v6_features=False,
            sequence_keys={"R1": "s1", "R2": "s1", "R3": "s2"},
        ),
    )
    assert isinstance(out, list)
    return out


class TestTheLedgerReachesTheRow:
    def test_the_row_carries_one_entry_per_donor(self) -> None:
        row = _rows(annotations_per_donor=1)[0]
        assert row["donor_accessions"] == ["R1", "R2", "R3"]
        assert row["donor_k_positions"] == [1, 2, 3]
        assert row["donor_sequence_ranks"] == [1, 1, 2]
        assert row["donor_count"] == 3

    def test_three_annotation_rows_per_donor_are_still_three_donors(self) -> None:
        """The defect this exists to end: vote_count counts the paperwork."""
        row = _rows(annotations_per_donor=3)[0]
        assert row["donor_count"] == 3
        assert row["vote_count"] == 9

    def test_a_cut_can_be_recounted_from_the_row_alone(self) -> None:
        """Depth 2 in sequences admits R1 and R2, which are one sequence."""
        row = _rows(annotations_per_donor=1)[0]
        ranks = row["donor_sequence_ranks"]
        assert sum(1 for r in ranks if r <= 1) == 2
        assert sum(1 for r in ranks if r <= 2) == 3
        by_protein = row["donor_k_positions"]
        assert sum(1 for k in by_protein if k <= 1) == 1
