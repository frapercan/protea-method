"""The query can leave after retrieval, as LAFA's search does.

LAFA's embedding baseline (democafa) retrieves its depth from the reference as
it stands and then removes the query's own accession: a query that is one of
its references keeps one neighbour fewer, any other keeps them all. The rule
"accession_after_retrieval" reproduces that. Its margin is zero and its drop is
the accession rule's, so it must leave exactly the neighbours the accession rule
leaves, at the same positions, one fewer when the query was retrieved.
"""

from __future__ import annotations

from types import SimpleNamespace
from typing import Any

import numpy as np
import pytest

from protea_method._self_by_sequence import (
    SELF_EXCLUSION_RULES,
    Queries,
    drop_self,
    resolve_self_exclusion,
    self_exclusion_margin,
)
from protea_method.pipeline import PredictConfig, PredictDiagnostics, predict

_AFTER = "accession_after_retrieval"


def _after(k: int, **kwargs: Any) -> PredictConfig:
    """A configuration under the after-retrieval rule, with the flag it needs."""
    return PredictConfig(k=k, exclude_self_neighbour=True, self_exclusion=_AFTER, **kwargs)


def _accession(k: int, **kwargs: Any) -> PredictConfig:
    return PredictConfig(k=k, exclude_self_neighbour=True, self_exclusion="accession", **kwargs)


def _search(ordered: list[tuple[str, float]], cfg: PredictConfig, bank: list[str]) -> list[str]:
    """What the step does for one query "Q": ask for k plus the margin, then drop."""
    queries = Queries(["Q"], np.zeros((1, 2)))
    asked = cfg.k + self_exclusion_margin(cfg, queries, bank)
    return [accession for accession, _ in drop_self(cfg, [ordered[:asked]], queries)[0]]


class TestTheFlagMustSayTheQueryLeaves:
    def test_it_is_a_rule_of_its_own(self) -> None:
        assert SELF_EXCLUSION_RULES == ("none", "sequence", "accession", _AFTER)
        assert resolve_self_exclusion(_after(5)) == _AFTER

    def test_it_is_refused_when_the_flag_says_the_query_may_stay(self) -> None:
        with pytest.raises(ValueError, match="set exclude_self_neighbour to true"):
            resolve_self_exclusion(PredictConfig(self_exclusion=_AFTER))

    def test_a_configuration_without_the_flag_reads_it_as_false(self) -> None:
        with pytest.raises(ValueError, match="set exclude_self_neighbour to true"):
            resolve_self_exclusion(SimpleNamespace(self_exclusion=_AFTER))

    @pytest.mark.parametrize("rule", ["sequence", "accession"])
    def test_the_older_rules_still_decide_over_an_unset_flag(self, rule: str) -> None:
        assert resolve_self_exclusion(PredictConfig(self_exclusion=rule)) == rule


class TestTheMarginIsZero:
    @pytest.mark.parametrize("bank", [["Q", "R1", "R2"], ["R1", "R2"]], ids=["in-pool", "not"])
    def test_whether_or_not_the_query_is_in_the_pool(self, bank: list[str]) -> None:
        queries = Queries(["Q"], np.zeros((1, 2)))
        assert self_exclusion_margin(_after(5), queries, bank) == 0

    def test_where_the_accession_rule_asks_for_one_more(self) -> None:
        queries = Queries(["Q"], np.zeros((1, 2)))
        assert self_exclusion_margin(_accession(5), queries, ["Q", "R1", "R2"]) == 1

    def test_even_with_twins_and_sequence_keys(self) -> None:
        keys = {"Q": "s", "T1": "s", "T2": "s", "R1": "r"}
        queries = Queries(["Q"], np.zeros((1, 2)))
        assert self_exclusion_margin(_after(5, sequence_keys=keys), queries, list(keys)) == 0


class TestTheDrop:
    def test_a_query_in_the_pool_keeps_k_minus_one(self) -> None:
        hits = [[("Q", 0.0), ("R1", 0.1), ("R2", 0.2)]]
        assert drop_self(_after(3), hits, Queries(["Q"], np.zeros((1, 2)))) == [
            [("R1", 0.1), ("R2", 0.2)]
        ]

    def test_a_query_outside_the_pool_keeps_k(self) -> None:
        hits = [[("R1", 0.1), ("R2", 0.2), ("R3", 0.3)]]
        assert drop_self(_after(3), hits, Queries(["Q"], np.zeros((1, 2)))) == hits

    def test_one_batch_holds_both_kinds_of_query(self) -> None:
        hits = [[("Q", 0.0), ("R1", 0.1), ("R2", 0.2)], [("R1", 0.1), ("R2", 0.2), ("R3", 0.3)]]
        kept = drop_self(_after(3), hits, Queries(["Q", "P"], np.zeros((2, 2))))
        assert [len(neighbours) for neighbours in kept] == [2, 3]

    def test_the_twins_stay_as_donors(self) -> None:
        hits = [[("Q", 0.0), ("T1", 0.0), ("R2", 0.4)]]
        assert drop_self(_after(3), hits, Queries(["Q"], np.zeros((1, 2)))) == [
            [("T1", 0.0), ("R2", 0.4)]
        ]

    def test_lists_and_queries_of_unequal_length_are_refused(self) -> None:
        """They are paired by position, so a mismatch would hand out other neighbourhoods."""
        hits = [[("Q", 0.0), ("R1", 0.1)]]
        with pytest.raises(ValueError, match="paired by position"):
            drop_self(_after(2), hits, Queries(["Q", "P"], np.zeros((2, 2))))


class TestThePositionsAreTheAccessionRules:
    @pytest.mark.parametrize("k", [1, 3, 5])
    def test_one_fewer_and_otherwise_the_same_when_the_query_is_in_the_pool(self, k: int) -> None:
        bank = ["Q", *[f"R{i}" for i in range(10)]]
        ordered = [("Q", 0.0)] + [(f"R{i}", 0.1 + i / 100) for i in range(10)]
        by_accession = _search(ordered, _accession(k), bank)
        after = _search(ordered, _after(k), bank)
        assert len(by_accession) == k
        assert len(after) == k - 1
        assert after == by_accession[: k - 1]

    def test_the_same_when_the_query_is_not_in_the_pool(self) -> None:
        bank = [f"R{i}" for i in range(10)]
        ordered = [(f"R{i}", 0.1 + i / 100) for i in range(10)]
        assert _search(ordered, _after(5), bank) == _search(ordered, _accession(5), bank)

    def test_a_big_twin_group_is_kept_nearest_first(self) -> None:
        """127 twins at distance zero, the largest group in the platform's corpus."""
        twins = [f"T{i}" for i in range(127)]
        bank = ["Q", *twins]
        ordered = [("Q", 0.0)] + [(t, 0.0) for t in twins]
        assert _search(ordered, _after(5), bank) == twins[:4]


def _run(
    rule: str, k: int, references: list[str], aspect_separated: bool = False
) -> tuple[list[dict[str, Any]], PredictDiagnostics]:
    """Predict for the query "Q" against named references of a fixed geometry.

    Each reference donates one term of its own, so a row is one donor and its
    ``k_position`` is that donor's position in the neighbour list.
    """
    angle = {"Q": 0.0, "A": 0.1, "B": 0.3, "C": 0.6, "D": 0.9, "E": 1.2}
    terms = {"Q": [1], "A": [2, 3], "B": [4, 5], "C": [6], "D": [7], "E": [8]}
    aspects = {1: "F", 2: "F", 3: "P", 4: "F", 5: "P", 6: "P", 7: "F", 8: "F"}
    cfg = PredictConfig(
        k=k,
        compute_v6_features=False,
        exclude_self_neighbour=True,
        self_exclusion=rule,
        aspect_separated=aspect_separated,
    )
    return predict(
        query_accessions=["Q"],
        query_embeddings=np.array([[1.0, 0.0]], dtype=np.float32),
        reference_accessions=references,
        reference_embeddings=np.array(
            [[np.cos(angle[r]), np.sin(angle[r])] for r in references], dtype=np.float32
        ),
        annotations={
            r: [{"go_term_id": t, "qualifier": "enables", "evidence_code": "EXP"} for t in terms[r]]
            for r in references
        },
        go_id_map={t: f"GO:{t:07d}" for t in aspects},
        go_aspect_map=aspects,
        config=cfg,
        return_diagnostics=True,
    )


def _positions(rows: list[dict[str, Any]]) -> dict[str, int]:
    return {row["ref_protein_accession"]: int(row["k_position"]) for row in rows}


class TestItReachesTheRows:
    """End to end: the rows a predict call writes under each rule."""

    def test_a_query_in_the_pool_is_read_one_shallower_at_the_same_positions(self) -> None:
        bank = ["Q", "A", "B", "C", "D"]
        by_accession, _ = _run("accession", 3, bank)
        after, diagnostics = _run(_AFTER, 3, bank)
        assert _positions(by_accession) == {"A": 1, "B": 2, "C": 3}
        assert _positions(after) == {"A": 1, "B": 2}
        assert [a for a, _ in diagnostics.neighbors_by_aspect[""][0]] == ["A", "B"]

    def test_a_query_outside_the_pool_keeps_its_depth(self) -> None:
        bank = ["A", "B", "C", "D"]
        after, _ = _run(_AFTER, 3, bank)
        by_accession, _ = _run("accession", 3, bank)
        assert _positions(after) == _positions(by_accession) == {"A": 1, "B": 2, "C": 3}

    def test_aspect_by_aspect_the_pool_decides(self) -> None:
        """The query is in the molecular function pool only, so only there it costs depth."""
        _, diagnostics = _run(_AFTER, 2, ["Q", "A", "B", "C", "D", "E"], aspect_separated=True)
        neighbours = {
            aspect: [a for a, _ in lists[0]]
            for aspect, lists in diagnostics.neighbors_by_aspect.items()
        }
        assert neighbours["F"] == ["A"]
        assert neighbours["P"] == ["A", "B"]
        assert neighbours["C"] == []
