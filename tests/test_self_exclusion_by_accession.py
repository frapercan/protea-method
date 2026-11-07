"""The self-exclusion rule is a parameter, and each rule's margin agrees with its drop.

The author chose on 2026-11-07, for a consumer that already reads the query's own
state as input, to exclude the query by ACCESSION and keep its twins, which are
other entries annotated by others. The kNN method keeps its own decision (the
self-vote stays, rule "none"). Mixing one rule's margin with another rule's drop
once left 887 donor holes, so every test here checks the two together.
"""

from __future__ import annotations

import numpy as np
import pytest

from protea_method._self_by_sequence import (
    Queries,
    drop_self,
    extra_neighbours_for_accession,
    resolve_self_exclusion,
    self_exclusion_margin,
    without_own_accession,
)
from protea_method.pipeline import PredictConfig, predict

_KEYS = {"Q1": "s1", "T1": "s1", "R2": "s2", "R3": "s3"}


def test_the_old_flag_keeps_its_meaning_when_no_rule_is_set() -> None:
    assert resolve_self_exclusion(PredictConfig()) == "none"
    assert resolve_self_exclusion(PredictConfig(exclude_self_neighbour=True)) == "sequence"
    assert resolve_self_exclusion(PredictConfig(self_exclusion="accession")) == "accession"
    assert resolve_self_exclusion(PredictConfig(exclude_self_neighbour=True, self_exclusion="accession")) == "accession"


def test_contradicting_or_unknown_rules_are_refused() -> None:
    with pytest.raises(ValueError, match="not both"):
        resolve_self_exclusion(PredictConfig(exclude_self_neighbour=True, self_exclusion="none"))
    with pytest.raises(ValueError, match="must be one of"):
        resolve_self_exclusion(PredictConfig(self_exclusion="twins"))


def test_the_accession_margin_is_the_query_itself() -> None:
    assert extra_neighbours_for_accession(["Q1"], ["Q1", "T1", "R2"]) == 1
    assert extra_neighbours_for_accession(["Q1"], ["T1", "R2"]) == 0


def test_the_accession_rule_keeps_the_twin() -> None:
    got = without_own_accession([[("Q1", 0.0), ("T1", 0.0), ("R2", 0.4)]], ["Q1"], 2)
    assert got == [[("T1", 0.0), ("R2", 0.4)]]


@pytest.mark.parametrize("rule", ["none", "sequence", "accession"])
def test_margin_and_drop_agree_on_a_bank_with_a_big_twin_group(rule: str) -> None:
    """A query in the bank with 127 twins, the largest group in the platform's corpus."""
    k = 5
    twins = [f"T{i}" for i in range(127)]
    others = [f"R{i}" for i in range(20)]
    keys = {"Q": "sq", **dict.fromkeys(twins, "sq"), **{r: f"s{r}" for r in others}}
    bank = ["Q", *twins, *others]
    ordered = [("Q", 0.0)] + [(t, 0.0) for t in twins] + [(r, 0.1 + i / 100) for i, r in enumerate(others)]
    cfg = PredictConfig(k=k, sequence_keys=keys, self_exclusion=rule)
    queries = Queries(["Q"], np.zeros((1, 2)))
    asked = k + self_exclusion_margin(cfg, queries, bank)
    kept = drop_self(cfg, [ordered[:asked]], queries)[0]
    names = [a for a, _ in kept]
    if rule == "none":
        assert names[0] == "Q" and len(names) == asked == k
    else:
        assert len(names) == k, f"{rule}: the margin did not cover the drop"
        assert "Q" not in names
        if rule == "accession":
            assert names == twins[:k]  # the twins are kept, nearest first
        else:
            assert names == others[:k]  # every twin went with the query


def test_the_accession_rule_reaches_the_row() -> None:
    """End to end: the query does not donate to itself, its twin does."""
    refs = np.array([[1.0, 0.0], [1.0, 0.0], [0.0, 1.0]], dtype=np.float32)
    ann = {acc: [{"go_term_id": 7, "qualifier": "enables", "evidence_code": "EXP"}] for acc in ("Q1", "T1", "R2")}
    out = predict(
        query_accessions=["Q1"],
        query_embeddings=np.array([[1.0, 0.0]], dtype=np.float32),
        reference_accessions=["Q1", "T1", "R2"],
        reference_embeddings=refs,
        annotations=ann,
        go_id_map={7: "GO:0000007"},
        go_aspect_map={7: "F"},
        config=PredictConfig(k=1, compute_v6_features=False, sequence_keys=_KEYS, self_exclusion="accession"),
    )
    assert isinstance(out, list)
    donors = {row["ref_protein_accession"] for row in out}
    assert donors == {"T1"}, f"the query donated to itself, or its twin was dropped: {donors}"
