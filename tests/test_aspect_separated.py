"""Contract tests for the ``aspect_separated=True`` KNN path.

Pinned to the four invariants stated in the F2C.6 slice spec:

1. three per-aspect indices are built (``F``, ``P``, ``C``);
2. candidates per query only come from refs annotated in the matching
   aspect (no cross-aspect bleed via the unified ref list);
3. when a per-aspect booster is supplied, it scores its aspect's rows
   only and is invoked exactly once per aspect per ``predict`` call
   (no per-query call explosion);
4. row ordering is deterministic for repeated invocations with the same
   inputs.

The slice also adds a marker-gated regression test
(``@pytest.mark.regression_v226``) that exercises the LB.2 champion
config end-to-end when ``PROTEA_LAB_DATA`` points at a directory with
the lab fixtures; it ``pytest.skip``s otherwise so the suite stays
runnable in the protea-method standalone CI environment.
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from unittest.mock import patch

import lightgbm as lgb
import numpy as np
import pandas as pd
import pytest

from protea_method.pipeline import (
    PredictConfig,
    load_boosters_by_aspect,
    predict,
)
from protea_method.reranker import ALL_FEATURES, LABEL_COLUMN, prepare_dataset

# ---------------------------------------------------------------------------
# Fixtures: tiny aspect-balanced corpus (5 references per aspect, 3 queries)
# ---------------------------------------------------------------------------


def _aspect_balanced_corpus() -> tuple[
    list[str],
    np.ndarray,
    list[str],
    np.ndarray,
    dict[str, list[dict[str, object]]],
    dict[int, str],
    dict[int, str],
]:
    """Build a deterministic corpus with 5 refs per aspect and 3 queries.

    GO term layout (one term per aspect, easy to assert against):

    - ``1`` (``GO:0000001``) -> ``F``
    - ``2`` (``GO:0000002``) -> ``P``
    - ``3`` (``GO:0000003``) -> ``C``

    Reference layout (15 refs total, 5 per aspect, no cross-tagged refs):

    - ``RF0..RF4`` -> only ``F`` annotations;
    - ``RP0..RP4`` -> only ``P`` annotations;
    - ``RC0..RC4`` -> only ``C`` annotations.

    Refs disjoint by aspect lets us assert post-hoc that a query
    candidate only comes from its aspect's bank.
    """
    rng = np.random.default_rng(20260517)
    query_accessions = ["Q1", "Q2", "Q3"]
    query_embeddings = rng.standard_normal(size=(3, 8)).astype(np.float32)

    ref_accessions: list[str] = []
    ref_embeddings_rows: list[np.ndarray] = []
    annotations: dict[str, list[dict[str, object]]] = {}
    for letter, gtid in (("F", 1), ("P", 2), ("C", 3)):
        for j in range(5):
            acc = f"R{letter}{j}"
            ref_accessions.append(acc)
            ref_embeddings_rows.append(rng.standard_normal(size=(8,)).astype(np.float32))
            annotations[acc] = [
                {"go_term_id": gtid, "qualifier": "enables", "evidence_code": "IEA"},
            ]
    ref_embeddings = np.stack(ref_embeddings_rows, axis=0)
    go_id_map = {1: "GO:0000001", 2: "GO:0000002", 3: "GO:0000003"}
    go_aspect_map = {1: "F", 2: "P", 3: "C"}
    return (
        query_accessions,
        query_embeddings,
        ref_accessions,
        ref_embeddings,
        annotations,
        go_id_map,
        go_aspect_map,
    )


def _train_aspect_booster(rng_seed: int) -> lgb.Booster:
    """Train a tiny LightGBM booster against random labels for a given seed."""
    rng = np.random.default_rng(rng_seed)
    n = 200
    train_rows: dict[str, list[object]] = {
        "protein_accession": [f"P{i}" for i in range(n)],
        "go_id": [f"GO:{i}" for i in range(n)],
        "aspect": rng.choice(["F", "P", "C"], n).tolist(),
        LABEL_COLUMN: rng.integers(0, 2, n).tolist(),
    }
    for col in ALL_FEATURES:
        train_rows.setdefault(col, rng.random(n).tolist())
    train_df = pd.DataFrame(train_rows)
    X, y = prepare_dataset(train_df)
    return lgb.train(
        {"objective": "binary", "verbose": -1, "num_leaves": 7, "learning_rate": 0.1},
        lgb.Dataset(X, label=y),
        num_boost_round=5,
    )


# ---------------------------------------------------------------------------
# Invariant 1: three indices built
# ---------------------------------------------------------------------------


def test_aspect_separated_partitions_into_three_banks() -> None:
    """``_partition_refs_by_aspect`` produces one bank per aspect code."""
    from protea_method.pipeline import _partition_refs_by_aspect

    _, _, ref_accessions, ref_embeddings, annotations, _, go_aspect_map = (
        _aspect_balanced_corpus()
    )
    banks = _partition_refs_by_aspect(
        ref_accessions, ref_embeddings, annotations, go_aspect_map,
    )
    assert set(banks.keys()) == {"F", "P", "C"}
    for aspect in ("F", "P", "C"):
        accs, embs = banks[aspect]
        # 5 refs per aspect in the balanced corpus.
        assert len(accs) == 5
        assert embs.shape == (5, ref_embeddings.shape[1])
        # No ref appears in another aspect's bank.
        for other in ("F", "P", "C"):
            if other == aspect:
                continue
            assert not set(accs) & set(banks[other][0])


def test_aspect_separated_invokes_search_knn_once_per_aspect() -> None:
    """The orchestrator calls ``search_knn`` exactly three times: F / P / C."""
    qa, qe, ra, re_, anns, gid_map, gasp_map = _aspect_balanced_corpus()
    with patch(
        "protea_method.pipeline.search_knn",
        wraps=__import__("protea_method.pipeline", fromlist=["search_knn"]).search_knn,
    ) as spy:
        predict(
            query_accessions=qa,
            query_embeddings=qe,
            reference_accessions=ra,
            reference_embeddings=re_,
            annotations=anns,
            go_id_map=gid_map,
            go_aspect_map=gasp_map,
            config=PredictConfig(k=3, aspect_separated=True, compute_v6_features=False),
        )
    assert spy.call_count == 3, (
        f"expected exactly three KNN searches (one per aspect), got {spy.call_count}"
    )


# ---------------------------------------------------------------------------
# Invariant 2: candidates per query only come from the matching aspect
# ---------------------------------------------------------------------------


def test_aspect_separated_candidates_stay_within_aspect() -> None:
    """A row's ``ref_protein_accession`` lives in that row's aspect bank."""
    qa, qe, ra, re_, anns, gid_map, gasp_map = _aspect_balanced_corpus()
    preds = predict(
        query_accessions=qa,
        query_embeddings=qe,
        reference_accessions=ra,
        reference_embeddings=re_,
        annotations=anns,
        go_id_map=gid_map,
        go_aspect_map=gasp_map,
        config=PredictConfig(k=3, aspect_separated=True, compute_v6_features=False),
    )
    assert preds
    bank_prefix = {"F": "RF", "P": "RP", "C": "RC"}
    for p in preds:
        aspect = p["aspect"]
        donor = p["ref_protein_accession"]
        assert donor.startswith(bank_prefix[aspect]), (
            f"row aspect={aspect} ref={donor} crossed aspect boundary"
        )


def test_aspect_separated_emits_predictions_for_every_aspect() -> None:
    """Each query produces at least one prediction per aspect when the
    bank for that aspect is non-empty."""
    qa, qe, ra, re_, anns, gid_map, gasp_map = _aspect_balanced_corpus()
    preds = predict(
        query_accessions=qa,
        query_embeddings=qe,
        reference_accessions=ra,
        reference_embeddings=re_,
        annotations=anns,
        go_id_map=gid_map,
        go_aspect_map=gasp_map,
        config=PredictConfig(k=2, aspect_separated=True, compute_v6_features=False),
    )
    per_query_aspects: dict[str, set[str]] = {q: set() for q in qa}
    for p in preds:
        per_query_aspects[p["protein_accession"]].add(p["aspect"])
    for q, aspects in per_query_aspects.items():
        assert aspects == {"F", "P", "C"}, (
            f"query {q} only covers {aspects}; expected all three aspects"
        )


# ---------------------------------------------------------------------------
# Invariant 3: booster routed once per aspect per call (not per query)
# ---------------------------------------------------------------------------


def test_aspect_separated_booster_called_once_per_aspect() -> None:
    """``_score_per_aspect`` invokes ``apply_reranker`` once per aspect with
    a non-empty mapping, regardless of how many queries are in the batch."""
    qa, qe, ra, re_, anns, gid_map, gasp_map = _aspect_balanced_corpus()
    boosters = {
        "F": _train_aspect_booster(1),
        "P": _train_aspect_booster(2),
        "C": _train_aspect_booster(3),
    }
    with patch(
        "protea_method.pipeline.apply_reranker",
        wraps=__import__(
            "protea_method.pipeline", fromlist=["apply_reranker"],
        ).apply_reranker,
    ) as spy:
        preds = predict(
            query_accessions=qa,
            query_embeddings=qe,
            reference_accessions=ra,
            reference_embeddings=re_,
            annotations=anns,
            go_id_map=gid_map,
            go_aspect_map=gasp_map,
            config=PredictConfig(
                k=3, aspect_separated=True, compute_v6_features=False,
            ),
            boosters_by_aspect=boosters,
        )
    # Three aspects, three calls.
    assert spy.call_count == 3
    # Every prediction got a score, no aspect was skipped.
    assert preds
    for p in preds:
        assert "reranker_score" in p
        assert 0.0 <= p["reranker_score"] <= 1.0


def test_aspect_separated_booster_partial_coverage_leaves_holes() -> None:
    """If only ``F`` and ``C`` are covered, ``P`` rows stay unscored and
    the booster is called exactly twice."""
    qa, qe, ra, re_, anns, gid_map, gasp_map = _aspect_balanced_corpus()
    boosters = {
        "F": _train_aspect_booster(1),
        "C": _train_aspect_booster(2),
    }
    with patch(
        "protea_method.pipeline.apply_reranker",
        wraps=__import__(
            "protea_method.pipeline", fromlist=["apply_reranker"],
        ).apply_reranker,
    ) as spy:
        preds = predict(
            query_accessions=qa,
            query_embeddings=qe,
            reference_accessions=ra,
            reference_embeddings=re_,
            annotations=anns,
            go_id_map=gid_map,
            go_aspect_map=gasp_map,
            config=PredictConfig(
                k=3, aspect_separated=True, compute_v6_features=False,
            ),
            boosters_by_aspect=boosters,
        )
    assert spy.call_count == 2
    for p in preds:
        if p["aspect"] == "P":
            assert "reranker_score" not in p
        else:
            assert "reranker_score" in p


# ---------------------------------------------------------------------------
# Invariant 4: deterministic ordering
# ---------------------------------------------------------------------------


def test_aspect_separated_predictions_are_deterministic() -> None:
    """Two runs with the same inputs produce identical row sequences."""
    qa, qe, ra, re_, anns, gid_map, gasp_map = _aspect_balanced_corpus()
    cfg = PredictConfig(k=3, aspect_separated=True, compute_v6_features=False)
    a = predict(
        query_accessions=qa,
        query_embeddings=qe,
        reference_accessions=ra,
        reference_embeddings=re_,
        annotations=anns,
        go_id_map=gid_map,
        go_aspect_map=gasp_map,
        config=cfg,
    )
    b = predict(
        query_accessions=qa,
        query_embeddings=qe,
        reference_accessions=ra,
        reference_embeddings=re_,
        annotations=anns,
        go_id_map=gid_map,
        go_aspect_map=gasp_map,
        config=cfg,
    )
    assert len(a) == len(b)
    for ra_row, rb_row in zip(a, b, strict=True):
        assert ra_row["protein_accession"] == rb_row["protein_accession"]
        assert ra_row["go_term_id"] == rb_row["go_term_id"]
        assert ra_row["aspect"] == rb_row["aspect"]
        assert ra_row["ref_protein_accession"] == rb_row["ref_protein_accession"]
        assert ra_row["min_distance"] == pytest.approx(rb_row["min_distance"])
        assert ra_row["vote_count"] == rb_row["vote_count"]


# ---------------------------------------------------------------------------
# Booster-shape contract
# ---------------------------------------------------------------------------


def test_validate_aspect_boosters_rejects_long_form_keys() -> None:
    """``boosters_by_aspect={'mfo': ...}`` raises rather than silently
    routing zero rows."""
    qa, qe, ra, re_, anns, gid_map, gasp_map = _aspect_balanced_corpus()
    bad_boosters = {"mfo": _train_aspect_booster(1)}  # long-form key, not F/P/C
    with pytest.raises(ValueError, match="disallowed keys"):
        predict(
            query_accessions=qa,
            query_embeddings=qe,
            reference_accessions=ra,
            reference_embeddings=re_,
            annotations=anns,
            go_id_map=gid_map,
            go_aspect_map=gasp_map,
            config=PredictConfig(
                k=2, aspect_separated=True, compute_v6_features=False,
            ),
            boosters_by_aspect=bad_boosters,
        )


def test_load_boosters_by_aspect_short_codes(tmp_path: Path) -> None:
    """``load_boosters_by_aspect`` reads ``F.txt`` / ``P.txt`` / ``C.txt``."""
    for letter, seed in (("F", 1), ("P", 2), ("C", 3)):
        b = _train_aspect_booster(seed)
        b.save_model(str(tmp_path / f"{letter}.txt"))
    out = load_boosters_by_aspect(tmp_path)
    assert set(out.keys()) == {"F", "P", "C"}
    for v in out.values():
        assert isinstance(v, lgb.Booster)


def test_load_boosters_by_aspect_long_codes(tmp_path: Path) -> None:
    """``load_boosters_by_aspect`` reads ``mfo.txt`` / ``bpo.txt`` / ``cco.txt``
    and normalises to ``F`` / ``P`` / ``C``."""
    for long_name, seed in (("mfo", 1), ("bpo", 2), ("cco", 3)):
        b = _train_aspect_booster(seed)
        b.save_model(str(tmp_path / f"{long_name}.txt"))
    out = load_boosters_by_aspect(tmp_path)
    assert set(out.keys()) == {"F", "P", "C"}


def test_load_boosters_by_aspect_partial(tmp_path: Path) -> None:
    """Two-aspect coverage is allowed; selective rerank handles the gap."""
    for letter, seed in (("F", 1), ("C", 2)):
        b = _train_aspect_booster(seed)
        b.save_model(str(tmp_path / f"{letter}.txt"))
    out = load_boosters_by_aspect(tmp_path)
    assert set(out.keys()) == {"F", "C"}


def test_load_boosters_by_aspect_missing_dir(tmp_path: Path) -> None:
    """Missing directory raises ``FileNotFoundError`` with the bad path."""
    with pytest.raises(FileNotFoundError, match="booster directory"):
        load_boosters_by_aspect(tmp_path / "does-not-exist")


def test_load_boosters_by_aspect_empty(tmp_path: Path) -> None:
    """Directory with no aspect-named artefacts raises ``ValueError``."""
    (tmp_path / "irrelevant.txt").write_text("not a booster")
    with pytest.raises(ValueError, match="no per-aspect booster artefacts"):
        load_boosters_by_aspect(tmp_path)


def test_load_boosters_by_aspect_conflict(tmp_path: Path) -> None:
    """``F.txt`` + ``mfo.txt`` both target aspect ``F``; raise rather than
    silently picking one."""
    for name, seed in (("F", 1), ("mfo", 2)):
        b = _train_aspect_booster(seed)
        b.save_model(str(tmp_path / f"{name}.txt"))
    with pytest.raises(ValueError, match="multiple artefacts target aspect"):
        load_boosters_by_aspect(tmp_path)


# ---------------------------------------------------------------------------
# Marker-gated regression against the LB.2 v226 champion
# ---------------------------------------------------------------------------


@pytest.mark.regression_v226
def test_regression_v226_selective_avg_fmax() -> None:
    """Reproduce the LB.2 champion's selective-average Fmax on v226.

    Champion: ``0.6215 ± 0.0014`` selective average cafaeval Fmax on
    ``bench-v1-K5-v226-lineage`` (multi-seed, leakage-fixed, v23
    config). The envelope is widened to ``±0.0050`` (3.3σ) so a single
    deterministic run absorbs the seed-to-seed jitter the multi-seed
    sweep measured.

    This test only runs end-to-end when ``PROTEA_LAB_DATA`` points at a
    directory holding:

    - ``bench-v1-K5-v226-lineage/eval.parquet`` (lab fixture),
    - ``bench-v1-K5-v226-lineage/queries.npz`` (query embeddings +
      accession list; not currently shipped by the lab),
    - ``bench-v1-K5-v226-lineage/references.npz`` (reference
      embeddings + accession list),
    - ``bench-v1-K5-v226-lineage/annotations.json`` (ref -> ann list),
    - ``boosters/{F,P,C}.txt`` (per-aspect champion artefacts),
    - ``champion.json`` with ``{"selective_avg_fmax": 0.6215}``.

    The lab does not currently ship the raw inputs the LAFA container
    needs (queries.npz / references.npz / annotations.json) as part of
    the dataset directory; until a follow-up slice publishes them this
    regression skips on every checkout. The skip is intentional and the
    PR body must surface it.
    """
    lab_data_root = os.environ.get("PROTEA_LAB_DATA")
    if not lab_data_root:
        pytest.skip(
            "regression requires PROTEA_LAB_DATA pointing at a directory with the "
            "bench-v1-K5-v226-lineage fixtures (queries.npz, references.npz, "
            "annotations.json) plus boosters/{F,P,C}.txt artefacts; lab does not "
            "publish those today, so the F2C.6 regression test is parked.",
        )
    root = Path(lab_data_root)
    required = [
        root / "bench-v1-K5-v226-lineage" / "queries.npz",
        root / "bench-v1-K5-v226-lineage" / "references.npz",
        root / "bench-v1-K5-v226-lineage" / "annotations.json",
        root / "boosters" / "F.txt",
        root / "boosters" / "P.txt",
        root / "boosters" / "C.txt",
        root / "champion.json",
    ]
    missing = [str(p) for p in required if not p.exists()]
    if missing:
        pytest.skip(f"missing lab fixtures under {root}: {missing}")

    queries = np.load(root / "bench-v1-K5-v226-lineage" / "queries.npz", allow_pickle=True)
    refs = np.load(root / "bench-v1-K5-v226-lineage" / "references.npz", allow_pickle=True)
    with (root / "bench-v1-K5-v226-lineage" / "annotations.json").open() as fh:
        anns_raw = json.load(fh)
    annotations = {k: list(v) for k, v in anns_raw["annotations"].items()}
    go_id_map = {int(k): str(v) for k, v in anns_raw["go_id_map"].items()}
    go_aspect_map = {int(k): str(v) for k, v in anns_raw["go_aspect_map"].items()}

    boosters = load_boosters_by_aspect(root / "boosters")

    predictions = predict(
        query_accessions=list(queries["accessions"]),
        query_embeddings=np.asarray(queries["embeddings"], dtype=np.float32),
        reference_accessions=list(refs["accessions"]),
        reference_embeddings=np.asarray(refs["embeddings"], dtype=np.float32),
        annotations=annotations,
        go_id_map=go_id_map,
        go_aspect_map=go_aspect_map,
        config=PredictConfig(
            k=5,
            aspect_separated=True,
            compute_v6_features=True,
            metric="cosine",
            pre_normalized=True,
        ),
        boosters_by_aspect=boosters,
    )
    assert predictions

    # Lazy import: ``cafaeval`` is an integration-only dep, not part of
    # the protea-method test extras.
    try:
        from cafaeval_protea import evaluate_predictions  # type: ignore[import-not-found]
    except ImportError:
        pytest.skip("cafaeval-protea not installed; cannot score regression")

    with (root / "champion.json").open() as fh:
        champion = json.load(fh)
    target = float(champion["selective_avg_fmax"])
    envelope = 0.0050

    scored = evaluate_predictions(predictions, eval_set_name="bench-v1-K5-v226-lineage")
    selective_avg = float(scored["selective_avg_fmax"])
    assert abs(selective_avg - target) <= envelope, (
        f"regression: selective_avg_fmax={selective_avg:.4f} drifted >"
        f"{envelope} from champion {target:.4f}"
    )
