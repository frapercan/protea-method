"""Unit tests for ``protea_method.io.lafa_tsv``."""

from __future__ import annotations

import gzip
from pathlib import Path

import pytest

from protea_method.io.lafa_tsv import SCORE_PRECISION, write_lafa_tsv


def _read_lines(path: Path) -> list[str]:
    if path.suffix == ".gz":
        with gzip.open(path, "rt", encoding="utf-8") as fh:
            return [line.rstrip("\n") for line in fh]
    return [line.rstrip("\n") for line in path.read_text().splitlines()]


def test_writer_emits_three_columns_no_header(tmp_path: Path) -> None:
    out = tmp_path / "preds.tsv"
    predictions = [
        {"protein_accession": "P00001", "go_id": "GO:0008150", "reranker_score": 0.5},
        {"protein_accession": "P00001", "go_id": "GO:0003674", "reranker_score": 0.25},
    ]
    written = write_lafa_tsv(predictions, out)
    assert written == 2
    lines = _read_lines(out)
    assert len(lines) == 2
    for line in lines:
        cols = line.split("\t")
        assert len(cols) == 3
        # 3rd column is a float with 6 decimal places.
        assert "." in cols[2]
        assert len(cols[2].split(".")[1]) == SCORE_PRECISION


def test_writer_sorts_rows_by_query_then_go(tmp_path: Path) -> None:
    out = tmp_path / "preds.tsv"
    predictions = [
        {"protein_accession": "P00002", "go_id": "GO:0003674", "reranker_score": 0.9},
        {"protein_accession": "P00001", "go_id": "GO:0008150", "reranker_score": 0.5},
        {"protein_accession": "P00001", "go_id": "GO:0003674", "reranker_score": 0.7},
    ]
    write_lafa_tsv(predictions, out)
    lines = _read_lines(out)
    queries_then_terms = [tuple(line.split("\t")[:2]) for line in lines]
    assert queries_then_terms == [
        ("P00001", "GO:0003674"),
        ("P00001", "GO:0008150"),
        ("P00002", "GO:0003674"),
    ]


def test_writer_gzip_when_path_ends_in_gz(tmp_path: Path) -> None:
    out = tmp_path / "preds.tsv.gz"
    predictions = [
        {"protein_accession": "Q1", "go_id": "GO:0000001", "reranker_score": 0.1},
    ]
    written = write_lafa_tsv(predictions, out)
    assert written == 1
    # Read raw bytes and check the gzip magic.
    raw = out.read_bytes()
    assert raw.startswith(b"\x1f\x8b"), "expected gzip magic bytes"
    lines = _read_lines(out)
    assert lines == ["Q1\tGO:0000001\t0.100000"]


def test_writer_empty_predictions(tmp_path: Path) -> None:
    out = tmp_path / "preds.tsv"
    written = write_lafa_tsv([], out)
    assert written == 0
    assert out.exists()
    assert out.read_text() == ""


def test_writer_fallback_score_from_min_distance(tmp_path: Path) -> None:
    out = tmp_path / "preds.tsv"
    predictions = [
        {"protein_accession": "Q1", "go_id": "GO:0000001", "min_distance": 0.2},
    ]
    write_lafa_tsv(predictions, out)
    lines = _read_lines(out)
    # Score should be 1 - 0.2 = 0.8
    assert lines == ["Q1\tGO:0000001\t0.800000"]


def test_writer_creates_parent_directories(tmp_path: Path) -> None:
    out = tmp_path / "deep" / "nested" / "preds.tsv"
    predictions = [{"protein_accession": "Q1", "go_id": "GO:0000001", "score": 0.5}]
    write_lafa_tsv(predictions, out)
    assert out.exists()


def test_writer_drops_rows_missing_identity(tmp_path: Path) -> None:
    out = tmp_path / "preds.tsv"
    predictions = [
        {"protein_accession": "Q1", "go_id": "GO:0000001", "score": 0.5},
        {"protein_accession": "Q2", "score": 0.4},  # missing go_id
        {"go_id": "GO:0000002", "score": 0.3},  # missing query
    ]
    written = write_lafa_tsv(predictions, out)
    assert written == 1


def test_writer_accepts_alternate_column_names(tmp_path: Path) -> None:
    out = tmp_path / "preds.tsv"
    predictions = [
        {"Query_ID": "P1", "GO_Term": "GO:0000001", "Score": 0.42},
    ]
    write_lafa_tsv(predictions, out)
    assert _read_lines(out) == ["P1\tGO:0000001\t0.420000"]


def test_writer_score_precision_constant_drives_format(tmp_path: Path) -> None:
    """Guardrail: a future change to SCORE_PRECISION must update format strings."""
    assert isinstance(SCORE_PRECISION, int) and SCORE_PRECISION >= 1
    out = tmp_path / "preds.tsv"
    write_lafa_tsv(
        [{"protein_accession": "Q", "go_id": "GO:0000001", "score": 1.0 / 3.0}],
        out,
    )
    score_str = _read_lines(out)[0].split("\t")[2]
    assert len(score_str.split(".")[1]) == SCORE_PRECISION


def test_writer_rejects_tab_in_field(tmp_path: Path) -> None:
    out = tmp_path / "preds.tsv"
    bad = [{"protein_accession": "Q\t1", "go_id": "GO:0000001", "score": 0.1}]
    with pytest.raises(Exception):  # noqa: B017 - csv raises Error on QUOTE_NONE
        write_lafa_tsv(bad, out)
