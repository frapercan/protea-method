"""Unit tests for ``protea_method.io.loaders``."""

from __future__ import annotations

import gzip
from pathlib import Path

import pytest

from protea_method.io.loaders import GAF_COLUMNS, read_fasta, read_gaf, read_obo

FIXTURES = Path(__file__).parent / "fixtures"


def test_read_fasta_basic_records() -> None:
    seqs = read_fasta(FIXTURES / "tiny.fasta")
    assert seqs == {
        "P00001": "MKVILGLLTLAGGKLSAN",
        "P00002": "QQQRRRDDDEEE",
        "P00003": "MGGGS",
    }


def test_read_fasta_handles_gzip(tmp_path: Path) -> None:
    src = (FIXTURES / "tiny.fasta").read_bytes()
    gz_path = tmp_path / "tiny.fasta.gz"
    with gzip.open(gz_path, "wb") as fh:
        fh.write(src)
    seqs = read_fasta(gz_path)
    assert "P00001" in seqs
    assert seqs["P00001"].startswith("MKVI")


def test_read_fasta_rejects_duplicate(tmp_path: Path) -> None:
    dup = tmp_path / "dup.fasta"
    dup.write_text(">A\nMM\n>A\nGG\n")
    with pytest.raises(ValueError, match="duplicate"):
        read_fasta(dup)


def test_read_fasta_rejects_empty_header(tmp_path: Path) -> None:
    empty = tmp_path / "empty.fasta"
    empty.write_text(">\nMM\n")
    with pytest.raises(ValueError, match="empty header"):
        read_fasta(empty)


def test_read_gaf_returns_minimum_columns() -> None:
    df = read_gaf(FIXTURES / "tiny.gaf")
    assert list(df.columns) == list(GAF_COLUMNS)
    assert len(df) == 4
    assert set(df["db_object_id"]) == {"P00001", "P00002", "P00003"}
    assert set(df["aspect"]) == {"P", "F", "C"}


def test_read_gaf_handles_gzip(tmp_path: Path) -> None:
    src = (FIXTURES / "tiny.gaf").read_bytes()
    gz_path = tmp_path / "tiny.gaf.gz"
    with gzip.open(gz_path, "wb") as fh:
        fh.write(src)
    df = read_gaf(gz_path)
    assert len(df) == 4


def test_read_gaf_skips_comment_and_short_rows(tmp_path: Path) -> None:
    bad = tmp_path / "bad.gaf"
    bad.write_text(
        "!comment\n"
        "tooshort\trow\n"  # < 9 fields
        # ensure 9 fields including go_id
        + "\t".join(["UniProtKB", "P9", "G", "", "GO:0000001", "PMID:0", "IDA", "", "P"])
        + "\n",
    )
    df = read_gaf(bad)
    assert len(df) == 1
    assert df.iloc[0]["db_object_id"] == "P9"


def test_read_obo_skips_obsolete_and_links_parents() -> None:
    terms = read_obo(FIXTURES / "tiny.obo")
    assert "GO:0099999" not in terms, "obsolete term must be skipped"
    assert terms["GO:0008150"]["name"] == "biological_process"
    assert terms["GO:0008150"]["namespace"] == "biological_process"
    assert terms["GO:0008150"]["parents"] == []
    assert terms["GO:0009987"]["parents"] == ["GO:0008150"]


def test_read_obo_part_of_relationship(tmp_path: Path) -> None:
    obo = tmp_path / "rel.obo"
    obo.write_text(
        "[Term]\n"
        "id: GO:1\n"
        "name: parent\n"
        "namespace: biological_process\n"
        "\n"
        "[Term]\n"
        "id: GO:2\n"
        "name: child\n"
        "namespace: biological_process\n"
        "relationship: part_of GO:1 ! parent\n"
        "\n",
    )
    terms = read_obo(obo)
    assert terms["GO:2"]["parents"] == ["GO:1"]


def test_read_obo_skips_non_term_stanzas(tmp_path: Path) -> None:
    obo = tmp_path / "typedef.obo"
    obo.write_text(
        "[Typedef]\n"
        "id: part_of\n"
        "name: part of\n"
        "\n"
        "[Term]\n"
        "id: GO:1\n"
        "name: only term\n"
        "namespace: biological_process\n"
        "\n",
    )
    terms = read_obo(obo)
    assert list(terms.keys()) == ["GO:1"]
