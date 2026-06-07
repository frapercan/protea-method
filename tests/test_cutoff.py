"""Tests for the self-contained no-future-data (cutoff) guard."""

from __future__ import annotations

from datetime import date
from pathlib import Path

import pytest

from protea_method.cutoff import (
    BAND_CUTOFFS,
    CutoffViolationError,
    assert_obo_not_after_cutoff,
    obo_data_version,
    parse_release_date,
    resolve_cutoff_date,
)

FIXTURES = Path(__file__).parent / "fixtures"


def _write_obo(path: Path, data_version: str | None) -> Path:
    header = "format-version: 1.2\n"
    if data_version is not None:
        header += f"data-version: {data_version}\n"
    path.write_text(header + "\n[Term]\nid: GO:0008150\nname: biological_process\n")
    return path


class TestParseReleaseDate:
    def test_parses_releases_token(self) -> None:
        assert parse_release_date("releases/2025-07-22") == date(2025, 7, 22)

    def test_parses_bare_date(self) -> None:
        assert parse_release_date("2025-09-04") == date(2025, 9, 4)

    def test_none_for_undated(self) -> None:
        assert parse_release_date("go-basic") is None
        assert parse_release_date(None) is None


class TestResolveCutoffDate:
    def test_band_name(self) -> None:
        assert resolve_cutoff_date("v226") == BAND_CUTOFFS["v226"]
        assert resolve_cutoff_date("v227") == date(2025, 9, 4)

    def test_band_embedded_in_token(self) -> None:
        assert resolve_cutoff_date("bench-v1-K5-v227-lineage-prott5") == date(2025, 9, 4)

    def test_bare_date(self) -> None:
        assert resolve_cutoff_date("2025-09-04") == date(2025, 9, 4)

    def test_unknown_raises(self) -> None:
        with pytest.raises(ValueError):
            resolve_cutoff_date("not-a-band")


class TestOboDataVersion:
    def test_reads_header(self, tmp_path: Path) -> None:
        obo = _write_obo(tmp_path / "go.obo", "releases/2025-07-22")
        assert obo_data_version(obo) == "releases/2025-07-22"

    def test_none_when_absent(self, tmp_path: Path) -> None:
        obo = _write_obo(tmp_path / "go.obo", None)
        assert obo_data_version(obo) is None


class TestAssertOboNotAfterCutoff:
    def test_passes_on_congruent_release(self, tmp_path: Path) -> None:
        # v227 cutoff is 2025-09-04; the congruent OBO (2025-07-22) is before.
        obo = _write_obo(tmp_path / "go.obo", "releases/2025-07-22")
        assert_obo_not_after_cutoff(obo, "v227")

    def test_rejects_future_release(self, tmp_path: Path) -> None:
        # An OBO dated after the v227 cutoff is the no-future-data violation.
        obo = _write_obo(tmp_path / "go.obo", "releases/2026-01-23")
        with pytest.raises(CutoffViolationError):
            assert_obo_not_after_cutoff(obo, "v227")

    def test_noop_on_undated_obo(self, tmp_path: Path) -> None:
        obo = _write_obo(tmp_path / "go.obo", None)
        # Cannot order an undated release; guard is a no-op, not an error.
        assert_obo_not_after_cutoff(obo, "v227")

    def test_fixture_obo_is_future_for_v227(self) -> None:
        # tests/fixtures/tiny.obo is dated 2026-05-01, future of every band.
        with pytest.raises(CutoffViolationError):
            assert_obo_not_after_cutoff(FIXTURES / "tiny.obo", "v227")
