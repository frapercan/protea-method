"""Self-contained no-future-data (cutoff) guard for the LAFA entrypoint.

The single-``cutoff`` inference path (F-EVAL-PROTOCOL.c) threads one knob
through the whole method so corpus, embeddings, candidate KNN, features and
IA all read ONLY data dated on/before the cutoff. PROTEA enforces this on
the export side via ``protea.core.band_registry``; protea-method enforces
the inference side here.

protea-method has a hard rule: no ``protea-core`` dependency (slim,
infra-free LAFA container). So the few constants and the tiny date-ordering
logic the guard needs are mirrored locally rather than imported. Both sides
agree by construction: an OBO release token is self-dating
(``releases/YYYY-MM-DD``) and each band's t0 is a fixed calendar date.

The guard reads the GO ontology's ``data-version:`` header (the release the
container will propagate against) and refuses a release dated after the
declared cutoff, so a frozen container cannot silently score against a
future ontology.
"""

from __future__ import annotations

import re
from datetime import date
from pathlib import Path

__all__ = [
    "BAND_CUTOFFS",
    "CutoffViolationError",
    "assert_obo_not_after_cutoff",
    "obo_data_version",
    "parse_release_date",
    "resolve_cutoff_date",
]


class CutoffViolationError(ValueError):
    """An artifact references a release dated after the declared cutoff."""


#: Band -> training-cutoff (t0) date. Mirrors the ``t0_cutoff`` values pinned
#: in ``protea.core.band_registry.BANDS``; kept here so the slim container has
#: no protea-core dependency. Add a row when PROTEA registers a new band.
BAND_CUTOFFS: dict[str, date] = {
    "v226": date(2025, 5, 3),
    "v227": date(2025, 9, 4),
}


_RELEASE_DATE_RE = re.compile(r"(\d{4})-(\d{2})-(\d{2})")


def parse_release_date(release_ref: str | None) -> date | None:
    """Parse the ``YYYY-MM-DD`` embedded in a release reference.

    Accepts ``releases/2025-07-22``, a bare ``2025-07-22``, or a URL ending
    in that token. Returns ``None`` when no date is present (an undated
    ``go-basic`` snapshot, say), in which case ordering is impossible and the
    guard is a no-op.
    """
    if not release_ref:
        return None
    match = _RELEASE_DATE_RE.search(release_ref)
    if match is None:
        return None
    try:
        return date(int(match.group(1)), int(match.group(2)), int(match.group(3)))
    except ValueError:
        return None


def obo_data_version(obo_path: Path | str) -> str | None:
    """Return the ``data-version:`` header from an OBO file, or ``None``.

    Reads only the header (lines before the first ``[Term]`` stanza), so it
    is cheap even on a full ``go-basic.obo``. Handles gzip transparently via
    the suffix.
    """
    path = Path(obo_path)
    opener = _open_text(path)
    with opener as handle:
        for raw in handle:
            line = raw.strip()
            if line.startswith("[") and line.endswith("]"):
                break
            if line.startswith("data-version:"):
                return line.split(":", 1)[1].strip() or None
    return None


def resolve_cutoff_date(cutoff: str) -> date:
    """Resolve a cutoff knob to its t0 date.

    Accepts a registered band name (``v226`` / ``v227`` / ...), a token that
    embeds one (``bench-v1-K5-v227-lineage-prott5``), or a bare
    ``YYYY-MM-DD`` date. Raises :class:`ValueError` for an unresolvable knob
    so a typo never silently disables the guard.
    """
    if cutoff in BAND_CUTOFFS:
        return BAND_CUTOFFS[cutoff]
    for token in re.findall(r"v\d+", cutoff or ""):
        if token in BAND_CUTOFFS:
            return BAND_CUTOFFS[token]
    parsed = parse_release_date(cutoff)
    if parsed is not None:
        return parsed
    known = ", ".join(sorted(BAND_CUTOFFS))
    raise ValueError(
        f"unknown cutoff {cutoff!r}; pass a registered band ({known}) or a "
        "YYYY-MM-DD date. New bands are added to protea_method.cutoff.BAND_CUTOFFS."
    )


def assert_obo_not_after_cutoff(obo_path: Path | str, cutoff: str) -> None:
    """Refuse a GO ontology dated after the declared cutoff.

    Reads the OBO ``data-version:`` header and orders it against the cutoff
    t0. A no-op when the header carries no parseable date (ordering is
    impossible). Raises :class:`CutoffViolationError` on a future release.
    """
    cutoff_date = resolve_cutoff_date(cutoff)
    data_version = obo_data_version(obo_path)
    released = parse_release_date(data_version)
    if released is None:
        return
    if released > cutoff_date:
        raise CutoffViolationError(
            f"no-future-data guard: --cutoff {cutoff!r} (t0 "
            f"{cutoff_date.isoformat()}) but --graph data-version "
            f"{data_version!r} is dated {released.isoformat()}, AFTER the "
            "cutoff. A frozen container must not propagate against a future "
            "ontology. Supply the GO release current at the cutoff."
        )


def _open_text(path: Path):  # type: ignore[no-untyped-def]
    """Open a (possibly gzipped) text file for header reading."""
    if path.suffix == ".gz":
        import gzip

        return gzip.open(path, "rt", encoding="utf-8", errors="replace")
    return open(path, encoding="utf-8", errors="replace")
