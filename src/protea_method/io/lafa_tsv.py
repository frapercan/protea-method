"""LAFA-format 3-column TSV writer.

The LAFA container guide (https://github.com/anphan0828/LAFA_container_guide)
expects predictions in a 3-column tab-separated file with columns
``Query_ID``, ``GO_Term`` and ``Score``, no header row, and all three
GO aspects (MFO / BPO / CCO) interleaved in a single file. The file
is optionally gzipped (decided by the output path suffix).

Scores are floats; the spec does not bound the range, but higher must
mean more confident. Reranker scores are emitted as ``reranker_score``
when a booster is loaded; otherwise the writer falls back to
``1 - min_distance`` (cosine similarity) so the column is always
populated. The fallback rule is documented at module scope so future
slices (per-aspect selective rerank) can adjust it in one place.

Rows are sorted by ``(Query_ID, GO_Term)`` for determinism. This is
not required by the spec but it makes diffing two submission files
trivial and aligns the writer with PROTEA's lab-side dump invariant.
"""

from __future__ import annotations

import csv
import gzip
import io
from collections.abc import Iterable, Mapping
from pathlib import Path
from typing import Any

#: Score precision in the LAFA TSV. Six decimals match the precision
#: used by the reranker booster output and keep the file size in line
#: with the LAFA baseline submissions.
SCORE_PRECISION: int = 6

#: Keys searched in order to extract the GO accession from a
#: prediction row. ``go_id`` is the canonical column emitted by
#: :func:`protea_method.pipeline.predict`; the other keys are
#: accepted for resilience against alternative upstream shapes.
_GO_ID_KEYS: tuple[str, ...] = ("go_id", "GO_Term", "go_term")

#: Keys searched in order to extract the query accession.
_QUERY_KEYS: tuple[str, ...] = ("protein_accession", "Query_ID", "query_id")

#: Keys searched in order to extract the confidence score. The order
#: is significant: a reranker score wins when available; otherwise the
#: writer falls back to ``1 - min_distance``.
_SCORE_KEYS: tuple[str, ...] = ("reranker_score", "score", "Score")


def _first_present(row: Mapping[str, Any], keys: Iterable[str]) -> Any | None:
    """Return the first value in ``row`` whose key is in ``keys``."""
    for key in keys:
        if key in row and row[key] is not None:
            return row[key]
    return None


def _extract_score(row: Mapping[str, Any]) -> float:
    """Pick a confidence score for one prediction row.

    Resolution order:

    1. ``reranker_score`` / ``score`` / ``Score`` if present.
    2. ``1 - min_distance`` as a cosine-similarity fallback when only
       the KNN base output is available.
    3. ``0.0`` for rows that lack both signals (defensive; the writer
       must always emit a numeric Score column).
    """
    explicit = _first_present(row, _SCORE_KEYS)
    if explicit is not None:
        return float(explicit)
    min_distance = row.get("min_distance")
    if min_distance is not None:
        return float(1.0 - float(min_distance))
    return 0.0


def _row_to_tuple(row: Mapping[str, Any]) -> tuple[str, str, float] | None:
    """Project a prediction dict onto the LAFA 3-column tuple.

    Returns ``None`` when the row is missing one of the two identity
    fields (query accession or GO accession). Score is allowed to
    fall back to the cosine-similarity rule documented in
    :func:`_extract_score`.
    """
    query = _first_present(row, _QUERY_KEYS)
    go_id = _first_present(row, _GO_ID_KEYS)
    if query is None or go_id is None:
        return None
    score = _extract_score(row)
    return str(query), str(go_id), float(score)


def _open_writer(output_path: Path) -> Any:
    """Open ``output_path`` for writing, gzipped if the suffix is ``.gz``."""
    if output_path.suffix == ".gz":
        return gzip.open(output_path, "wt", encoding="utf-8", newline="")
    return output_path.open("w", encoding="utf-8", newline="")


def write_lafa_tsv(
    predictions: Iterable[Mapping[str, Any]],
    output_path: Path | str,
) -> int:
    """Write predictions to a LAFA-format 3-column TSV.

    Parameters
    ----------
    predictions:
        Iterable of prediction dicts as emitted by
        :func:`protea_method.pipeline.predict`. Each dict must carry a
        query accession (``protein_accession`` / ``Query_ID``), a GO
        accession (``go_id`` / ``GO_Term``) and a score-bearing field
        (``reranker_score`` / ``score`` or ``min_distance`` for the
        cosine-similarity fallback). Rows that lack the two identity
        fields are silently dropped (the caller can validate upstream
        if a strict mode is required).
    output_path:
        Destination file path. If it ends in ``.gz`` the writer
        gzip-compresses the output transparently. Parent directories
        are created when missing.

    Returns
    -------
    int
        Number of rows written.

    Notes
    -----
    Rows are sorted by ``(Query_ID, GO_Term)`` for reproducibility.
    All three GO aspects are interleaved in the single output file;
    per-aspect splitting is done by LAFA on its side.
    """
    out_path = Path(output_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    tuples: list[tuple[str, str, float]] = []
    for row in predictions:
        projected = _row_to_tuple(row)
        if projected is not None:
            tuples.append(projected)
    tuples.sort(key=lambda t: (t[0], t[1]))

    fh = _open_writer(out_path)
    try:
        # ``csv.writer`` so any embedded tabs in field values are
        # rejected via the dialect; in practice GO ids and accessions
        # never contain tabs but the guard is cheap.
        writer = csv.writer(
            _ensure_text_stream(fh),
            delimiter="\t",
            lineterminator="\n",
            quoting=csv.QUOTE_NONE,
            escapechar=None,
        )
        for query, go_id, score in tuples:
            writer.writerow([query, go_id, f"{score:.{SCORE_PRECISION}f}"])
    finally:
        fh.close()
    return len(tuples)


def _ensure_text_stream(handle: Any) -> io.TextIOBase:
    """Return a text-mode stream for the csv writer.

    Both :func:`gzip.open` (with mode ``"wt"``) and
    :meth:`pathlib.Path.open` already yield text streams; this helper
    exists to make the type narrowing explicit for mypy --strict.
    """
    return handle  # type: ignore[no-any-return]


__all__ = ["SCORE_PRECISION", "write_lafa_tsv"]
