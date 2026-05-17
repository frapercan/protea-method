"""Input loaders for the LAFA submission container.

Three readers, one per file format LAFA's standard interface requires:

* :func:`read_fasta` parses FASTA (gzipped or not) into an
  accession-to-sequence dict.
* :func:`read_gaf` parses GO Annotation Format 2.x files into a
  pandas DataFrame with the minimum columns the reranker needs.
* :func:`read_obo` parses ``go-basic.obo`` into an id -> metadata
  mapping with name and parent links. A full OBO library is not
  pulled in; a 60-line streaming parser is sufficient for the
  inference container (the lab side already uses obonet via the
  reranker training repo).

All three readers accept either a plain text file or a gzipped
variant (auto-detected by the ``.gz`` suffix). The container ships
LAFA-supplied files directly; no preprocessing step required.
"""

from __future__ import annotations

import gzip
import io
from collections.abc import Iterator
from pathlib import Path
from typing import IO, Any

import pandas as pd

#: Minimum columns kept from a GAF file. The reranker only consumes
#: these four; everything else (dates, sources, evidence sources) is
#: discarded to keep memory low on the 7,401-protein LAFA testbed.
GAF_COLUMNS: tuple[str, ...] = (
    "db_object_id",
    "go_id",
    "aspect",
    "evidence_code",
)

#: GAF 2.x column index -> meaning. Reference:
#: https://geneontology.org/docs/go-annotation-file-gaf-format-2.2/
_GAF_FIELD_INDEX: dict[str, int] = {
    "db_object_id": 1,
    "go_id": 4,
    "evidence_code": 6,
    "aspect": 8,
}


def _open_text(path: Path) -> IO[str]:
    """Return a text stream, transparently decompressing ``.gz`` files."""
    if path.suffix == ".gz":
        return gzip.open(path, "rt", encoding="utf-8")
    return path.open("r", encoding="utf-8")


def read_fasta(path: Path | str) -> dict[str, str]:
    """Parse a FASTA file into an accession-to-sequence dict.

    Accession is taken from the first whitespace-separated token of
    the header line, with the leading ``>`` stripped. Sequence
    whitespace (newlines, spaces) is removed. Lower-case letters are
    preserved as-is; the embedding backends accept both cases.

    Parameters
    ----------
    path:
        Path to a ``.fasta`` / ``.fa`` / ``.faa`` file (optionally
        gzipped).

    Returns
    -------
    dict[str, str]
        Mapping from accession to sequence. Duplicate accessions are
        rejected with a :class:`ValueError`; the LAFA inputs are
        deduplicated upstream, so a collision indicates a corrupt
        file.

    Notes
    -----
    The parser streams the file (does not materialise it twice) so a
    full SwissProt FASTA (~550k entries, ~280 MB) fits in RAM only
    once.
    """
    fasta_path = Path(path)
    sequences: dict[str, str] = {}
    current_accession: str | None = None
    current_chunks: list[str] = []

    def _flush() -> None:
        if current_accession is None:
            return
        if current_accession in sequences:
            raise ValueError(
                f"duplicate FASTA accession {current_accession!r} in {fasta_path}"
            )
        sequences[current_accession] = "".join(current_chunks)

    with _open_text(fasta_path) as handle:
        for raw in handle:
            line = raw.rstrip("\r\n")
            if not line:
                continue
            if line.startswith(">"):
                _flush()
                current_accession = line[1:].split()[0] if len(line) > 1 else ""
                current_chunks = []
                continue
            current_chunks.append(line.strip())
        _flush()

    if "" in sequences:
        raise ValueError(f"FASTA file {fasta_path} has an empty header")
    return sequences


def read_gaf(path: Path | str) -> pd.DataFrame:
    """Parse a GO Annotation Format file into a pandas DataFrame.

    Comment lines (``!`` prefix) are skipped. The returned DataFrame
    has the four columns listed in :data:`GAF_COLUMNS` (object id, GO
    id, aspect letter, evidence code). Rows with empty GO ids are
    dropped.

    Parameters
    ----------
    path:
        Path to a ``.gaf`` or ``.gaf.gz`` file. Both GAF 2.1 and 2.2
        layouts work because only the first 9 columns are read and
        those indices are stable between versions.

    Returns
    -------
    pandas.DataFrame
        Columns: ``db_object_id``, ``go_id``, ``aspect``,
        ``evidence_code``. The ``aspect`` column carries the canonical
        single letter (``F`` / ``P`` / ``C``).
    """
    gaf_path = Path(path)
    rows: list[dict[str, str]] = []
    with _open_text(gaf_path) as handle:
        for raw in handle:
            if not raw or raw.startswith("!"):
                continue
            fields = raw.rstrip("\r\n").split("\t")
            if len(fields) < 9:
                continue
            go_id = fields[_GAF_FIELD_INDEX["go_id"]]
            if not go_id:
                continue
            rows.append({
                "db_object_id": fields[_GAF_FIELD_INDEX["db_object_id"]],
                "go_id": go_id,
                "aspect": fields[_GAF_FIELD_INDEX["aspect"]],
                "evidence_code": fields[_GAF_FIELD_INDEX["evidence_code"]],
            })
    return pd.DataFrame(rows, columns=list(GAF_COLUMNS))


def _stanzas(handle: IO[str]) -> Iterator[dict[str, list[str]]]:
    """Yield OBO stanzas as dicts of field-name -> list of values.

    Only ``[Term]`` stanzas are emitted. Header lines before the first
    stanza are skipped. Inside a stanza, each ``key: value`` pair is
    appended to a list (most keys are repeatable, eg ``is_a`` and
    ``relationship``).
    """
    current: dict[str, list[str]] | None = None
    stanza_kind: str | None = None
    for raw in handle:
        line = raw.rstrip("\r\n")
        if not line:
            if current is not None and stanza_kind == "Term":
                yield current
            current = None
            stanza_kind = None
            continue
        if line.startswith("[") and line.endswith("]"):
            if current is not None and stanza_kind == "Term":
                yield current
            stanza_kind = line[1:-1]
            current = {}
            continue
        if current is None or ":" not in line:
            continue
        key, _, value = line.partition(":")
        current.setdefault(key.strip(), []).append(value.strip())
    if current is not None and stanza_kind == "Term":
        yield current


def _parents_from_stanza(stanza: dict[str, list[str]]) -> list[str]:
    """Extract direct parents (``is_a`` + ``part_of`` relationships)."""
    parents: list[str] = []
    for raw in stanza.get("is_a", []):
        token = raw.split("!", 1)[0].strip()
        if token:
            parents.append(token)
    for raw in stanza.get("relationship", []):
        # ``relationship: part_of GO:0008150 ! biological_process``
        head = raw.split("!", 1)[0].strip()
        parts = head.split()
        if len(parts) >= 2 and parts[0] == "part_of":
            parents.append(parts[1])
    return parents


def read_obo(path: Path | str) -> dict[str, dict[str, Any]]:
    """Parse an OBO file into a minimal id -> metadata mapping.

    Only ``[Term]`` stanzas are returned. Obsolete terms (``is_obsolete:
    true``) are skipped to match the eval-time behaviour of LAFA's
    propagator. The returned dict carries:

    * ``name``: human-readable label (or empty string when missing).
    * ``namespace``: ``biological_process`` / ``molecular_function`` /
      ``cellular_component`` (or empty string).
    * ``parents``: list of direct ``is_a`` and ``part_of`` GO ids.

    Parameters
    ----------
    path:
        Path to ``go-basic.obo`` (or any OBO 1.2 file).

    Returns
    -------
    dict[str, dict[str, Any]]
        Mapping ``GO:XXXXXXX -> {"name": ..., "namespace": ..., "parents": [...]}``.
    """
    obo_path = Path(path)
    terms: dict[str, dict[str, Any]] = {}
    with _open_text(obo_path) as handle:
        for stanza in _stanzas(handle):
            if "true" in stanza.get("is_obsolete", []):
                continue
            ids = stanza.get("id", [])
            if not ids:
                continue
            term_id = ids[0]
            name = stanza.get("name", [""])[0]
            namespace = stanza.get("namespace", [""])[0]
            terms[term_id] = {
                "name": name,
                "namespace": namespace,
                "parents": _parents_from_stanza(stanza),
            }
    return terms


def _ensure_text_stream(handle: Any) -> io.TextIOBase:
    """Type-narrowing helper for the gzip / pathlib union."""
    return handle  # type: ignore[no-any-return]


__all__ = [
    "GAF_COLUMNS",
    "read_fasta",
    "read_gaf",
    "read_obo",
]
