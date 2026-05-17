"""LAFA submission entrypoint for protea-method.

This script implements the standard LAFA container interface
documented at https://github.com/anphan0828/LAFA_container_guide and
delegates the inference work to
:func:`protea_method.pipeline.predict`. It is what the Docker image
runs at start-up:

.. code-block:: bash

   docker run \\
       -v ./data:/app/data:ro \\
       -v ./output:/app/output:rw \\
       protea-method-lafa:latest \\
       --query_file /app/data/test_sequences.fasta \\
       --train_sequences /app/data/train_sequences.fasta \\
       --annot_file /app/data/goa_uniprot_sprot.gaf.gz \\
       --graph /app/data/go-basic.obo \\
       --output_file /app/output/predictions.tsv.gz

Scope of this slice
-------------------

The LAFA-CONTAINER.1 slice ships the container plumbing only:

* CLI argument parsing matching the LAFA spec
* FASTA / GAF / OBO input validation (file existence checks)
* Output TSV writing through :func:`protea_method.io.write_lafa_tsv`
* Pre-computed embedding consumption via ``--query_embeds`` /
  ``--reference_embeds`` (parquet files bind-mounted in)

Wiring an in-container embedding backend (ESM / ProstT5) is the
LAFA-EMB.1 slice. Without ``--query_embeds`` and ``--reference_embeds``
the script exits with a clear error pointing at that follow-up.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any

import numpy as np

from protea_method.io import read_fasta, read_gaf, read_obo, write_lafa_tsv
from protea_method.pipeline import PredictConfig, predict

#: Mapping from GAF aspect letter (``F`` / ``P`` / ``C``) to the
#: canonical PROTEA aspect code. They already match; the indirection
#: makes the contract explicit so a future LAFA bump that changes the
#: letters does not silently break the reranker routing.
_GAF_ASPECT_TO_PROTEA: dict[str, str] = {"F": "F", "P": "P", "C": "C"}


def _build_parser() -> argparse.ArgumentParser:
    """Construct the CLI parser for the LAFA standard interface."""
    parser = argparse.ArgumentParser(
        prog="method_main.py",
        description=(
            "LAFA submission entrypoint for protea-method. "
            "Reads FASTA / GAF / OBO inputs, runs the KNN + reranker "
            "inference pipeline, writes a 3-column predictions TSV."
        ),
    )
    parser.add_argument(
        "--query_file", "-q", required=True, type=Path,
        help="FASTA of query (test) sequences (LAFA standard arg).",
    )
    parser.add_argument(
        "--train_sequences", required=True, type=Path,
        help="FASTA of training (reference) sequences.",
    )
    parser.add_argument(
        "--annot_file", "-a", required=True, type=Path,
        help="Annotation file in GAF or GAF.GZ format.",
    )
    parser.add_argument(
        "--graph", required=True, type=Path,
        help="GO ontology in OBO format (e.g. go-basic.obo).",
    )
    parser.add_argument(
        "--output_file", "-o", required=True, type=Path,
        help="Destination for the predictions TSV. Use a .gz suffix to gzip.",
    )
    parser.add_argument(
        "--query_embeds", type=Path, default=None,
        help=(
            "Optional parquet with pre-computed query embeddings. "
            "Required until LAFA-EMB.1 wires an in-container embedder."
        ),
    )
    parser.add_argument(
        "--reference_embeds", type=Path, default=None,
        help=(
            "Optional parquet with pre-computed reference embeddings. "
            "Required until LAFA-EMB.1 wires an in-container embedder."
        ),
    )
    parser.add_argument(
        "--top_k", type=int, default=5,
        help="KNN k value (default: 5; matches PROTEA's bench-v1-K5).",
    )
    parser.add_argument(
        "--metric", choices=["cosine", "l2"], default="cosine",
        help="KNN distance metric.",
    )
    parser.add_argument(
        "--backend", choices=["numpy", "faiss"], default="numpy",
        help="KNN backend implementation.",
    )
    parser.add_argument(
        "--num_threads", type=int, default=None,
        help=(
            "Optional NUM_THREADS hint. Currently informational only; "
            "numpy and faiss-cpu read OMP_NUM_THREADS from the environment."
        ),
    )
    return parser


def _validate_paths(args: argparse.Namespace) -> list[str]:
    """Return a list of human-readable errors for missing input files."""
    errors: list[str] = []
    required: tuple[tuple[str, Path], ...] = (
        ("--query_file", args.query_file),
        ("--train_sequences", args.train_sequences),
        ("--annot_file", args.annot_file),
        ("--graph", args.graph),
    )
    for flag, path in required:
        if not path.exists():
            errors.append(f"{flag}: file not found at {path}")
        elif path.is_dir():
            errors.append(f"{flag}: expected file, got directory {path}")
    if args.query_embeds is not None and not args.query_embeds.exists():
        errors.append(f"--query_embeds: file not found at {args.query_embeds}")
    if args.reference_embeds is not None and not args.reference_embeds.exists():
        errors.append(
            f"--reference_embeds: file not found at {args.reference_embeds}"
        )
    if (args.query_embeds is None) != (args.reference_embeds is None):
        errors.append(
            "--query_embeds and --reference_embeds must be provided together "
            "(or both omitted once LAFA-EMB.1 wires the embedding backend)."
        )
    return errors


def _annotations_from_gaf(
    gaf_path: Path,
    obo: dict[str, dict[str, Any]],
) -> tuple[
    dict[str, list[dict[str, Any]]],
    dict[int, str],
    dict[int, str],
]:
    """Convert a GAF file + OBO graph into the predict() input shape.

    Returns three structures aligned on a deterministic integer id
    space:

    * ``annotations``: ``ref_acc -> [{"go_term_id": int, "qualifier":
      "", "evidence_code": str}, ...]``
    * ``go_id_map``: ``go_term_id -> "GO:XXXXXXX"`` (stable per run)
    * ``go_aspect_map``: ``go_term_id -> "F" | "P" | "C"``
    """
    gaf = read_gaf(gaf_path)
    go_id_to_int: dict[str, int] = {}
    go_id_map: dict[int, str] = {}
    go_aspect_map: dict[int, str] = {}
    annotations: dict[str, list[dict[str, Any]]] = {}
    for record in gaf.itertuples(index=False):
        go_id = str(record.go_id)
        if go_id not in obo:
            # Term obsolete or absent from this OBO release; skip
            # silently so a slightly older OBO file does not poison
            # the inference run.
            continue
        gtid = go_id_to_int.get(go_id)
        if gtid is None:
            gtid = len(go_id_to_int)
            go_id_to_int[go_id] = gtid
            go_id_map[gtid] = go_id
            go_aspect_map[gtid] = _GAF_ASPECT_TO_PROTEA.get(
                str(record.aspect), str(record.aspect),
            )
        annotations.setdefault(str(record.db_object_id), []).append({
            "go_term_id": gtid,
            "qualifier": "",
            "evidence_code": str(record.evidence_code),
        })
    return annotations, go_id_map, go_aspect_map


def _load_embeddings(
    parquet_path: Path,
    expected_accessions: list[str],
) -> np.ndarray:
    """Load an embedding parquet aligned with ``expected_accessions``.

    The parquet must have an ``accession`` (or ``protein_accession``)
    column plus one of:

    * an ``embedding`` column carrying a list / numpy array per row, or
    * numeric columns ``e0 .. eN`` (one per dimension).

    Rows are reindexed to ``expected_accessions`` order. Missing
    accessions raise :class:`ValueError`.
    """
    import pandas as pd

    df = pd.read_parquet(parquet_path)
    acc_col = "accession" if "accession" in df.columns else "protein_accession"
    if acc_col not in df.columns:
        raise ValueError(
            f"{parquet_path}: missing 'accession' / 'protein_accession' column"
        )
    df = df.set_index(acc_col)
    missing = [a for a in expected_accessions if a not in df.index]
    if missing:
        sample = ", ".join(missing[:5])
        raise ValueError(
            f"{parquet_path}: {len(missing)} accessions missing from embeddings "
            f"(first: {sample})"
        )
    df = df.loc[expected_accessions]
    if "embedding" in df.columns:
        arr = np.stack([np.asarray(v, dtype=np.float32) for v in df["embedding"]])
        return arr
    numeric_cols = [c for c in df.columns if c.startswith("e") and c[1:].isdigit()]
    if numeric_cols:
        numeric_cols.sort(key=lambda c: int(c[1:]))
        return df[numeric_cols].to_numpy(dtype=np.float32)
    raise ValueError(
        f"{parquet_path}: no 'embedding' column and no e0..eN numeric columns"
    )


def _run(args: argparse.Namespace) -> int:
    """Top-level orchestration. Returns the process exit code."""
    sys.stderr.write(f"[lafa] loading FASTA: {args.query_file}\n")
    queries = read_fasta(args.query_file)
    sys.stderr.write(f"[lafa] loading FASTA: {args.train_sequences}\n")
    references = read_fasta(args.train_sequences)
    sys.stderr.write(f"[lafa] loading OBO: {args.graph}\n")
    obo = read_obo(args.graph)
    sys.stderr.write(f"[lafa] loading GAF: {args.annot_file}\n")
    annotations, go_id_map, go_aspect_map = _annotations_from_gaf(
        args.annot_file, obo,
    )

    if args.query_embeds is None or args.reference_embeds is None:
        sys.stderr.write(
            "[lafa] error: --query_embeds and --reference_embeds are "
            "required in this slice (LAFA-CONTAINER.1). The in-container "
            "embedding backend is deferred to LAFA-EMB.1.\n"
        )
        return 2

    query_accessions = list(queries.keys())
    reference_accessions = [a for a in references.keys() if a in annotations]
    if not reference_accessions:
        sys.stderr.write(
            "[lafa] error: zero reference proteins carry annotations in the "
            "supplied GAF; check that --annot_file matches --train_sequences.\n"
        )
        return 3
    sys.stderr.write(
        f"[lafa] loading embeddings (queries={len(query_accessions)}, "
        f"references={len(reference_accessions)})\n"
    )
    query_embeds = _load_embeddings(args.query_embeds, query_accessions)
    reference_embeds = _load_embeddings(args.reference_embeds, reference_accessions)

    cfg = PredictConfig(
        k=args.top_k,
        metric=args.metric,
        backend=args.backend,
        compute_v6_features=False,
        compute_taxonomy=False,
    )
    sys.stderr.write(
        f"[lafa] predict(k={cfg.k}, metric={cfg.metric}, backend={cfg.backend})\n"
    )
    predictions = predict(
        query_accessions=query_accessions,
        query_embeddings=query_embeds,
        reference_accessions=reference_accessions,
        reference_embeddings=reference_embeds,
        annotations=annotations,
        go_id_map=go_id_map,
        go_aspect_map=go_aspect_map,
        config=cfg,
    )
    sys.stderr.write(f"[lafa] writing {len(predictions)} rows to {args.output_file}\n")
    written = write_lafa_tsv(predictions, args.output_file)
    sys.stderr.write(f"[lafa] done. {written} rows written.\n")
    return 0


def main(argv: list[str] | None = None) -> int:
    """Entrypoint. Returns process exit code."""
    parser = _build_parser()
    args = parser.parse_args(argv)
    errors = _validate_paths(args)
    if errors:
        for err in errors:
            sys.stderr.write(f"[lafa] error: {err}\n")
        return 2
    try:
        return _run(args)
    except (ValueError, FileNotFoundError, OSError) as exc:
        sys.stderr.write(f"[lafa] error: {exc}\n")
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
