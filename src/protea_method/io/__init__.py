"""Input / output helpers for the LAFA submission container.

The :mod:`protea_method.io` package contains the file-format adapters
that the standalone container entrypoint needs to bridge LAFA's input
contract (FASTA, GAF, OBO) and output contract (3-column TSV) with the
in-memory shapes that :func:`protea_method.pipeline.predict` consumes
and emits.

Nothing in this package depends on the embedding backend; embeddings
are bind-mounted as parquet files and loaded by the caller via the
``--query_embeds`` flag (see ``method_main.py``).
"""

from protea_method.io.lafa_tsv import write_lafa_tsv
from protea_method.io.loaders import read_fasta, read_gaf, read_obo

__all__ = [
    "read_fasta",
    "read_gaf",
    "read_obo",
    "write_lafa_tsv",
]
