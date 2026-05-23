io (LAFA file adapters)
=======================

File-format adapters that bridge LAFA's input contract (FASTA, GAF,
OBO) and output contract (3-column TSV) with the in-memory shapes that
:func:`protea_method.pipeline.predict` consumes and emits.

Nothing in this package depends on the embedding backend; embeddings are
bind-mounted as parquet files.

protea_method.io
----------------

.. automodule:: protea_method.io
   :members:
   :undoc-members:
   :show-inheritance:
   :no-index:

protea_method.io.loaders
------------------------

FASTA, GAF, and OBO readers for the container entrypoint. All readers
accept either plain text or gzipped variants (auto-detected by ``.gz``
suffix).

.. automodule:: protea_method.io.loaders
   :members:
   :undoc-members:
   :show-inheritance:
   :no-index:

protea_method.io.lafa_tsv
-------------------------

3-column LAFA TSV writer (``Query_ID``, ``GO_Term``, ``Score``). Accepts
gzip compression when the output path ends in ``.gz``.

.. automodule:: protea_method.io.lafa_tsv
   :members:
   :undoc-members:
   :show-inheritance:
   :no-index:
