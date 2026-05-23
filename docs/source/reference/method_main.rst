method_main (LAFA entrypoint)
==============================

``method_main.py`` is the LAFA submission entrypoint. It implements the
standard LAFA container interface and delegates all inference work to
:func:`protea_method.pipeline.predict`. It is the script the Docker image
runs at start-up.

Two run modes are supported:

**Bind-mount mode** (pre-computed embeddings)
  Pass ``--query_embeds`` / ``--reference_embeds`` pointing at parquet
  files with pre-computed embeddings. The slim image (no torch, no
  HuggingFace cache) is sufficient.

**Self-contained mode** (in-container PLM embedder)
  Omit the two embed flags. The entrypoint reads each FASTA, runs the
  configured backend (``--backend_id esm2_t36_3B`` by default), caches
  the result under ``LAFA_EMBED_CACHE``, and proceeds to predict.
  Requires the ``[esm]`` extras group at install time.

.. automodule:: method_main
   :members:
   :undoc-members:
   :show-inheritance:
