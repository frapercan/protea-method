embed (in-container PLM embedder)
==================================

Optional sub-package for the LAFA self-contained container mode. When
``--query_embeds`` / ``--reference_embeds`` are omitted from the
entrypoint CLI, this package resolves a backend id (e.g.
``esm2_t36_3B`` for the LB.2 champion config) to a
``protea_backends`` plugin via ``entry_points``, computes mean-pooled
embeddings for every sequence in the FASTA, and caches the result to
disk keyed by ``(backend_id, fasta_sha256)``.

Heavy ML dependencies (``torch``, ``transformers``, ``protea-backends``)
are imported lazily. The default ``pip install protea-method`` stays
slim; users opt in with ``pip install "protea-method[esm]"`` to bring
in the ESM-2 runtime.

Install the matching extras group for real PLMs:

.. code-block:: bash

   pip install "protea-method[esm]"   # esm2_t36_3B + esm2_t33_650M
   pip install "protea-method[t5]"    # prost_t5_xl_uniref50

protea_method.embed
-------------------

.. automodule:: protea_method.embed
   :members:
   :undoc-members:
   :show-inheritance:
   :no-index:

protea_method.embed.backend
---------------------------

Backend resolver and ``embed_fasta`` orchestrator. The canonical
backend id map (``BACKEND_IDS``) exposes four entries: ``esm2_t36_3B``
(champion), ``esm2_t33_650M`` (lighter), ``prost_t5_xl_uniref50``
(cross-check), and ``mock_constant`` (tests only).

.. automodule:: protea_method.embed.backend
   :members:
   :undoc-members:
   :show-inheritance:
   :no-index:

protea_method.embed.cache
-------------------------

Disk-backed embedding cache keyed by ``(backend_id, fasta_sha256)``.
Avoids re-running the multi-hour PLM forward pass on repeated LAFA
container invocations against the same FASTA.

.. automodule:: protea_method.embed.cache
   :members:
   :undoc-members:
   :show-inheritance:
   :no-index:
