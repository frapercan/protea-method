Container usage
===============

``method_main.py`` implements the standard LAFA container interface. This
page covers building the image and the two run modes.

The LAFA container contract
---------------------------

The container follows the
`LAFA container guide <https://github.com/anphan0828/LAFA_container_guide>`_:

* inputs are bind-mounted read-only at ``/app/data``,
* outputs are written to ``/app/output`` (read-write),
* the entrypoint takes generic flags (``--query_file``,
  ``--train_sequences``, ``--annot_file``, ``--graph``,
  ``--output_file``),
* predictions are a 3-column TSV (``Query_ID``, ``GO_Term``, ``Score``),
  no header, all aspects interleaved, optionally gzipped.

Build the image
---------------

.. code-block:: bash

   docker build -t protea-method-lafa:latest .

Bind-mount mode (pre-computed embeddings)
-----------------------------------------

Pass ``--query_embeds`` / ``--reference_embeds`` pointing at parquet
files with pre-computed embeddings. The slim image (no torch, no
HuggingFace cache) is sufficient:

.. code-block:: bash

   docker run \
       -v ./data:/app/data:ro \
       -v ./output:/app/output:rw \
       protea-method-lafa:latest \
       --query_file /app/data/test_sequences.fasta \
       --train_sequences /app/data/train_sequences.fasta \
       --annot_file /app/data/goa_uniprot_sprot.gaf.gz \
       --graph /app/data/go-basic.obo \
       --query_embeds /app/data/query_embeds.parquet \
       --reference_embeds /app/data/reference_embeds.parquet \
       --output_file /app/output/predictions.tsv.gz

The repository ships ``docker/example_run.sh`` as a ready-to-edit
wrapper.

Self-contained mode (in-container embedder)
-------------------------------------------

Omit both embed flags. The container computes embeddings in-process via
a ``protea-backends`` plugin (default ``esm2_t36_3B``), caching them under
``$LAFA_EMBED_CACHE`` (``/app/output/.embed_cache`` in the image) keyed
by ``(backend_id, fasta_sha256)``:

.. code-block:: bash

   docker run \
       -v ./data:/app/data:ro \
       -v ./output:/app/output:rw \
       -v hf-cache:/app/.hf-cache \
       protea-method-lafa:latest \
       --query_file /app/data/test_sequences.fasta \
       --train_sequences /app/data/train_sequences.fasta \
       --annot_file /app/data/goa_uniprot_sprot.gaf.gz \
       --graph /app/data/go-basic.obo \
       --backend_id esm2_t36_3B \
       --output_file /app/output/predictions.tsv.gz

Mount a host directory to ``/app/.hf-cache`` to avoid re-downloading the
weights on every fresh container. ``docker/example_run_selfcontained.sh``
is the matching wrapper.

Accepted backend ids
--------------------

.. list-table::
   :header-rows: 1

   * - ID
     - Model
     - Size
   * - ``esm2_t36_3B`` (default)
     - ``facebook/esm2_t36_3B_UR50D``
     - ~12 GB
   * - ``esm2_t33_650M``
     - ``facebook/esm2_t33_650M_UR50D``
     - ~2.5 GB
   * - ``prost_t5_xl_uniref50``
     - ``Rostlab/ProstT5``
     - ~5.5 GB
   * - ``mock_constant``
     - deterministic constant vector
     - tests only

Temporal cutoff
---------------

Pass ``--cutoff`` (a band name or a ``YYYY-MM-DD`` date) to enable the
no-future-data guard described in the
:ref:`own-reference cutoff section <own-reference>` of the overview. The
container then refuses a ``--graph`` ontology dated after the cutoff
before any heavy work runs.

Releasing to a registry
-----------------------

Three operator documents in ``docker/`` cover the full publish workflow:

* ``docker/RELEASE_RUNBOOK.md``: the numbered build, smoke-test, push,
  submit, and tag checklist. Start here.
* ``docker/DOCKERHUB_README.md``: the long-form registry description.
* ``docker/FUNCTIONBENCH_METHODCARD.md``: the one-page method card with
  validation numbers.

The registry push is intentionally manual; no CI job performs it.
