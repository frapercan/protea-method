Overview and concepts
======================

This page is the conceptual tour. It introduces the key abstractions,
walks the end-to-end inference flow as a story, and explains the
own-reference, temporal-cutoff design that keeps a frozen submission
honest. For runnable code see the :doc:`quickstart`; for the container
surface see :doc:`container_usage`.

Key abstractions
----------------

The library is built from a handful of plain objects. None of them owns
state beyond a single call; everything flows through arguments and return
values.

The reference pool
    A set of reference proteins, each with an embedding vector and a list
    of GO annotations. It is PROTEA's own curated bank (see
    :ref:`own-reference` below), not whatever training set a benchmark
    hands you. KNN votes are cast by the references nearest to each query.

``PredictConfig``
    A frozen dataclass of inference knobs: ``k`` (neighbours per query),
    ``metric`` (``cosine`` or ``l2``), ``backend`` (``numpy`` or
    ``faiss``), ``aspect_separated`` (one index, or three), and the
    feature-computation toggles. Its defaults match PROTEA's production
    KNN behaviour, so the LAFA-side caller reproduces the lab numbers
    without re-specifying every field.

The prediction row
    The unit of output is a dict, one per ``(query, GO term)`` candidate.
    It carries identity (``protein_accession``, ``go_id``, ``aspect``,
    the donor reference), the KNN aggregates that feed the reranker
    (``vote_count``, ``k_position``, ``min_distance``,
    ``neighbor_vote_fraction``, and so on), and, when a booster runs, a
    ``reranker_score``. The :func:`~protea_method.io.write_lafa_tsv`
    writer projects these dicts onto the 3-column submission format.

The booster
    An optional, already-trained LightGBM model. A single ``booster``
    scores every row, or three ``boosters_by_aspect`` score each GO
    aspect with its own model (selective rerank). Boosters are produced
    by the lab; this library only *applies* them.

The end-to-end inference flow
-----------------------------

The orchestration lives in :func:`protea_method.pipeline.predict` (the
library core) and ``method_main.py`` (the LAFA CLI that loads files and
calls it). Read top to bottom, the path is::

   query FASTA ─┐
                ├─> embeddings ─> KNN search ─> vote tally ─> features ─> reranker ─> 3-col TSV
   reference ───┘

Stage 1: load inputs
~~~~~~~~~~~~~~~~~~~~~

``method_main.py`` reads the four LAFA-standard files:

* ``--query_file``: FASTA of test sequences, parsed by
  :func:`protea_method.io.read_fasta`.
* ``--train_sequences``: FASTA of reference sequences.
* ``--annot_file``: GAF (or GAF.GZ) annotations, parsed by
  :func:`protea_method.io.read_gaf`.
* ``--graph``: ``go-basic.obo``, parsed by
  :func:`protea_method.io.read_obo`.

The GAF and OBO are folded into the ``predict`` input shape: an
``annotations`` map (reference accession to GO term records) plus a
``go_id_map`` and a ``go_aspect_map`` over a deterministic integer GO id
space. A GAF term absent from the supplied OBO release is skipped, and
references that carry no annotation in the GAF are dropped before
retrieval.

Stage 2: embeddings
~~~~~~~~~~~~~~~~~~~

Embeddings are resolved in one of two modes:

* **Bind-mount mode.** ``--query_embeds`` and ``--reference_embeds``
  point at parquet files with pre-computed vectors. The slim image (no
  torch) loads them directly and reindexes each to the FASTA order.
* **Self-contained mode.** Both flags are omitted. The entrypoint runs
  the backend selected with ``--backend_id`` (default ``esm2_t36_3B``)
  via :func:`protea_method.embed.embed_fasta`, caching the result on disk
  keyed by ``(backend_id, fasta_sha256)`` so re-runs skip the multi-hour
  forward pass.

Both modes produce two ``[N, D]`` matrices: one for queries, one for the
annotated references.

Stage 3: KNN search
~~~~~~~~~~~~~~~~~~~

:func:`protea_method.knn_search.search_knn` retrieves the ``k`` nearest
references for each query. All backends return distances where lower
means more similar: cosine distance ``1 - cos`` or squared L2. Retrieval
is numpy or FAISS, never pgvector (a hard rule for pools above ~500k
vectors).

Two retrieval modes are available via ``PredictConfig.aspect_separated``:

* **Unified KNN** (default): one index across all reference embeddings.
* **Aspect-separated KNN**: three indices, one per GO aspect
  (F / P / C), each restricted to references annotated in that aspect.
  This guarantees BPO / MFO / CCO candidates per query and lifts the BPO
  recall ceiling that a unified index suffers when the globally-nearest
  neighbours all annotate a single aspect.

Stage 4: vote tally and features
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

For each query, the KNN neighbours vote for the GO terms they carry. The
tally records, per ``(query, go_term)``: vote count, min and mean
distance, the ``k`` position of the first donor, GO-term frequency, and
reference annotation density. These become the base reranker features.

When ``compute_v6_features`` is set,
:func:`protea_method.feature_enricher.enrich_v6_features` adds the v6
family on top: Anc2Vec neighbour-centroid cosines (GO-DAG proximity),
optional taxonomy-voter fractions, and the PCA projection of the query
embedding. Alignment and taxonomy pair features are merged from a
caller-supplied ``pair_features`` map when present.

Stage 5: reranking
~~~~~~~~~~~~~~~~~~

If a booster is supplied, candidates are re-scored:

* a single ``booster`` scores every row, or
* ``boosters_by_aspect`` scores each aspect with its own model and leaves
  un-covered aspects distance-ranked (selective rerank).

Lambdarank scores are unbounded reals, so they are calibrated through a
sigmoid into ``[0, 1]``; binary boosters already emit probabilities and
pass through unchanged.

Stage 6: write the TSV
~~~~~~~~~~~~~~~~~~~~~~~

:func:`protea_method.io.write_lafa_tsv` projects each prediction dict onto
the LAFA 3-column tuple ``(Query_ID, GO_Term, Score)``. The score is the
``reranker_score`` when present, else ``1 - min_distance`` (cosine
similarity) so the column is always populated. Rows are sorted by
``(Query_ID, GO_Term)`` for reproducibility and all three aspects are
interleaved in a single file, optionally gzipped by output suffix.

.. note::

   Stages 4 and 5 are capabilities of the :func:`~protea_method.pipeline.predict`
   library, not of the current container default. ``method_main.py``
   today calls ``predict`` with ``compute_v6_features=False`` and no
   booster, so the shipped container runs a unified KNN and writes the
   ``1 - min_distance`` similarity as the score. The reranker and v6
   feature stages are wired in by callers (the platform worker, the lab)
   that supply a booster and enable the toggles.

.. _own-reference:

Own-reference, temporal-cutoff design
-------------------------------------

The submission consumes only the query (test) sequences from a benchmark
and scores them against PROTEA's own reference pool, frozen to a temporal
cutoff. This section explains why, and how the no-future-data guard
enforces it.

Own reference, not the benchmark's train set
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

A LAFA run is handed a query FASTA, a training FASTA, an annotation GAF,
and a GO ontology. ``protea-method`` uses the query FASTA and the GO
ontology directly, but scores against PROTEA's own reference pool rather
than rebuilding an index from the supplied training set. The PROTEA pool
is the curated, embedded reference bank the platform already maintains;
reusing it keeps the container reproducible and aligned with the lab-side
validation numbers. The LAFA container guide and annotation rules permit
external-data methods, so no special permission is needed to bring an own
reference.

The hard rule: no future data
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

A submission must not see anything dated after its declared cutoff
``t0``. Every reference annotation, embedding, candidate, and feature has
to read only data on or before ``t0``. PROTEA enforces this on the export
side (the reference pool is built from annotations no later than the
band's ``t0``). ``protea-method`` enforces the inference side with a
self-contained guard, so the slim container needs no ``protea-core``
dependency.

The ``--cutoff`` knob
~~~~~~~~~~~~~~~~~~~~~~

A single ``--cutoff`` flag threads the cutoff through the run. It accepts
either a registered band name (``v226``, ``v227``, ...) or a bare
``YYYY-MM-DD`` date. The module :mod:`protea_method.cutoff` resolves the
knob to a ``t0`` date:

* a band name maps through ``BAND_CUTOFFS`` (mirrored from PROTEA's
  ``band_registry`` so both sides agree by construction),
* a token that embeds a band (for example a dataset id) is matched,
* a bare date is parsed directly.

An unresolvable knob raises, so a typo never silently disables the guard.

The guard itself
~~~~~~~~~~~~~~~~~

Before any GAF, embedding, or KNN work runs (fail fast),
:func:`protea_method.cutoff.assert_obo_not_after_cutoff` reads the GO
ontology's ``data-version:`` header and orders it against ``t0``:

* if the header carries a parseable ``YYYY-MM-DD`` release date and it is
  **after** ``t0``, the guard raises
  :class:`protea_method.cutoff.CutoffViolationError`;
* if the header has no parseable date, ordering is impossible and the
  guard is a no-op.

A frozen container therefore cannot silently propagate against a future
ontology: supply the GO release current at the cutoff. A retrained
container changes only this one knob. See :doc:`reference/cutoff` for the
full API.
