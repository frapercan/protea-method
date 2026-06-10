LAFA inference flow
===================

This page traces a single LAFA container run end to end, from the input
FASTA to the 3-column predictions TSV. The orchestration lives in
``method_main.py`` (the entrypoint) and :func:`protea_method.pipeline.predict`
(the library core).

The pipeline at a glance::

   query FASTA ─┐
                ├─> embeddings ─> KNN search ─> vote tally ─> features ─> reranker ─> 3-col TSV
   reference ───┘

Stage 1: load inputs
--------------------

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
``go_id_map`` and ``go_aspect_map`` over a deterministic integer GO id
space. References that carry no annotation in the supplied GAF are
dropped before retrieval.

Stage 2: embeddings
-------------------

Embeddings are resolved in one of two modes:

* **Bind-mount mode.** ``--query_embeds`` and ``--reference_embeds``
  point at parquet files with pre-computed vectors. The slim image
  (no torch) loads them directly and reindexes each to the FASTA order.
* **Self-contained mode.** Both flags are omitted. The entrypoint runs
  the backend selected with ``--backend_id`` (default ``esm2_t36_3B``)
  via :func:`protea_method.embed.embed_fasta`, caching the result on disk
  keyed by ``(backend_id, fasta_sha256)`` so re-runs skip the multi-hour
  forward pass.

Both modes produce two ``[N, D]`` matrices: one for queries, one for the
annotated references.

Stage 3: KNN search
-------------------

:func:`protea_method.knn_search.search_knn` retrieves the ``k`` nearest
references for each query. All backends return distances (lower means
more similar): cosine distance ``1 - cos`` or squared L2.

Two retrieval modes are available via
``PredictConfig.aspect_separated``:

* **Unified KNN** (default): one index across all reference embeddings.
* **Aspect-separated KNN**: three indices, one per GO aspect
  (F / P / C), each restricted to references annotated in that aspect.
  This guarantees BPO / MFO / CCO candidates per query and lifts the BPO
  recall ceiling a unified index suffers when the globally-nearest
  neighbours all annotate a single aspect.

Stage 4: vote tally and features
--------------------------------

For each query, the KNN neighbours vote for the GO terms they carry.
The tally records, per ``(query, go_term)``: vote count, min and mean
distance, k position of the first donor, GO-term frequency, and
reference annotation density. These become the base re-ranker features.

When ``compute_v6_features`` is set, :func:`protea_method.feature_enricher.enrich_v6_features`
adds the v6 family on top: Anc2Vec neighbour centroid cosines (GO-DAG
proximity), optional taxonomy-voter fractions, and the PCA projection of
the query embedding. Alignment and taxonomy pair features are merged
from a caller-supplied ``pair_features`` map when present.

Stage 5: re-ranking
-------------------

If a booster is supplied, candidates are re-scored:

* a single ``booster`` scores every row, or
* ``boosters_by_aspect`` scores each aspect with its own model and
  leaves un-covered aspects distance-ranked (selective rerank).

Lambdarank scores are unbounded reals, so they are calibrated through a
sigmoid into ``[0, 1]``; binary boosters already emit probabilities and
pass through unchanged.

Stage 6: write the TSV
----------------------

:func:`protea_method.io.write_lafa_tsv` projects each prediction dict onto
the LAFA 3-column tuple ``(Query_ID, GO_Term, Score)``. The score is the
``reranker_score`` when present, else ``1 - min_distance`` (cosine
similarity) so the column is always populated. Rows are sorted by
``(Query_ID, GO_Term)`` for reproducibility and all three aspects are
interleaved in a single file, optionally gzipped by output suffix.
