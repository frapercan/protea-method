Quickstart
==========

Install
-------

The default install is torch-free. KNN search, feature enrichment,
Anc2Vec, lineage, and the LightGBM re-ranker run on numpy, faiss-cpu, and
lightgbm only.

.. code-block:: bash

   pip install protea-method

Optional extras pull in heavier runtimes only when you need them:

.. code-block:: bash

   pip install "protea-method[mil]"   # gated-attention MIL head (torch)
   pip install "protea-method[esm]"   # in-container ESM-2 embedder
   pip install "protea-method[t5]"    # in-container ProstT5 embedder

Score a batch
-------------

The library entry point is :func:`protea_method.predict`. Hand it query
and reference embeddings plus the reference annotations and GO maps; it
returns a list of prediction dicts.

.. code-block:: python

   import numpy as np
   from protea_method import predict, PredictConfig

   # Query / reference embeddings, shape [N, D] (float16 or float32).
   query_embeddings = np.load("query_embeddings.npy")
   ref_embeddings = np.load("ref_embeddings.npy")

   query_accessions = ["Q1", "Q2"]
   reference_accessions = ["R1", "R2", "R3"]

   # Reference annotations keyed by accession.
   annotations = {
       "R1": [{"go_term_id": 0, "qualifier": "", "evidence_code": "EXP"}],
       "R2": [{"go_term_id": 1, "qualifier": "", "evidence_code": "IDA"}],
   }
   go_id_map = {0: "GO:0008150", 1: "GO:0003674"}
   go_aspect_map = {0: "P", 1: "F"}

   cfg = PredictConfig(k=5, metric="cosine", backend="numpy")
   predictions = predict(
       query_accessions=query_accessions,
       query_embeddings=query_embeddings,
       reference_accessions=reference_accessions,
       reference_embeddings=ref_embeddings,
       annotations=annotations,
       go_id_map=go_id_map,
       go_aspect_map=go_aspect_map,
       config=cfg,
   )
   # Each row carries protein_accession, go_id, aspect, the KNN feature
   # aggregates, and (when a booster is supplied) reranker_score.

Add a re-ranker
---------------

When a PROTEA-trained LightGBM booster is available, pass it in and the
candidates are re-scored:

.. code-block:: python

   from protea_method.reranker import load_from_bytes

   with open("booster.txt", "rb") as fh:
       booster = load_from_bytes(fh.read())

   predictions = predict(
       query_accessions=query_accessions,
       query_embeddings=query_embeddings,
       reference_accessions=reference_accessions,
       reference_embeddings=ref_embeddings,
       annotations=annotations,
       go_id_map=go_id_map,
       go_aspect_map=go_aspect_map,
       config=PredictConfig(k=15),
       booster=booster,
   )

For per-aspect selective re-ranking, load three boosters with
:func:`protea_method.pipeline.load_boosters_by_aspect` and pass them via
``boosters_by_aspect``; aspects without a booster stay distance-ranked.

Write a LAFA submission file
----------------------------

:func:`protea_method.io.write_lafa_tsv` projects the prediction dicts onto
the 3-column ``Query_ID`` / ``GO_Term`` / ``Score`` TSV FunctionBench
expects (gzipped automatically when the path ends in ``.gz``):

.. code-block:: python

   from protea_method.io import write_lafa_tsv

   n = write_lafa_tsv(predictions, "output/predictions.tsv.gz")

Next steps
----------

* :doc:`inference_flow` traces the full path end to end.
* :doc:`container_usage` shows how to run the same logic as a Docker
  container.
