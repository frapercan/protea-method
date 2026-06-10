protea-method
=============

**protea-method** is the standalone inference artefact of the PROTEA
protein function annotation stack. It is the frozen, dependency-light
library that turns a batch of query protein sequences into ranked Gene
Ontology (GO) predictions, packaged so it can be deployed on its own,
with no platform behind it.

The problem it solves
---------------------

PROTEA the platform is a full research system: FastAPI, SQLAlchemy,
Postgres, RabbitMQ, an object store, an ORM, and a fleet of workers that
train and evaluate models. None of that is needed to *score* proteins
with an already-trained model. A `FunctionBench <https://functionbench.net/>`_
submission, in particular, has to be a self-contained container that
reads a few files and writes one file, with no database and no network.

``protea-method`` is the deliberate cut line. It holds exactly the code
on the inference path (KNN retrieval, feature computation, and reranker
application) wired together behind a single
:func:`~protea_method.pipeline.predict` entry point, and it carries a
hard rule against any runtime dependency on ``fastapi``, ``sqlalchemy``,
or ``protea-core``. The default ``pip install protea-method`` is
torch-free. The result is a reproducible artefact: the same code path
that the platform runs in production also ships, unchanged, as the LAFA
submission container.

Its role in the stack
---------------------

The package sits between two neighbours and consumes what they produce:

* ``protea-contracts`` supplies the canonical feature schema
  (``ALL_FEATURES``, the PCA dimension, the label column). It is
  SemVer-coordinated, so a booster trained in the lab scores identically
  here, and a breaking schema change forces a major bump on both sides.
* ``protea-backends`` supplies the protein language model embedders (the
  ESM family, ProstT5, Ankh, ESM3-C). They are an optional extra; the
  bind-mount container path does not need them at all.

The platform side materialises the reference pool and the trained
boosters; this library consumes them. The contract at the boundary is
simply: hand it embeddings, reference annotations, and (optionally) a
booster, and it returns ranked predictions. The same library is also
imported by the PROTEA platform worker that dispatches batch prediction
jobs, so there is one inference code path, not two.

What lives here
---------------

The library is small and every module has one job:

* :mod:`protea_method.pipeline` is the spine: the
  :func:`~protea_method.pipeline.predict` orchestrator and its
  ``PredictConfig``. Everything else is a stage it calls.
* :mod:`protea_method.knn_search` retrieves the ``k`` nearest reference
  embeddings per query (numpy or FAISS; never pgvector).
* :mod:`protea_method.feature_enricher`, :mod:`protea_method.anc2vec`,
  :mod:`protea_method.lineage`, and :mod:`protea_method.pca_cache`
  compute the reranker feature families on top of the raw KNN votes.
* :mod:`protea_method.reranker` applies a trained LightGBM booster to the
  enriched candidates and calibrates its scores.
* :mod:`protea_method.cutoff` is the self-contained no-future-data guard
  that enforces the temporal cutoff at inference time.
* :mod:`protea_method.io` adapts LAFA's file formats (FASTA, GAF, OBO in;
  3-column TSV out) to the in-memory shapes ``predict`` expects.
* :mod:`protea_method.embed` is the optional in-container embedder, used
  only when pre-computed embeddings are not bind-mounted.

``method_main.py``, at the repository root, is the LAFA CLI entrypoint
that wires these modules into the standard container interface.

Where to start
--------------

* New here? Read the :doc:`overview` for the abstractions, the
  end-to-end inference flow, and the own-reference cutoff design, then
  run the :doc:`quickstart`.
* Shipping a submission? See :doc:`container_usage` for the LAFA
  container contract and the two embedding modes.
* Working on the code? Symbol-level docs are in the
  :doc:`reference/index`, and the workflow is in :doc:`contributing`.

.. toctree::
   :maxdepth: 2
   :caption: Guide
   :hidden:

   overview
   quickstart
   container_usage

.. toctree::
   :maxdepth: 2
   :caption: Reference
   :hidden:

   reference/index
   contributing

Indices
-------

* :ref:`genindex`
* :ref:`modindex`
