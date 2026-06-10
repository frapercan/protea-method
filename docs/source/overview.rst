Overview
========

What it is
----------

``protea-method`` is the pure inference path of the PROTEA protein
function annotation stack, packaged as a slim, standalone library. It
holds exactly the code needed to turn a set of query protein embeddings
into ranked GO term predictions:

* K-nearest-neighbour retrieval against a reference embedding pool
  (numpy, FAISS, or torch backends; never pgvector),
* feature enrichment (alignment, taxonomy, Anc2Vec GO-DAG proximity,
  lineage, PCA),
* LightGBM re-ranker application over the enriched candidates,
* the ``predict`` orchestrator that wires the three stages together.

The same code powers the PROTEA platform worker and ships, unchanged, as
the LAFA submission container for `FunctionBench <https://functionbench.net/>`_.

Why it is a separate package
----------------------------

PROTEA the platform carries a heavy runtime: FastAPI, SQLAlchemy,
Postgres, RabbitMQ, an ORM, and an object store. A FunctionBench
submission needs none of that. ``protea-method`` is the deliberate cut
line: it has a hard rule against any runtime dependency on ``fastapi``,
``sqlalchemy``, or ``protea-core``. The default ``pip install
protea-method`` is torch-free and pulls only numpy, pandas, lightgbm,
and faiss-cpu.

That discipline buys three things:

#. **A reproducible container.** The LAFA image is a thin wrapper over
   this library; no database, no message broker, no network call is
   required to score a batch.
#. **A clean test surface.** Every public function takes already-loaded
   inputs (numpy arrays, dicts, DataFrames) and returns plain Python
   objects. The full unit suite runs in under a minute with no GPU and
   no network.
#. **A versioned feature contract.** The feature schema is shared with
   ``protea-contracts`` and SemVer-coordinated, so a booster trained in
   the lab scores identically here.

How it fits the stack
---------------------

``protea-method`` sits between two neighbours:

* ``protea-contracts`` supplies the canonical feature schema
  (``ALL_FEATURES``, the PCA dimension, the label column). A breaking
  change there forces a major bump here and re-training of every
  downstream booster.
* ``protea-backends`` supplies the protein language model embedders
  (ESM family, ProstT5, Ankh, ESM3-C). They are an optional extra; the
  bind-mount container path does not need them at all.

The platform side (``PROTEA``) materialises the reference pool and the
trained boosters; this library consumes them. The boundary is "hand me
embeddings and a booster, I will return ranked predictions".
