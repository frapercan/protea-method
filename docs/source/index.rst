protea-method
=============

**protea-method** is the pure inference library of the PROTEA stack.
It provides KNN search, v6 feature enrichment, LightGBM re-ranker
application, and the ``predict`` pipeline orchestrator as a
dependency-light package (no FastAPI, no SQLAlchemy, no protea-core).

The library is consumed by LAFA-style inference containers and by the
PROTEA platform worker that dispatches batch prediction jobs.

Contents
--------

.. toctree::
   :maxdepth: 2
   :caption: Reference

   reference/index
