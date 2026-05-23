protea-method
=============

**protea-method** is the LAFA submission layer and pure inference library
of the PROTEA stack. It wraps the PROTEA pipeline for FunctionBench,
providing KNN search, feature enrichment, LightGBM re-ranker application,
and the ``predict`` pipeline orchestrator as a dependency-light package
(no FastAPI, no SQLAlchemy, no protea-core).

The library is consumed by the LAFA inference container (``method_main.py``)
and by the PROTEA platform worker that dispatches batch prediction jobs.
It is also the package shipped to consumers who want to run predictions
without the full platform stack.

Contents
--------

.. toctree::
   :maxdepth: 2
   :caption: Reference

   reference/index

.. toctree::
   :maxdepth: 1
   :caption: Submission

   reference/method_main
