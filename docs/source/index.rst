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

Where to start
--------------

* New here? Read the :doc:`overview` for the what and why, then run the
  :doc:`quickstart`.
* Shipping a submission? See :doc:`container_usage` for the LAFA container
  contract and the two embedding modes.
* Want the mechanics? :doc:`inference_flow` walks the full path from a
  query FASTA to the 3-column TSV.
* Care about leakage? :doc:`own_reference_cutoff` explains the
  own-reference, temporal-cutoff design.

.. toctree::
   :maxdepth: 2
   :caption: Guide

   overview
   quickstart
   inference_flow
   own_reference_cutoff
   container_usage
   contributing

.. toctree::
   :maxdepth: 2
   :caption: API Reference

   reference/index

.. toctree::
   :maxdepth: 1
   :caption: Submission

   reference/method_main

Indices
-------

* :ref:`genindex`
* :ref:`modindex`
