Own-reference, temporal-cutoff design
=====================================

The submission consumes only the query (test) sequences from LAFA and
scores them against PROTEA's own reference pool, frozen to a temporal
cutoff. This page explains why, and how the no-future-data guard
enforces it.

Own reference, not LAFA train
-----------------------------

A LAFA run is handed a query FASTA, a training FASTA, an annotation GAF,
and a GO ontology. ``protea-method`` uses the query FASTA and the GO
ontology directly, but scores against PROTEA's own reference pool rather
than rebuilding an index from the supplied training set. The PROTEA pool
is the curated, embedded reference bank the platform already maintains;
reusing it keeps the container reproducible and aligned with the lab-side
validation numbers.

The LAFA container guide and annotation rules permit external-data
methods, so no special permission is needed to bring an own reference.

The hard rule: no future data
-----------------------------

A submission must not see anything dated after its declared cutoff
``t0``. Every reference annotation, embedding, candidate, and feature has
to read only data on or before ``t0``. PROTEA enforces this on the export
side (the reference pool is built from annotations no later than the
band's ``t0``). ``protea-method`` enforces the inference side with a
self-contained guard so the slim container needs no ``protea-core``
dependency.

The ``--cutoff`` knob
---------------------

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
----------------

Before any GAF, embedding, or KNN work runs (fail fast),
:func:`protea_method.cutoff.assert_obo_not_after_cutoff` reads the GO
ontology's ``data-version:`` header and orders it against ``t0``:

* if the header carries a parseable ``YYYY-MM-DD`` release date and it is
  **after** ``t0``, the guard raises
  :class:`protea_method.cutoff.CutoffViolationError`;
* if the header has no parseable date, ordering is impossible and the
  guard is a no-op.

This means a frozen container cannot silently propagate against a future
ontology: supply the GO release current at the cutoff. A retrained
container changes only this one knob.

.. code-block:: bash

   # Refuses --graph if its data-version is dated after v227's t0.
   docker run ... protea-method-lafa:latest \
       --query_file /app/data/test_sequences.fasta \
       --graph /app/data/go-basic.obo \
       --cutoff v227 \
       ...

See :doc:`reference/cutoff` for the full API.
