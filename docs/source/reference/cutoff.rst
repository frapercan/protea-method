cutoff
======

Self-contained no-future-data guard for the LAFA entrypoint. Resolves a
single ``--cutoff`` knob (a band name or a ``YYYY-MM-DD`` date) to a
``t0`` date and refuses a GO ontology dated after it, so a frozen
container cannot propagate against a future ontology. See
:doc:`../own_reference_cutoff` for the design rationale.

.. automodule:: protea_method.cutoff
   :members:
   :undoc-members:
   :show-inheritance:
   :no-index:
