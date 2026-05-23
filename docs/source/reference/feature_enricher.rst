feature_enricher
================

Adds alignment, taxonomy, Anc2Vec, and lineage features to KNN
candidates. Produces the feature matrix that the LightGBM re-ranker
consumes. The feature schema is versioned via ``protea-contracts``
``ALL_FEATURES`` and pinned by ``feature_schema_sha`` in each trained
booster.

.. automodule:: protea_method.feature_enricher
   :members:
   :undoc-members:
   :show-inheritance:
   :no-index:
