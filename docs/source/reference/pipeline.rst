pipeline
========

End-to-end inference orchestrator. Wires the KNN search, feature
enrichment (alignment, taxonomy, Anc2Vec, lineage), and LightGBM re-ranker into the single ``predict``
function. Supports both unified KNN and aspect-separated KNN modes via
``PredictConfig.aspect_separated``.

.. automodule:: protea_method.pipeline
   :members:
   :undoc-members:
   :show-inheritance:
   :no-index:
