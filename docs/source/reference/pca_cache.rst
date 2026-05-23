pca_cache
=========

Lazy PCA fitting and loading for embedding compression. Fits a
``PCA(16)`` on the full pool (train + test, transductive, unsupervised),
caches by ``config_id``, and serves cached projections to the pipeline
to reduce the feature-matrix dimensionality before re-ranking.

.. automodule:: protea_method.pca_cache
   :members:
   :undoc-members:
   :show-inheritance:
   :no-index:
