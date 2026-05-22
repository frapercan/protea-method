pca_cache
=========

On-disk PCA state cache for the inference pipeline. Fits
``PCA(16)`` on the full reference embedding pool for a given
configuration ID and persists ``(mean, components)`` to a ``.npz``
artifact so that subsequent workers and container runs skip the
expensive fitting step.

.. automodule:: protea_method.pca_cache
   :members:
   :undoc-members:
   :show-inheritance:
