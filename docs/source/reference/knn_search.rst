knn_search
==========

K-nearest-neighbour search backends for GO term prediction. Provides a
numpy brute-force backend for small reference sets and a FAISS backend
for large sets (greater than 100K vectors). Both backends expose the
same ``search_knn`` interface returning distances rather than
similarities.

.. automodule:: protea_method.knn_search
   :members:
   :undoc-members:
   :show-inheritance:
