knn_search
==========

FAISS IVFFlat and numpy chunked KNN search. Hard constraint: pgvector
is never used here. All KNN backends are either FAISS IVFFlat (for
large reference sets) or numpy brute-force chunked (for small sets or
when FAISS is unavailable).

.. automodule:: protea_method.knn_search
   :members:
   :undoc-members:
   :show-inheritance:
   :no-index:
