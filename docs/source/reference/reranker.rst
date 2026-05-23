reranker
========

LightGBM booster load, dataset preparation, and apply + score
operations. The booster is loaded from bytes so it can be retrieved
from PROTEA's ``/reranker-models/{id}/artifact`` endpoint and applied
in-process without touching the filesystem.

.. automodule:: protea_method.reranker
   :members:
   :undoc-members:
   :show-inheritance:
   :no-index:
