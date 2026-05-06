"""Pure inference path of PROTEA: KNN, feature compute, apply reranker."""

from protea_method.anc2vec import Anc2VecIndex, get_index
from protea_method.feature_enricher import (
    ASPECT_CODES,
    NEW_V6_FEATURE_KEYS,
    enrich_v6_features,
)
from protea_method.knn_search import search_knn
from protea_method.pca_cache import load_or_fit_pca_state
from protea_method.reranker import (
    ALL_FEATURES,
    CATEGORICAL_FEATURES,
    EMBEDDING_PCA_DIM,
    LABEL_COLUMN,
    NUMERIC_FEATURES,
    apply_reranker,
    fit_embedding_pca,
    infer_active_feature_families,
    load_from_bytes,
    model_from_string,
    predict,
    prepare_dataset,
)

__version__ = "0.0.1"

__all__ = [
    "ALL_FEATURES",
    "ASPECT_CODES",
    "CATEGORICAL_FEATURES",
    "EMBEDDING_PCA_DIM",
    "LABEL_COLUMN",
    "NEW_V6_FEATURE_KEYS",
    "NUMERIC_FEATURES",
    "Anc2VecIndex",
    "__version__",
    "apply_reranker",
    "enrich_v6_features",
    "fit_embedding_pca",
    "get_index",
    "infer_active_feature_families",
    "load_from_bytes",
    "load_or_fit_pca_state",
    "model_from_string",
    "predict",
    "prepare_dataset",
    "search_knn",
]
