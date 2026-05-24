"""Contract-conformance pin-tests for protea-method.

protea-method is a library (no ``entry_points`` group) but it
re-exports several constants and the canonical ``ALL_FEATURES`` column
set from :mod:`protea_contracts`. The 2026-05-13 incident where a
column was added to ``ALL_FEATURES`` without an unconditional producer
took ~5 hours of KNN compute to discover; the symmetric class of bug
here is when protea-method's re-exports silently diverge from the
upstream contract surface (e.g., someone replaces the
``from protea_contracts import ALL_FEATURES`` with a hand-maintained
list and forgets to update it).

These tests pin the re-exports to the upstream so any drift breaks CI
on this repo's PR rather than producing a silently-wrong inference
pipeline.
"""

from __future__ import annotations

import protea_contracts

import protea_method
from protea_method import reranker as method_reranker


def test_all_features_pinned_to_contracts() -> None:
    """protea-method.ALL_FEATURES must be identical to protea_contracts.ALL_FEATURES.

    If a downstream caller adds a column to one side and forgets the
    other, the inference pipeline produces rows that the parquet
    exporter rejects with a missing-columns error (the
    ``_assert_canonical_columns`` invariant).
    """
    assert list(method_reranker.ALL_FEATURES) == list(protea_contracts.ALL_FEATURES), (
        "protea_method.ALL_FEATURES diverged from protea_contracts.ALL_FEATURES; "
        "either the upstream contract changed without a method-side bump or "
        "method-side maintains its own list (it must not)."
    )


def test_categorical_and_numeric_partition_pinned() -> None:
    """The CATEGORICAL/NUMERIC partition must round-trip via the contracts surface."""
    assert list(method_reranker.CATEGORICAL_FEATURES) == list(
        protea_contracts.CATEGORICAL_FEATURES
    )
    assert list(method_reranker.NUMERIC_FEATURES) == list(
        protea_contracts.NUMERIC_FEATURES
    )


def test_embedding_pca_dim_pinned() -> None:
    """EMBEDDING_PCA_DIM is one of the 16-vs-N drift hotspots; pin it."""
    assert method_reranker.EMBEDDING_PCA_DIM == protea_contracts.EMBEDDING_PCA_DIM


def test_label_column_pinned() -> None:
    """LABEL_COLUMN drift would silently re-target the booster's positive class."""
    assert method_reranker.LABEL_COLUMN == protea_contracts.LABEL_COLUMN


def test_package_reexports_canonical_constants() -> None:
    """The top-level ``protea_method`` package must re-export the canonical four.

    Catches the case where someone removes a re-export from
    ``protea_method.__init__`` and breaks a LAFA / runner consumer that
    imports ``from protea_method import ALL_FEATURES``.
    """
    for name in ("ALL_FEATURES", "CATEGORICAL_FEATURES", "NUMERIC_FEATURES", "LABEL_COLUMN"):
        assert hasattr(protea_method, name), (
            f"protea_method.{name} re-export missing; downstream "
            "imports `from protea_method import ALL_FEATURES` will break."
        )
        # The re-export must be value-equal to the underlying constant
        # so a typo in __init__.py that shadows the import is caught.
        assert getattr(protea_method, name) == getattr(method_reranker, name)


def test_active_feature_families_subset_of_canonical() -> None:
    """``infer_active_feature_families`` must return names from the canonical set.

    The function's output is consumed by the LAFA scoring path to drop
    inactive families; if it ever leaked an unknown name the downstream
    selector would silently drop everything.
    """
    families = method_reranker.infer_active_feature_families(
        compute_alignments=True,
        compute_taxonomy=True,
        compute_v6_features=True,
    )
    # Sanity: the helper is non-empty when every flag is on and returns
    # a subset of the canonical FEATURE_FAMILIES keyset.
    assert families, "active families should be non-empty when every flag is on"
    canonical = set(protea_contracts.FEATURE_FAMILIES)
    leaked = set(families) - canonical
    assert not leaked, (
        f"infer_active_feature_families returned unknown families {leaked!r}; "
        f"expected subset of {sorted(canonical)!r}."
    )
