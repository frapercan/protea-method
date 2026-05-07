"""Lineage features: candidate GO term vs the query's pre-cutoff
"known" GO set.

Adds four columns per prediction dict:

- ``lineage_is_ancestor_of_known``: 1.0 iff the candidate is an
  ancestor of at least one known GO term.
- ``lineage_is_descendant_of_known``: 1.0 iff the candidate has at
  least one known term as ancestor.
- ``lineage_ancestor_of_count``: number of known terms whose
  ancestor closure contains the candidate.
- ``lineage_descendant_of_count``: number of known terms in the
  candidate's own ancestor closure.

The features capture "how related is this candidate to what we
already know about this protein" in a way the existing v6 features
miss: anc2vec-cosine asks for embedding similarity, which is
neither ``is_a`` nor ``part_of`` aware. A reranker can use these
flags to boost candidates that sit on the same DAG branch as the
known-GO scaffold even when their Anc2Vec vectors happen to be far.

The function takes the pre-built ``parents`` map (``go_id → list of
immediate parents``, both ``is_a`` and ``relationship: part_of``
edges) and a ``known_by_protein`` mapping; it does not touch any DB
or filesystem. The OBO file is a separate input owned by the
caller.
"""

from __future__ import annotations

from typing import Any

#: Feature columns this module writes into each prediction dict.
LINEAGE_FEATURE_KEYS: tuple[str, ...] = (
    "lineage_is_ancestor_of_known",
    "lineage_is_descendant_of_known",
    "lineage_ancestor_of_count",
    "lineage_descendant_of_count",
)


def _ancestor_closure(
    go_id: str,
    parents: dict[str, list[str]],
    cache: dict[str, frozenset[str]],
) -> frozenset[str]:
    """BFS the ancestor closure including ``go_id`` itself; memoised."""
    cached = cache.get(go_id)
    if cached is not None:
        return cached
    seen = {go_id}
    stack = [go_id]
    while stack:
        node = stack.pop()
        for parent in parents.get(node, ()):
            if parent not in seen:
                seen.add(parent)
                stack.append(parent)
    closure = frozenset(seen)
    cache[go_id] = closure
    return closure


def compute_lineage_features(
    predictions: list[dict[str, Any]],
    *,
    parents: dict[str, list[str]],
    known_by_protein: dict[str, set[str]],
) -> None:
    """Compute the 4 lineage features and merge them in-place.

    Each ``pred`` is expected to carry ``protein_accession`` and a
    string GO accession under ``go_id`` (e.g. ``"GO:0006357"``).

    Proteins missing from ``known_by_protein`` (or with an empty
    known set) get all four features set to 0. This matches the
    convention used elsewhere for query-dependent features at
    predict time: the value is well-defined (no known terms means
    no lineage relation to report) rather than NaN.

    The implementation memoises ancestor closures across calls
    within one batch so the cost is dominated by candidate-set size,
    not by per-prediction repeated walks.
    """
    if not predictions:
        return

    cache: dict[str, frozenset[str]] = {}

    for pred in predictions:
        prot = pred.get("protein_accession", "")
        cand = pred.get("go_id", "")
        known = known_by_protein.get(prot)

        if not known or not cand:
            pred["lineage_is_ancestor_of_known"] = 0.0
            pred["lineage_is_descendant_of_known"] = 0.0
            pred["lineage_ancestor_of_count"] = 0.0
            pred["lineage_descendant_of_count"] = 0.0
            continue

        # The candidate is an ancestor of a known term k iff
        # cand ∈ ancestors(k). Walk each known once (cached).
        ancestor_of_count = 0
        for k in known:
            if cand in _ancestor_closure(k, parents, cache):
                ancestor_of_count += 1

        # The candidate is a descendant of a known term k iff
        # k ∈ ancestors(cand). Walk the candidate once.
        cand_ancestors = _ancestor_closure(cand, parents, cache)
        descendant_of_count = sum(1 for k in known if k in cand_ancestors)
        # Subtract self-overlap so a candidate that equals one of
        # the known terms is not double-counted as both ancestor
        # and descendant of itself.
        if cand in known:
            ancestor_of_count -= 1
            descendant_of_count -= 1

        pred["lineage_is_ancestor_of_known"] = 1.0 if ancestor_of_count > 0 else 0.0
        pred["lineage_is_descendant_of_known"] = 1.0 if descendant_of_count > 0 else 0.0
        pred["lineage_ancestor_of_count"] = float(ancestor_of_count)
        pred["lineage_descendant_of_count"] = float(descendant_of_count)


__all__ = ["LINEAGE_FEATURE_KEYS", "compute_lineage_features"]
