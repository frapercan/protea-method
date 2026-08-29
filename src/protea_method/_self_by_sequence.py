"""A query is not its own neighbour, and neither is its twin.

Excluding a query from its own neighbourhood by ACCESSION removes its
identity and leaves its similarity. In this bank 38,694 sequences belong
to more than one protein, and 784 of the 14,032 queries in the evaluation
window share their sequence with another protein, so for those the
accession-based filter drops the query and leaves a copy of it at
distance zero under a different name.

So the exclusion is by sequence. A neighbour whose sequence is the
query's own is dropped whatever it is called, which is what the flag was
always meant to say.

WHY THE SEARCH HAS TO ASK FOR MORE. Dropping after the fact costs depth:
ask for thirty and drop two and you scored twenty-eight. Asking for one
extra is enough only when a query has exactly one self, which is the
assumption the accession-based version made. Counted here instead: the
queries in this window have between zero and forty-five twins, and 94.4
per cent have none, so a fixed margin is either too small for the tail or
wasteful for almost everyone. The margin is measured from the bank.
"""

from __future__ import annotations

import collections
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    import numpy as np

from protea_method._sequence_depth import SequenceIdentityMissingError

__all__ = (
    "Queries",
    "drop_own_sequence",
    "extra_neighbours_for",
    "self_margin",
    "without_own_sequence",
)


def extra_neighbours_for(
    query_accessions: Sequence[str],
    reference_accessions: Sequence[str],
    sequence_keys: Mapping[str, str],
) -> int:
    """How many neighbours beyond k the search must return to survive the drop.

    The worst case over the batch, not the average: one query with forty
    twins would otherwise come back short while every other query is fine,
    and a shortfall on one query is invisible in an aggregate.

    Args:
        query_accessions: The queries about to be searched.
        reference_accessions: The bank they are searched against.
        sequence_keys: Accession to sequence identity, covering both.

    Returns:
        The largest number of bank entries that share any one query's
        sequence. Zero when no query is represented in the bank, which is
        the common case and costs nothing.

    Raises:
        SequenceIdentityMissingError: If a query has no sequence identity.
            Without it the query's own sequence cannot be recognised, so
            the exclusion would silently not happen, which is the failure
            this replaces.
    """
    unmapped = [a for a in query_accessions if a not in sequence_keys]
    if unmapped:
        raise SequenceIdentityMissingError(
            f"{len(unmapped)} of {len(query_accessions)} queries have no sequence "
            f"identity, the first being {unmapped[0]!r}. A query whose sequence is "
            f"unknown cannot be excluded from its own neighbourhood, and excluding "
            f"it by accession alone would leave a copy of it at distance zero."
        )
    in_bank: collections.Counter[str] = collections.Counter(
        sequence_keys[a] for a in reference_accessions if a in sequence_keys
    )
    return max((in_bank[sequence_keys[a]] for a in query_accessions), default=0)


def without_own_sequence(
    neighbours: Sequence[Sequence[tuple[str, float]]],
    query_accessions: Sequence[str],
    k: int,
    sequence_keys: Mapping[str, str],
) -> list[list[tuple[str, float]]]:
    """Drop every neighbour carrying the query's own sequence, then cut to k.

    Args:
        neighbours: One list per query, positionally paired with
            ``query_accessions`` and already ordered by distance.
        query_accessions: The queries, in the same order.
        k: The depth to return once the drop has happened.
        sequence_keys: Accession to sequence identity.

    Returns:
        The surviving neighbours, at most ``k`` per query. A list may come
        back shorter than ``k`` when the bank did not hold enough distinct
        sequences, which is a fact about the bank rather than an error.

    Raises:
        ValueError: If the two sequences are not the same length. They are
            paired by position and nothing else, so a length mismatch
            silently pairs each query with another query's neighbours.
    """
    if len(neighbours) != len(query_accessions):
        raise ValueError(
            f"{len(neighbours)} neighbour lists for {len(query_accessions)} "
            f"queries. They are paired by position, so an unequal pairing gives "
            f"each query somebody else's neighbourhood."
        )
    kept: list[list[tuple[str, float]]] = []
    for top, accession in zip(neighbours, query_accessions, strict=True):
        own = sequence_keys.get(accession)
        kept.append(
            [pair for pair in top if own is None or sequence_keys.get(pair[0]) != own][:k]
        )
    return kept


@dataclass(frozen=True)
class Queries:
    """The queries of one call: what they are called and where they sit.

    Grouped because a search needs both and every caller had them side by
    side already. The accessions arrived late, when the exclusion stopped
    being by accession and became by sequence, and passing them separately
    would have put this function over the argument ceiling for a pair that
    is never apart.
    """

    accessions: list[str]
    embeddings: np.ndarray


def self_margin(
    cfg: Any, queries: Queries, reference_accessions: list[str]
) -> int:
    """Extra neighbours to ask for so the self-drop does not cost depth."""
    if not cfg.exclude_self_neighbour or cfg.sequence_keys is None:
        return 0
    return extra_neighbours_for(
        queries.accessions, reference_accessions, cfg.sequence_keys
    )


def drop_own_sequence(
    cfg: Any, hits: list[list[tuple[str, float]]], queries: Queries
) -> list[list[tuple[str, float]]]:
    """Remove every neighbour carrying the query's own sequence, then cut to k."""
    if not cfg.exclude_self_neighbour or cfg.sequence_keys is None:
        return hits
    return without_own_sequence(hits, queries.accessions, cfg.k, cfg.sequence_keys)
