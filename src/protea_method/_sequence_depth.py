"""Depth counted in sequences rather than in proteins.

A neighbour list is ordered by distance and numbered by position, and
that number is what ``k_position`` records. It counts proteins. Two
neighbours that are the same sequence occupy two positions, so a cut at
position ``d`` does not admit ``d`` distinct sequences, it admits
however many the bank happened to duplicate. In this corpus 38,694
sequences are shared by more than one protein, one of them by 114, so
the difference is not a rounding detail.

``sequence_rank`` numbers the same list by distinct sequence instead:
neighbours that carry the same sequence share a rank, and the next new
sequence takes the next integer. The rank is fixed at retrieval and
does not move when the list is later cut, which is what makes a cut at
depth ``d`` mean the same thing in every arm.

The map from accession to sequence identity has to arrive complete. A
neighbour missing from it cannot be ranked, and guessing a rank would
put a number in the column that nothing produced, so the absence is
raised rather than filled.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence

__all__ = ("SequenceIdentityMissingError", "dense_sequence_ranks")


class SequenceIdentityMissingError(RuntimeError):
    """A neighbour was retrieved whose sequence identity was not supplied."""


def dense_sequence_ranks(
    neighbours: Sequence[tuple[str, float]],
    sequence_keys: Mapping[str, str],
) -> list[int]:
    """Number one ordered neighbour list by distinct sequence.

    Args:
        neighbours: One query's neighbours, ordered as retrieved. Only
            the accession is read; the distance is ignored, because the
            order already carries it.
        sequence_keys: Accession to sequence identity. Every accession
            in ``neighbours`` must be present.

    Returns:
        One rank per position, parallel to ``neighbours``. Ranks are
        dense and start at 1: the first sequence seen is 1, a repeat of
        it is 1 again, the next new sequence is 2.

    Raises:
        SequenceIdentityMissingError: If any neighbour has no identity. The
            message names the first offender and how many share its
            fate, so the caller can tell a gap from a wrong map.
    """
    absent = [acc for acc, _ in neighbours if acc not in sequence_keys]
    if absent:
        raise SequenceIdentityMissingError(
            f"{len(absent)} of {len(neighbours)} neighbours have no sequence "
            f"identity, the first being {absent[0]!r}. Depth cannot be counted "
            f"in sequences over a bank that is only partly mapped."
        )
    seen: dict[str, int] = {}
    ranks: list[int] = []
    for acc, _ in neighbours:
        key = sequence_keys[acc]
        rank = seen.get(key)
        if rank is None:
            rank = len(seen) + 1
            seen[key] = rank
        ranks.append(rank)
    return ranks
