"""The donors behind one term, kept so a later cut can recount them.

A stored prediction row is one ``(protein, go_term)`` pair carrying
aggregates: how many votes the term got, the mean distance of the
voters, the fraction of the neighbourhood that voted. Those aggregates
are functions of the neighbourhood the retrieval used. Truncating that
neighbourhood afterwards does not change them, so an arm cut to depth 2
carries a consensus measured over depth 30 and reports it as if it were
its own. Nothing fails; the numbers are simply about a different
candidate set than the one they are labelled with.

The ledger is the missing detail. One entry per distinct donor, holding
where that donor sat in the list and how far it was, so any depth cut
can recount rather than inherit. It is three parallel lists rather than
a list of records because it is stored in array columns and read by
arithmetic, not by object graphs.

It also fixes what a vote is. The tally increments once per annotation
row, and 37.6 per cent of ``(protein, term)`` pairs in this corpus carry
more than one such row, up to sixteen. So a single donor can vote
sixteen times for one term, which is why the stored fraction of a
ten-neighbour retrieval reaches 4.9. The ledger holds donors, so a count
taken from it is a count of voters.
"""

from __future__ import annotations

from dataclasses import dataclass, field

__all__ = ("DonorLedger",)


@dataclass
class DonorLedger:
    """The distinct donors of one term, in the order the retrieval found them.

    A donor met twice keeps its shallowest sighting. That can happen when
    the same reference carries the term under several evidence codes, and
    it is the difference between counting voters and counting paperwork.
    """

    #: Donor accessions, in first-sighting order.
    accessions: list[str] = field(default_factory=list)
    #: Position of each donor in the neighbour list, counted in proteins.
    k_positions: list[int] = field(default_factory=list)
    #: Position of each donor counted in distinct sequences, when known.
    sequence_ranks: list[int] = field(default_factory=list)
    #: Distance from the query to each donor.
    distances: list[float] = field(default_factory=list)
    _seen: dict[str, int] = field(default_factory=dict, repr=False)

    def record(
        self,
        accession: str,
        k_position: int,
        sequence_rank: int | None,
        distance: float,
    ) -> None:
        """Note one donor of this term, ignoring a repeat of one already held.

        Args:
            accession: The donor. Repeats are dropped rather than merged,
                because the first sighting is the shallowest and a cut
                admits a donor by its shallowest position.
            k_position: Where the donor sat, counted in proteins.
            sequence_rank: Where it sat counted in distinct sequences, or
                None when no sequence map was supplied. A ledger is
                either wholly ranked by sequence or not at all, so a None
                here leaves ``sequence_ranks`` short and
                :meth:`sequence_ranks_or_none` reports that plainly.
            distance: Distance from the query to this donor.
        """
        if accession in self._seen:
            return
        self._seen[accession] = len(self.accessions)
        self.accessions.append(accession)
        self.k_positions.append(k_position)
        self.distances.append(distance)
        if sequence_rank is not None:
            self.sequence_ranks.append(sequence_rank)

    def sequence_ranks_or_none(self) -> list[int] | None:
        """The sequence ranks, or None if they were not known for every donor.

        A partial column would be read as "these donors are shallow and
        the rest are missing" rather than "this run did not count in
        sequences", so the partial case is refused in favour of nothing.
        """
        if len(self.sequence_ranks) != len(self.accessions):
            return None
        return self.sequence_ranks

    def __len__(self) -> int:
        return len(self.accessions)
