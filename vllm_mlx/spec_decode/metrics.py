# SPDX-License-Identifier: Apache-2.0
"""
Statistics tracking for speculative decoding.

Tracks acceptance rates, draft counts, and per-position acceptance
statistics to monitor speculation quality and guide tuning decisions.
"""

from dataclasses import dataclass, field


@dataclass
class SpecDecodeStats:
    """
    Aggregated statistics for speculative decoding performance.

    Tracks how many tokens were drafted, how many were accepted, and
    per-position acceptance rates across multiple speculation rounds.

    Attributes:
        num_drafts: Total number of speculation rounds performed.
        num_draft_tokens: Total number of draft tokens proposed.
        num_accepted_tokens: Total number of draft tokens accepted by the
            target model.
        acceptance_rate_per_position: Per-position acceptance rates. Index i
            holds the acceptance rate for the (i+1)-th draft token in the
            sequence. Updated as a running average.
    """

    num_drafts: int = 0
    num_draft_tokens: int = 0
    num_accepted_tokens: int = 0
    acceptance_rate_per_position: list[float] = field(default_factory=list)

    # Internal counters for per-position running averages
    _position_accepted_counts: list[int] = field(default_factory=list)
    _position_total_counts: list[int] = field(default_factory=list)

    def acceptance_rate(self) -> float:
        """
        Compute the overall acceptance rate.

        Returns:
            The fraction of draft tokens that were accepted, or 0.0
            if no tokens have been drafted yet.
        """
        if self.num_draft_tokens == 0:
            return 0.0
        return self.num_accepted_tokens / self.num_draft_tokens

    def mean_accepted_length(self) -> float:
        """
        Compute the mean number of accepted tokens per speculation round.

        Returns:
            The average number of accepted tokens per draft round, or 0.0
            if no drafts have been performed.
        """
        if self.num_drafts == 0:
            return 0.0
        return self.num_accepted_tokens / self.num_drafts

    def update(
        self,
        num_drafted: int,
        num_accepted: int,
        position_accepted: list[bool],
    ) -> None:
        """
        Update statistics with results from one speculation round.

        Args:
            num_drafted: Number of draft tokens proposed in this round.
            num_accepted: Number of draft tokens accepted in this round.
            position_accepted: Boolean list where position_accepted[i] is True
                if the (i+1)-th draft token was accepted. Length should equal
                num_drafted.
        """
        self.num_drafts += 1
        self.num_draft_tokens += num_drafted
        self.num_accepted_tokens += num_accepted

        # Expand per-position counters if needed
        while len(self._position_accepted_counts) < len(position_accepted):
            self._position_accepted_counts.append(0)
            self._position_total_counts.append(0)

        # Update per-position counts and recompute rates
        for i, accepted in enumerate(position_accepted):
            self._position_total_counts[i] += 1
            if accepted:
                self._position_accepted_counts[i] += 1

        # Recompute per-position acceptance rates
        self.acceptance_rate_per_position = [
            self._position_accepted_counts[i] / self._position_total_counts[i]
            if self._position_total_counts[i] > 0
            else 0.0
            for i in range(len(self._position_total_counts))
        ]

    def reset(self) -> None:
        """Reset all statistics to initial state."""
        self.num_drafts = 0
        self.num_draft_tokens = 0
        self.num_accepted_tokens = 0
        self.acceptance_rate_per_position = []
        self._position_accepted_counts = []
        self._position_total_counts = []
