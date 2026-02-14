# SPDX-License-Identifier: Apache-2.0
"""
Abstract base class for speculative decoding proposers.

A proposer generates draft tokens given a context sequence. Different
implementations (n-gram, EAGLE, draft model) provide different tradeoffs
between draft quality (acceptance rate) and drafting cost.

The propose() method works per-request (single sequence). The caller
(scheduler/runtime) is responsible for batching across requests.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass


@dataclass
class ProposerConfig:
    """
    Base configuration for all proposers.

    Subclasses should extend this with method-specific fields
    (e.g., n-gram window size, draft model path).

    Attributes:
        num_speculative_tokens: Default number of draft tokens to propose (k).
    """

    num_speculative_tokens: int = 5


class BaseProposer(ABC):
    """
    Abstract base class for draft token proposers.

    Each proposer implements a strategy for generating candidate tokens
    that will be verified by the target model. Proposers operate on a
    single request at a time; the runtime handles batching.

    Subclasses must implement:
        - propose(): Generate k draft tokens given a context.
        - reset(): Clear any internal state.
    """

    def __init__(self, config: ProposerConfig) -> None:
        """
        Initialize the proposer with configuration.

        Args:
            config: Proposer configuration parameters.
        """
        self.config = config

    @abstractmethod
    def propose(self, token_ids: list[int], k: int) -> list[int]:
        """
        Propose k draft tokens given context token_ids.

        Args:
            token_ids: The context token IDs (prompt + generated so far).
            k: Number of draft tokens to propose.

        Returns:
            A list of k proposed token IDs. May return fewer than k if the
            proposer cannot generate enough candidates (e.g., n-gram miss).
        """
        ...

    @abstractmethod
    def reset(self) -> None:
        """
        Reset internal state.

        Called when starting a new request or when the proposer needs
        to clear cached state (e.g., n-gram tables, draft model KV cache).
        """
        ...
