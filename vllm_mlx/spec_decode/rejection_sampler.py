# SPDX-License-Identifier: Apache-2.0
"""
Rejection sampling for speculative decoding.
"""

from __future__ import annotations

from dataclasses import dataclass

import mlx.core as mx


@dataclass
class RejectionResult:
    """
    Result of rejection sampling for a batch.

    Attributes:
        accepted_token_ids: Per-request list of accepted tokens.
        num_accepted: Per-request count of accepted tokens.
        bonus_token_ids: Per-request bonus token (or None).
    """

    accepted_token_ids: list[list[int]]  # per-request list of accepted tokens
    num_accepted: list[int]  # per-request count of accepted tokens
    bonus_token_ids: list[int | None]  # per-request bonus token (or None)


class RejectionSampler:
    """
    Rejection sampler for speculative decoding.

    Supports greedy and stochastic rejection methods.
    """

    def __init__(self, method: str = "greedy") -> None:
        """
        Initialize the rejection sampler.

        Args:
            method: Rejection method. Must be "greedy" or "stochastic".

        Raises:
            ValueError: If method is not supported.
        """
        if method not in {"greedy", "stochastic"}:
            raise ValueError(
                f"Invalid rejection method '{method}'. "
                "Must be 'greedy' or 'stochastic'."
            )
        self.method = method

    def __call__(
        self,
        target_logits: mx.array,
        draft_token_ids: list[list[int]],
        draft_logits: mx.array | None = None,
    ) -> RejectionResult:
        """
        Run rejection sampling for a batch of draft tokens.

        Args:
            target_logits: Target-model logits with shape (batch, k + 1, vocab).
            draft_token_ids: Per-request draft token IDs.
            draft_logits: Draft-model logits with shape (batch, k, vocab).
                Required for stochastic rejection.

        Returns:
            A RejectionResult containing accepted tokens and bonus tokens.
        """
        if self.method == "greedy":
            return self._greedy_rejection(target_logits, draft_token_ids)
        return self._stochastic_rejection(target_logits, draft_token_ids, draft_logits)

    def _greedy_rejection(
        self,
        target_logits: mx.array,
        draft_token_ids: list[list[int]],
    ) -> RejectionResult:
        """Greedy rejection using argmax matching."""
        accepted_token_ids: list[list[int]] = []
        num_accepted: list[int] = []
        bonus_token_ids: list[int | None] = []

        batch_size = len(draft_token_ids)

        for b in range(batch_size):
            request_draft = draft_token_ids[b]

            if not request_draft:
                bonus_token = int(mx.argmax(target_logits[b, 0, :]).item())
                accepted_token_ids.append([])
                num_accepted.append(0)
                bonus_token_ids.append(bonus_token)
                continue

            accepted: list[int] = []
            bonus_token: int | None = None

            for i, token in enumerate(request_draft):
                target_token = int(mx.argmax(target_logits[b, i, :]).item())
                if target_token == token:
                    accepted.append(token)
                else:
                    bonus_token = target_token
                    break

            if bonus_token is None:
                bonus_token = int(mx.argmax(target_logits[b, len(request_draft), :]).item())

            accepted_token_ids.append(accepted)
            num_accepted.append(len(accepted))
            bonus_token_ids.append(bonus_token)

        return RejectionResult(
            accepted_token_ids=accepted_token_ids,
            num_accepted=num_accepted,
            bonus_token_ids=bonus_token_ids,
        )

    def _stochastic_rejection(
        self,
        target_logits: mx.array,
        draft_token_ids: list[list[int]],
        draft_logits: mx.array | None,
    ) -> RejectionResult:
        """Stochastic rejection sampling with adjusted bonus distribution."""
        if draft_logits is None:
            raise ValueError("draft_logits must be provided for stochastic rejection.")

        p_target = mx.softmax(target_logits, axis=-1)
        p_draft = mx.softmax(draft_logits, axis=-1)

        accepted_token_ids: list[list[int]] = []
        num_accepted: list[int] = []
        bonus_token_ids: list[int | None] = []

        batch_size = len(draft_token_ids)
        eps = mx.array(1e-10)

        for b in range(batch_size):
            request_draft = draft_token_ids[b]

            if not request_draft:
                sampled = mx.random.categorical(target_logits[b, 0, :])
                bonus_token = int(sampled.item())
                accepted_token_ids.append([])
                num_accepted.append(0)
                bonus_token_ids.append(bonus_token)
                continue

            accepted: list[int] = []
            bonus_token: int | None = None

            for i, token in enumerate(request_draft):
                p_t = p_target[b, i, token]
                p_d = p_draft[b, i, token]
                p_d_safe = mx.maximum(p_d, eps)

                ratio = float((p_t / p_d_safe).item())
                r = float(mx.random.uniform().item())

                if ratio >= r:
                    accepted.append(token)
                    continue

                adjusted = mx.maximum(p_target[b, i, :] - p_draft[b, i, :], 0.0)
                adjusted_sum = mx.sum(adjusted)

                if float(adjusted_sum.item()) > 0.0:
                    adjusted_normalized = adjusted / adjusted_sum
                else:
                    adjusted_normalized = p_target[b, i, :]

                sampled = mx.random.categorical(mx.log(adjusted_normalized))
                bonus_token = int(sampled.item())
                break

            if bonus_token is None:
                sampled = mx.random.categorical(target_logits[b, len(request_draft), :])
                bonus_token = int(sampled.item())

            accepted_token_ids.append(accepted)
            num_accepted.append(len(accepted))
            bonus_token_ids.append(bonus_token)

        return RejectionResult(
            accepted_token_ids=accepted_token_ids,
            num_accepted=num_accepted,
            bonus_token_ids=bonus_token_ids,
        )
