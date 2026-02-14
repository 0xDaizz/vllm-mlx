# SPDX-License-Identifier: Apache-2.0
"""
MLX Distributed Communication Layer for vLLM.

This module provides distributed communication primitives for multi-node
inference on Apple Silicon using MLX's distributed backend. All broadcast
patterns use all_sum (source rank sends the value, others send zeros).

Supports JACCL (RDMA over Thunderbolt) and ring (TCP) backends.
"""

from __future__ import annotations

import logging
import os
import pickle
from dataclasses import dataclass
from typing import Any

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------


@dataclass
class InsertOp:
    """A request to insert into the running batch."""

    request_id: str
    tokens: list[int]
    max_tokens: int
    cache_info: dict | None = None


@dataclass
class StepPlan:
    """Plan broadcast from rank 0 to all workers each decode step."""

    step_id: int  # monotonically increasing counter
    inserts: list[InsertOp]  # requests to add
    removes: list[str]  # request_ids to remove
    sampling_seeds: dict[str, int]  # per-request RNG seeds
    fingerprint: str  # batch composition hash for sync verification


@dataclass
class StepResult:
    """Result of a single decode step."""

    step_id: int
    token_ids: dict[str, int]  # request_id -> sampled token


# ---------------------------------------------------------------------------
# MLXCommunicator
# ---------------------------------------------------------------------------


class MLXCommunicator:
    """
    Distributed communication wrapper for MLX.

    All broadcast patterns use ``all_sum`` — the source rank contributes
    the real value while every other rank contributes zeros, so the sum
    equals the original value on every rank.

    When ``world_size == 1`` all methods are cheap pass-throughs.
    """

    def __init__(
        self,
        backend: str = "any",
        group: Any = None,
    ) -> None:
        """Initialize the distributed group.

        Args:
            backend: ``"jaccl"``, ``"ring"``, or ``"any"`` (auto-detect).
                Ignored when *group* is provided.
            group: A pre-initialized ``mx.distributed.Group``.  When
                ``None``, a new group is created via
                ``mx.distributed.init(backend=backend)``.
        """
        import mlx.core as mx

        if group is not None:
            self._group = group
        else:
            self._group = mx.distributed.init(backend=backend, strict=False)
        self._rank = self._group.rank()
        self._world_size = self._group.size()

        logger.info(
            "MLXCommunicator initialized: rank=%d, world_size=%d, backend=%s",
            self._rank,
            self._world_size,
            backend,
        )

    # -- properties ---------------------------------------------------------

    @property
    def group(self):
        """The underlying ``mx.distributed.Group``."""
        return self._group

    @property
    def rank(self) -> int:
        return self._rank

    @property
    def world_size(self) -> int:
        return self._world_size

    @property
    def is_distributed(self) -> bool:
        """``True`` when running with more than one rank."""
        return self._world_size > 1

    # -- primitives ---------------------------------------------------------

    def barrier(self) -> None:
        """Synchronize all ranks.

        Uses ``all_sum`` on the CPU stream so the synchronization is
        evaluated eagerly (not deferred by lazy evaluation).
        """
        if not self.is_distributed:
            return

        import mlx.core as mx

        mx.eval(mx.distributed.all_sum(mx.array(1.0), group=self._group, stream=mx.cpu))

    def broadcast_int(self, value: int, src: int = 0) -> int:
        """Broadcast a single integer from *src* to every rank.

        The source rank contributes the value; all others contribute 0.
        ``all_sum`` yields the original value everywhere.
        """
        if not self.is_distributed:
            return value

        import mlx.core as mx

        buf = mx.array(value if self._rank == src else 0)
        result = mx.distributed.all_sum(buf, group=self._group)
        return result.item()

    def broadcast_tensor(self, tensor, src: int = 0):
        """Broadcast an ``mx.array`` from *src* to every rank.

        Non-source ranks substitute a zeros tensor of the same shape and
        dtype so that ``all_sum`` reproduces the original.
        """
        if not self.is_distributed:
            return tensor

        import mlx.core as mx

        if self._rank == src:
            return mx.distributed.all_sum(tensor, group=self._group)
        else:
            return mx.distributed.all_sum(
                mx.zeros_like(tensor), group=self._group
            )

    def broadcast_object(self, obj: Any, src: int = 0) -> Any:
        """Broadcast an arbitrary picklable object from *src* to all.

        Protocol (mirrors the mlx-lm server pattern):
        1. Source rank pickles the object to bytes.
        2. Broadcast the byte-length as an integer.
        3. Convert the bytes to a ``uint8`` array and broadcast.
        4. Every rank unpickles the result.

        When *src* is the calling rank the original *obj* is returned
        directly (avoids a redundant pickle round-trip on that rank).
        """
        if not self.is_distributed:
            return obj

        import mlx.core as mx

        if self._rank == src:
            data = mx.array(list(pickle.dumps(obj)), dtype=mx.uint8)
            size = data.size
            # Broadcast size
            mx.eval(mx.distributed.all_sum(mx.array(size), group=self._group))
            # Broadcast data
            mx.eval(mx.distributed.all_sum(data, group=self._group))
            return obj
        else:
            # Receive size
            size = mx.distributed.all_sum(
                mx.array(0), group=self._group
            ).item()
            # Receive data
            data = mx.distributed.all_sum(
                mx.zeros(size, dtype=mx.uint8), group=self._group
            )
            return pickle.loads(bytes(data.tolist()))

    # -- high-level helpers -------------------------------------------------

    def broadcast_step_plan(self, plan: StepPlan) -> None:
        """Broadcast a :class:`StepPlan` from rank 0 to all workers.

        Only rank 0 should call this with an actual *plan*.  Non-rank-0
        callers should use :meth:`receive_step_plan` instead.
        """
        if not self.is_distributed:
            return

        if self._rank != 0:
            logger.warning(
                "broadcast_step_plan called on rank %d; "
                "use receive_step_plan on non-zero ranks",
                self._rank,
            )
        self.broadcast_object(plan, src=0)

    def receive_step_plan(self) -> StepPlan:
        """Receive a :class:`StepPlan` on a non-rank-0 worker.

        Internally calls :meth:`broadcast_object` with ``src=0``.
        """
        return self.broadcast_object(None, src=0)

    def broadcast_tokens(self, token_ids, src: int = 0):
        """Broadcast sampled token IDs from *src* to every rank.

        Rank 0 samples tokens and uses this to synchronize all workers
        so that KV caches stay consistent.

        Args:
            token_ids: An ``mx.array`` of sampled token IDs.
            src: The rank that performed the sampling.

        Returns:
            The token ID array available on every rank.
        """
        if not self.is_distributed:
            return token_ids

        import mlx.core as mx

        if self._rank == src:
            return mx.distributed.all_sum(token_ids, group=self._group)
        else:
            return mx.distributed.all_sum(
                mx.zeros_like(token_ids), group=self._group
            )


# ---------------------------------------------------------------------------
# Environment helpers
# ---------------------------------------------------------------------------


def setup_distributed_env(
    backend: str,
    rank: int,
    hostfile: str | None = None,
    coordinator: str | None = None,
) -> None:
    """Set environment variables required by ``mx.distributed.init``.

    For the **ring** backend:
        * ``MLX_HOSTFILE`` — path to the host file.
        * ``MLX_RANK`` — this process's rank.

    For the **jaccl** backend:
        * ``MLX_IBV_DEVICES`` — InfiniBand / RDMA device list (optional).
        * ``MLX_RANK`` — this process's rank.
        * ``MLX_JACCL_COORDINATOR`` — coordinator address
          (e.g. ``"hostname:port"``).
        * ``MLX_METAL_FAST_SYNCH=1`` — required for multi-node JACCL.

    Args:
        backend: ``"ring"``, ``"jaccl"``, or ``"any"``.
        rank: The rank to assign to this process.
        hostfile: Path to the host file (ring backend).
        coordinator: Coordinator address (jaccl backend).
    """
    os.environ["MLX_RANK"] = str(rank)
    logger.info("Set MLX_RANK=%d", rank)

    if backend in ("ring", "any"):
        if hostfile is not None:
            os.environ["MLX_HOSTFILE"] = hostfile
            logger.info("Set MLX_HOSTFILE=%s", hostfile)

    if backend in ("jaccl", "any"):
        if coordinator is not None:
            os.environ["MLX_JACCL_COORDINATOR"] = coordinator
            logger.info("Set MLX_JACCL_COORDINATOR=%s", coordinator)

        # Fast GPU synchronization for multi-node JACCL
        os.environ["MLX_METAL_FAST_SYNCH"] = "1"
        logger.info("Set MLX_METAL_FAST_SYNCH=1")


# ---------------------------------------------------------------------------
# Singleton accessor
# ---------------------------------------------------------------------------

_global_communicator: MLXCommunicator | None = None


def get_communicator(backend: str = "any") -> MLXCommunicator:
    """Get or create the global :class:`MLXCommunicator` singleton.

    Args:
        backend: Passed to :class:`MLXCommunicator` only on first call.

    Returns:
        The shared communicator instance.
    """
    global _global_communicator
    if _global_communicator is None:
        _global_communicator = MLXCommunicator(backend=backend)
    return _global_communicator
