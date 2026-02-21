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
import time as _tp_time
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
    spec_decode_plan: SpecDecodePlan | None = None


@dataclass
class StepResult:
    """Result of a single decode step."""

    step_id: int
    token_ids: dict[str, int]  # request_id -> sampled token


@dataclass
class SpecDecodePlan:
    """Draft proposal plan broadcast from rank 0 to workers for TP spec decode."""

    draft_tokens: dict[str, list[int]]  # request_id → draft token IDs
    max_draft_len: int  # max draft length for padding alignment
    batch_order: list[tuple[str, int]]  # (request_id, batch_idx) order
    batch_y: list[int]  # current batch.y values per position (for input tensor construction)


@dataclass
class SpecDecodeResult:
    """Result of rejection sampling, broadcast from rank 0 to workers."""

    step_id: int  # matches StepPlan.step_id for phase-1/phase-2 sanity check
    accepted_tokens: dict[str, list[int]]  # committed tokens per request
    trim_amounts: list[int]  # KV cache trim per batch position
    new_y: list[int]  # new batch.y values per batch position
    finished_ids: list[str]  # request IDs that completed during spec decode


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

        The result is eagerly evaluated to ensure the all_sum completes
        before any subsequent communication.
        """
        if not self.is_distributed:
            return tensor

        import mlx.core as mx

        if self._rank == src:
            result = mx.distributed.all_sum(tensor, group=self._group)
        else:
            result = mx.distributed.all_sum(
                mx.zeros_like(tensor), group=self._group
            )
        mx.eval(result)
        return result

    def broadcast_object(self, obj: Any, src: int = 0) -> Any:
        """Broadcast an arbitrary picklable object from *src* to all.

        Protocol:
        1. Source rank pickles the object to bytes.
        2. Broadcast the byte-length as an integer.
        3. Convert the bytes to a ``uint8`` array and broadcast.
        4. Broadcast a CRC32 checksum for integrity verification.
        5. Every rank unpickles the result.

        When *src* is the calling rank the original *obj* is returned
        directly (avoids a redundant pickle round-trip on that rank).
        """
        if not self.is_distributed:
            return obj

        import zlib

        import mlx.core as mx

        _bo_t0 = _tp_time.perf_counter()
        _obj_type = type(obj).__name__ if obj is not None else "None"
        logger.debug(
            "[TP-DEBUG] broadcast_object ENTER: rank=%d src=%d obj_type=%s",
            self._rank, src, _obj_type,
        )

        if self._rank == src:
            raw = pickle.dumps(obj)
            data = mx.array(list(raw), dtype=mx.uint8)
            size = data.size
            checksum = zlib.crc32(raw) & 0xFFFFFFFF
            # Broadcast size
            logger.debug(
                "[TP-DEBUG] broadcast_object rank=%d all_sum #1 START (size=%d bytes)",
                self._rank, size,
            )
            mx.eval(mx.distributed.all_sum(mx.array(size), group=self._group))
            logger.debug(
                "[TP-DEBUG] broadcast_object rank=%d all_sum #1 DONE (size) t=%.3f",
                self._rank, _tp_time.perf_counter() - _bo_t0,
            )
            # Broadcast data
            logger.debug(
                "[TP-DEBUG] broadcast_object rank=%d all_sum #2 START (data, %d bytes)",
                self._rank, size,
            )
            mx.eval(mx.distributed.all_sum(data, group=self._group))
            logger.debug(
                "[TP-DEBUG] broadcast_object rank=%d all_sum #2 DONE (data) t=%.3f",
                self._rank, _tp_time.perf_counter() - _bo_t0,
            )
            # Broadcast checksum
            logger.debug(
                "[TP-DEBUG] broadcast_object rank=%d all_sum #3 START (crc=%08x)",
                self._rank, checksum,
            )
            mx.eval(mx.distributed.all_sum(mx.array(checksum, dtype=mx.uint32), group=self._group))
            logger.debug(
                "[TP-DEBUG] broadcast_object rank=%d all_sum #3 DONE (checksum) t=%.3f",
                self._rank, _tp_time.perf_counter() - _bo_t0,
            )
            return obj
        else:
            # Receive size
            logger.debug(
                "[TP-DEBUG] broadcast_object rank=%d all_sum #1 START (recv size)",
                self._rank,
            )
            size = mx.distributed.all_sum(
                mx.array(0), group=self._group
            ).item()
            logger.debug(
                "[TP-DEBUG] broadcast_object rank=%d all_sum #1 DONE (size=%d) t=%.3f",
                self._rank, size, _tp_time.perf_counter() - _bo_t0,
            )
            # Receive data
            logger.debug(
                "[TP-DEBUG] broadcast_object rank=%d all_sum #2 START (recv data, %d bytes)",
                self._rank, size,
            )
            data = mx.distributed.all_sum(
                mx.zeros(size, dtype=mx.uint8), group=self._group
            )
            mx.eval(data)
            logger.debug(
                "[TP-DEBUG] broadcast_object rank=%d all_sum #2 DONE (data) t=%.3f",
                self._rank, _tp_time.perf_counter() - _bo_t0,
            )
            raw = bytes(data.tolist())
            # Receive and verify checksum
            logger.debug(
                "[TP-DEBUG] broadcast_object rank=%d all_sum #3 START (recv checksum)",
                self._rank,
            )
            expected_checksum = mx.distributed.all_sum(
                mx.array(0, dtype=mx.uint32), group=self._group
            ).item()
            logger.debug(
                "[TP-DEBUG] broadcast_object rank=%d all_sum #3 DONE (crc=%08x) t=%.3f",
                self._rank, expected_checksum, _tp_time.perf_counter() - _bo_t0,
            )
            actual_checksum = zlib.crc32(raw) & 0xFFFFFFFF
            if actual_checksum != expected_checksum:
                raise RuntimeError(
                    f"broadcast_object integrity check failed: "
                    f"expected CRC32={expected_checksum:#010x}, "
                    f"got {actual_checksum:#010x} (size={size})"
                )
            logger.debug(
                "[TP-DEBUG] broadcast_object rank=%d checksum OK, total=%.3fs",
                self._rank, _tp_time.perf_counter() - _bo_t0,
            )
            return pickle.loads(raw)

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
        _bsp_t0 = _tp_time.perf_counter()
        _plan_desc = "None"
        if plan is not None:
            _plan_desc = (
                f"step={plan.step_id} ins={len(plan.inserts)} "
                f"rem={len(plan.removes)} fp={plan.fingerprint}"
            )
        logger.debug(
            "[TP-DEBUG] broadcast_step_plan ENTER: rank=%d plan=(%s)",
            self._rank, _plan_desc,
        )
        self.broadcast_object(plan, src=0)
        logger.debug(
            "[TP-DEBUG] broadcast_step_plan EXIT: rank=%d elapsed=%.3fs",
            self._rank, _tp_time.perf_counter() - _bsp_t0,
        )

    def receive_step_plan(self) -> StepPlan:
        """Receive a :class:`StepPlan` on a non-rank-0 worker.

        Internally calls :meth:`broadcast_object` with ``src=0``.
        """
        _rsp_t0 = _tp_time.perf_counter()
        logger.debug(
            "[TP-DEBUG] receive_step_plan ENTER: rank=%d",
            self._rank,
        )
        result = self.broadcast_object(None, src=0)
        _plan_desc = "None"
        if result is not None:
            _plan_desc = (
                f"step={result.step_id} ins={len(result.inserts)} "
                f"rem={len(result.removes)} fp={result.fingerprint}"
            )
        logger.debug(
            "[TP-DEBUG] receive_step_plan EXIT: rank=%d plan=(%s) elapsed=%.3fs",
            self._rank, _plan_desc, _tp_time.perf_counter() - _rsp_t0,
        )
        return result

    def broadcast_tokens(self, token_ids, src: int = 0):
        """Broadcast sampled token IDs from *src* to every rank.

        Convenience wrapper around :meth:`broadcast_tensor`.
        """
        return self.broadcast_tensor(token_ids, src=src)

    def broadcast_spec_decode_result(self, result: SpecDecodeResult) -> None:
        """Broadcast a SpecDecodeResult from rank 0 to all workers.

        Called after rank 0 performs rejection sampling. Workers need
        the result to update their KV cache and batch state.
        """
        if not self.is_distributed:
            return
        if self._rank != 0:
            logger.warning(
                "broadcast_spec_decode_result called on rank %d; "
                "use receive_spec_decode_result on non-zero ranks",
                self._rank,
            )
        self.broadcast_object(result, src=0)

    def receive_spec_decode_result(self) -> SpecDecodeResult:
        """Receive a SpecDecodeResult on a non-rank-0 worker."""
        return self.broadcast_object(None, src=0)


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
