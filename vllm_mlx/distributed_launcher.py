# SPDX-License-Identifier: Apache-2.0
"""
Distributed process launcher for vllm-mlx.

Enables running vllm-mlx across multiple Mac nodes connected via
Thunderbolt 5 RDMA (JACCL) or TCP (ring). This module wraps the
``mlx.launch`` CLI tool and provides Python APIs for:

- Launching distributed processes (``launch_distributed``)
- Generating hostfiles for multi-node setups (``generate_hostfile``,
  ``generate_jaccl_hostfile``)
- Serving as the entry point for each spawned rank process
  (``distributed_main``, ``worker_loop``)

Usage as launcher (no MLX_RANK set)::

    python -m vllm_mlx.distributed_launcher \\
        --backend jaccl --hostfile hosts.json -- --model my-model

Usage as spawned worker (MLX_RANK set by mlx.launch)::

    # Automatically called by mlx.launch; not invoked directly.
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import signal
import subprocess
import sys
import tempfile
from typing import Any

logger = logging.getLogger(__name__)

# Worker-side prefix cache (initialized in worker_loop for rank >= 1)
_worker_cache = None

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

_SUPPORTED_BACKENDS = ("jaccl", "ring", "any")

_JACCL_DEFAULT_ENVS = ["MLX_METAL_FAST_SYNCH=1"]

# Pre-loaded model/tokenizer for distributed mode (set in distributed_main)
_preloaded_model: tuple | None = None


# ---------------------------------------------------------------------------
# Hostfile generation
# ---------------------------------------------------------------------------


def generate_hostfile(
    hosts: list[dict[str, Any]],
    backend: str = "jaccl",
    output_path: str | None = None,
) -> str:
    """Generate a hostfile JSON for ``mlx.launch``.

    Args:
        hosts: List of host configurations.
            For ring: ``[{"ssh": "hostname", "ips": ["ip1"]}, ...]``
            For jaccl: ``[{"ssh": "hostname", "ips": ["ip1"],
            "rdma": [null, "rdma_en2"]}, ...]``
        backend: Backend type (``"jaccl"``, ``"ring"``, or ``"any"``).
        output_path: Where to save the JSON file.  If ``None``, a
            temporary file is created and its path returned.

    Returns:
        Path to the generated hostfile.
    """
    if backend not in _SUPPORTED_BACKENDS:
        raise ValueError(
            f"Unsupported backend {backend!r}. "
            f"Choose from {_SUPPORTED_BACKENDS}"
        )

    if backend == "jaccl":
        hostfile_data: dict[str, Any] = {
            "backend": "jaccl",
            "envs": list(_JACCL_DEFAULT_ENVS),
            "hosts": hosts,
        }
    elif backend == "ring":
        hostfile_data = {
            "backend": "ring",
            "envs": [],
            "hosts": hosts,
        }
    else:
        # "any" -- default to ring format (mlx parser requires
        # "backend" and "envs" keys when the hostfile is a dict).
        hostfile_data = {
            "backend": "ring",
            "envs": [],
            "hosts": hosts,
        }

    return _write_hostfile(hostfile_data, output_path)


def generate_jaccl_hostfile(
    hosts: list[str],
    rdma_devices: list[list[str | None]] | None = None,
) -> str:
    """Generate a JACCL-specific hostfile.

    The JACCL hostfile format for ``mlx.launch`` is::

        {
            "backend": "jaccl",
            "envs": ["MLX_METAL_FAST_SYNCH=1"],
            "hosts": [
                {"ssh": "host1", "ips": ["10.254.0.1"],
                 "rdma": [null, "rdma_en2"]},
                {"ssh": "host2", "ips": ["10.254.0.2"],
                 "rdma": ["rdma_en2", null]}
            ]
        }

    Args:
        hosts: List of hostnames or IPs (used for both SSH and IP
            fields).
        rdma_devices: Per-host list of RDMA device names.  Each
            element is a list whose length equals the number of
            peer connections (``len(hosts)``); use ``None`` for
            "self" entries.  If ``None``, auto-detection is
            attempted via ``_detect_rdma_devices``.

    Returns:
        Path to the generated temporary hostfile.
    """
    n_hosts = len(hosts)
    if n_hosts < 2:
        raise ValueError("JACCL requires at least 2 hosts")

    if rdma_devices is None:
        rdma_devices = _detect_rdma_devices(n_hosts)

    if len(rdma_devices) != n_hosts:
        raise ValueError(
            f"rdma_devices length ({len(rdma_devices)}) must match "
            f"hosts length ({n_hosts})"
        )

    host_entries = []
    for i, hostname in enumerate(hosts):
        entry: dict[str, Any] = {
            "ssh": hostname,
            "ips": [hostname],
            "rdma": rdma_devices[i],
        }
        host_entries.append(entry)

    return generate_hostfile(host_entries, backend="jaccl")


def _detect_rdma_devices(n_hosts: int) -> list[list[str | None]]:
    """Attempt to detect Thunderbolt 5 RDMA device names.

    This is a best-effort heuristic.  On macOS, Thunderbolt network
    interfaces typically appear as ``en*`` with corresponding RDMA
    devices named ``rdma_en*``.  For now, we return a sensible
    default pattern where each host has ``null`` for itself and
    ``"rdma_en2"`` for each peer.

    In production, users should provide explicit RDMA device mappings
    or use ``mlx.distributed_config`` to discover them.

    Args:
        n_hosts: Number of hosts in the cluster.

    Returns:
        A list of RDMA device lists, one per host.
    """
    logger.warning(
        "RDMA device auto-detection is a placeholder. "
        "For production use, specify rdma_devices explicitly or "
        "run 'mlx.distributed_config' to discover devices."
    )

    # Default pattern: null for self, "rdma_en2" for every peer.
    devices = []
    for i in range(n_hosts):
        host_rdma: list[str | None] = []
        for j in range(n_hosts):
            if i == j:
                host_rdma.append(None)
            else:
                host_rdma.append("rdma_en2")
        devices.append(host_rdma)
    return devices


def _write_hostfile(data: dict[str, Any], output_path: str | None) -> str:
    """Serialize *data* to a JSON file and return its path."""
    if output_path is None:
        fd, output_path = tempfile.mkstemp(
            suffix=".json", prefix="vllm_mlx_hostfile_"
        )
        with os.fdopen(fd, "w") as f:
            json.dump(data, f, indent=2)
    else:
        with open(output_path, "w") as f:
            json.dump(data, f, indent=2)

    logger.info(f"Hostfile written to {output_path}")
    return output_path


# ---------------------------------------------------------------------------
# Hostfile helpers for simple host lists
# ---------------------------------------------------------------------------


def _hosts_to_hostfile(
    hosts: list[str],
    backend: str,
) -> str:
    """Convert a simple list of host IPs to a hostfile path."""
    if backend == "jaccl":
        return generate_jaccl_hostfile(hosts)

    # Ring or any: just use the IPs
    host_entries = [{"ssh": h, "ips": [h]} for h in hosts]
    return generate_hostfile(host_entries, backend=backend)


# ---------------------------------------------------------------------------
# Main launcher
# ---------------------------------------------------------------------------


def launch_distributed(
    script_or_module: str,
    args: list[str] | None = None,
    *,
    backend: str = "jaccl",
    num_ranks: int | None = None,
    hostfile: str | None = None,
    hosts: list[str] | None = None,
    envs: dict[str, str] | None = None,
) -> None:
    """Launch distributed vllm-mlx processes using ``mlx.launch``.

    This is a Python wrapper around the ``mlx.launch`` CLI tool.

    Args:
        script_or_module: Path to the Python script to run.
        args: Additional CLI arguments to pass to the script.
        backend: ``"jaccl"``, ``"ring"``, or ``"any"``.
        num_ranks: Number of ranks (processes).  If ``None``,
            inferred from the hostfile.
        hostfile: Path to hostfile JSON (for multi-node).
        hosts: List of host IPs (alternative to *hostfile*, for
            simple cases).
        envs: Additional environment variables to set in each
            spawned process.
    """
    if backend not in _SUPPORTED_BACKENDS:
        raise ValueError(
            f"Unsupported backend {backend!r}. "
            f"Choose from {_SUPPORTED_BACKENDS}"
        )

    # Build environment
    env = os.environ.copy()

    # Always set fast synch for JACCL
    if backend == "jaccl":
        env["MLX_METAL_FAST_SYNCH"] = "1"

    # Propagate backend choice to spawned processes
    env["MLX_DISTRIBUTED_BACKEND"] = backend

    # Merge user-specified environment variables
    if envs:
        env.update(envs)

    # Resolve hostfile: explicit > hosts list > num_ranks only
    if hostfile is None and hosts is not None:
        hostfile = _hosts_to_hostfile(hosts, backend)

    # Build the mlx.launch command
    # mlx.launch is a console script pointing to mlx._distributed_utils.launch
    cmd: list[str] = [sys.executable, "-m", "mlx._distributed_utils.launch"]

    # Map "any" to a concrete backend; mlx.launch only accepts
    # ring/mpi/nccl/jaccl/jaccl-ring.
    launch_backend = backend if backend != "any" else "ring"

    # JACCL requires hostfile with RDMA device mapping
    if launch_backend == "jaccl" and hostfile is None and hosts is None:
        raise ValueError(
            "JACCL backend requires --hostfile or --hosts with RDMA device mapping. "
            "Use --backend ring for local multi-process testing with -n flag."
        )

    if hostfile is not None:
        cmd.extend(["--backend", launch_backend, "--hostfile", hostfile])
    elif num_ranks is not None:
        cmd.extend(["-n", str(num_ranks), "--backend", launch_backend])
    else:
        raise ValueError(
            "Must specify at least one of: num_ranks, hostfile, or hosts"
        )

    # Append the target script/module and its arguments
    cmd.append(script_or_module)
    if args:
        cmd.extend(args)

    logger.info(f"Launching distributed processes: {' '.join(cmd)}")

    # Run mlx.launch as a subprocess, forwarding signals
    process = subprocess.Popen(cmd, env=env)

    def _signal_handler(signum: int, _frame: Any) -> None:
        """Forward termination signals to the child process."""
        logger.info(f"Received signal {signum}, forwarding to child process")
        process.send_signal(signum)

    original_sigterm = signal.getsignal(signal.SIGTERM)
    original_sigint = signal.getsignal(signal.SIGINT)

    signal.signal(signal.SIGTERM, _signal_handler)
    signal.signal(signal.SIGINT, _signal_handler)

    try:
        returncode = process.wait()
    finally:
        signal.signal(signal.SIGTERM, original_sigterm)
        signal.signal(signal.SIGINT, original_sigint)

    if returncode != 0:
        logger.error(f"mlx.launch exited with code {returncode}")
        sys.exit(returncode)

    logger.info("Distributed launch completed successfully")


# ---------------------------------------------------------------------------
# Worker loop (rank >= 1)
# ---------------------------------------------------------------------------


def _worker_spec_decode_step(
    spec_plan,
    batch_generator,
    model,
    communicator,
    request_id_to_uid: dict,
    uid_to_request_id: dict,
) -> set[str]:
    """Execute a speculative decoding step on a worker rank.

    Workers mirror rank 0's spec decode path:
    1. Build the same input tensor from SpecDecodePlan
    2. Run model forward (TP all_sum synchronizes with rank 0)
    3. Receive SpecDecodeResult from rank 0
    4. Apply KV cache trim and batch state updates

    Args:
        spec_plan: SpecDecodePlan from StepPlan
        batch_generator: The worker's BatchGenerator
        model: The sharded model
        communicator: MLXCommunicator
        request_id_to_uid: request_id -> batch uid mapping
        uid_to_request_id: batch uid -> request_id mapping

    Returns:
        Set of finished request IDs to remove from tracking
    """
    import mlx.core as mx

    batch = batch_generator.active_batch
    if batch is None:
        raise RuntimeError(
            f"[Rank {communicator.rank}] spec_decode_plan received but no active batch. "
            "Worker batch state is desynchronized from rank 0. "
            "Cannot participate in TP forward — this would cause a deadlock."
        )

    # 1. Build input tensor from SpecDecodePlan (must match rank 0 exactly)
    batch_y = spec_plan.batch_y
    max_draft_len = spec_plan.max_draft_len

    input_rows = []
    for rid, batch_idx in spec_plan.batch_order:
        y_token = batch_y[batch_idx]
        drafts = spec_plan.draft_tokens.get(rid, [])
        padded_drafts = list(drafts) + [0] * (max_draft_len - len(drafts))
        row = [y_token] + padded_drafts
        input_rows.append(row)

    input_tokens = mx.array(input_rows, dtype=mx.int32)  # (B_spec, max_k+1)

    # 2. Run model forward (TP all_sum happens inside sharded model)
    rank = communicator.rank
    logger.info("[SD-W] PRE-FORWARD input_shape=%s cache_idx=%s",
                input_tokens.shape,
                batch.cache[0]._idx if batch.cache else "N/A")
    logits = model(input_tokens, cache=batch.cache)
    # Eager eval is CRITICAL: completes the TP all_sum synchronization
    # with rank 0 before we wait for SpecDecodeResult.
    mx.eval(logits)
    logger.info("[SD-W] POST-FORWARD logits_shape=%s", logits.shape)

    # 3. Receive SpecDecodeResult from rank 0
    logger.info("[SD-W] PRE-RECEIVE waiting for SpecDecodeResult")
    spec_result = communicator.receive_spec_decode_result()
    logger.info("[SD-W] POST-RECEIVE trim=%s", spec_result.trim_amounts)

    # 4. Apply KV cache trim
    if spec_result.trim_amounts and any(t > 0 for t in spec_result.trim_amounts):
        from vllm_mlx.spec_decode.cache_utils import batch_variable_trim
        trim_array = mx.array(spec_result.trim_amounts, dtype=mx.int32)
        batch_variable_trim(batch.cache, trim_array)
        # Materialize trim results before further cache operations
        if batch.cache:
            mx.eval(batch.cache[0].offset, batch.cache[0].left_padding)

    # Force eval to prevent lazy graph accumulation in worker
    if batch.cache:
        mx.eval(batch.cache[0].offset, batch.cache[0].left_padding)

    # 5. Update batch state
    # Set new batch.y
    batch.y = mx.array(spec_result.new_y, dtype=mx.int32)

    # Update tokens and num_tokens using canonical committed tokens (post-clipping)
    for rid, emitted in spec_result.accepted_tokens.items():
        if rid not in request_id_to_uid:
            continue
        uid = request_id_to_uid[rid]
        batch_idx = None
        for idx, b_uid in enumerate(batch.uids):
            if b_uid == uid:
                batch_idx = idx
                break
        if batch_idx is None:
            continue

        if emitted:
            batch.tokens[batch_idx] = mx.concatenate(
                (batch.tokens[batch_idx], mx.array(emitted))
            )
        batch.num_tokens[batch_idx] += len(emitted)

    # Materialize all updated tensors to prevent lazy graph accumulation
    mx.eval(batch.y, *batch.tokens)

    # Materialize cache tensors to prevent lazy graph accumulation.
    # trim_per_sequence and filter create lazy chains on offset,
    # left_padding, keys, and values. Without explicit eval, these
    # chains grow with every step and eventually cause Metal/RDMA
    # hangs when the graph becomes too deep.
    if batch.cache:
        cache_tensors = []
        for c in batch.cache:
            cache_tensors.extend([c.keys, c.values, c.offset, c.left_padding])
        mx.eval(*cache_tensors)

    # Final eval barrier before next spec decode iteration
    if batch.cache:
        mx.eval(batch.cache[0].offset, batch.cache[0].left_padding)

    # 6. Return finished IDs for removal
    return set(spec_result.finished_ids)


def worker_loop(
    communicator: Any,
    model_name: str,
    model: Any = None,
    tokenizer: Any = None,
    tokenizer_config: dict | None = None,
) -> None:
    """Worker loop for non-rank-0 processes.

    Receives :class:`StepPlan` messages from rank 0 and executes model
    forward passes with a local :class:`BatchGenerator` using the
    sharded model.

    Args:
        communicator: An ``MLXCommunicator`` instance (from
            ``vllm_mlx.distributed``).
        model_name: HuggingFace model name or local path to load.
        tokenizer_config: Optional tokenizer configuration.
    """
    import threading

    import mlx.core as mx
    from mlx_lm.utils import sharded_load

    rank = communicator.rank
    world_size = communicator.world_size

    logger.info(f"[Rank {rank}] Worker starting (world_size={world_size})")

    # Step 1: Load sharded model
    if model is not None and tokenizer is not None:
        logger.info(f"[Rank {rank}] Using pre-loaded model")
    else:
        # Legacy path: load model here (for non-distributed_main callers)
        logger.info(f"[Rank {rank}] Loading sharded model: {model_name}")
        model, tokenizer = sharded_load(
            model_name,
            pipeline_group=None,
            tensor_group=communicator.group,
        )
        communicator.barrier()
    logger.info(f"[Rank {rank}] Model loaded successfully")

    # Step 2: Create BatchGenerator with sharded model
    from mlx_lm.sample_utils import make_sampler

    # Default sampler (will be overridden per-request)
    sampler = make_sampler(temp=0.0)

    # Import BatchGenerator from mlx-lm
    from mlx_lm.generate import BatchGenerator

    # Build stop_tokens matching Rank 0's _get_stop_tokens() logic
    # to prevent asymmetric batch.filter() → all_sum shape mismatch → deadlock
    # Extract actual tokenizer from processor wrapper (mirrors scheduler._get_actual_tokenizer)
    _actual_tokenizer = tokenizer
    if hasattr(tokenizer, "tokenizer"):
        _actual_tokenizer = tokenizer.tokenizer
    stop_tokens = set()
    for tok in [tokenizer, _actual_tokenizer]:
        if tok is None:
            continue
        if hasattr(tok, "eos_token_id") and tok.eos_token_id is not None:
            if isinstance(tok.eos_token_id, list):
                stop_tokens.update(tok.eos_token_id)
            else:
                stop_tokens.add(tok.eos_token_id)
        if hasattr(tok, "eos_token_ids") and tok.eos_token_ids is not None:
            if isinstance(tok.eos_token_ids, (list, set, tuple)):
                stop_tokens.update(tok.eos_token_ids)
            else:
                stop_tokens.add(tok.eos_token_ids)

    batch_generator = BatchGenerator(
        model=model,
        max_tokens=4096,
        stop_tokens=stop_tokens,
        sampler=sampler,
    )

    # Initialize per-rank prefix cache for TP cache validation
    from vllm_mlx.memory_cache import MemoryAwarePrefixCache, MemoryCacheConfig
    global _worker_cache
    worker_cache_config = MemoryCacheConfig(
        max_memory_mb=None,  # auto-detect
        max_memory_percent=0.20,
        max_entries=1000,
        tp_world_size=world_size,
        tp_rank=rank,
    )
    _worker_cache = MemoryAwarePrefixCache(model, worker_cache_config)
    logger.info(f"[Rank {rank}] Worker prefix cache initialized: "
                f"limit={_worker_cache.memory_limit_mb:.1f}MB")

    # uid -> full prompt tokens, populated at insert time, consumed by hook
    _worker_cache_keys: dict[int, list[int]] = {}

    # Install _process_prompts hook to store KV cache at prompt-only state.
    # This mirrors rank 0's _install_chunked_prefill hook in scheduler.py.
    # The hook fires INSIDE next() AFTER _process_prompts() returns and
    # BEFORE _generation_step(), so KV cache has exactly N prompt entries
    # and num_tokens == 0.
    _orig_process_prompts = batch_generator._process_prompts

    def _worker_cache_save_hook(prompts, _orig=_orig_process_prompts):
        batch = _orig(prompts)
        for e, uid_val in enumerate(batch.uids):
            if batch.num_tokens[e] == 0 and uid_val in _worker_cache_keys:
                try:
                    _extracted = batch.extract_cache(e)
                    _ck = _worker_cache_keys.pop(uid_val)
                    _worker_cache.store(_ck, _extracted)
                    logger.info(
                        "[TP-CACHE] worker(%d) stored cache via hook: "
                        "uid=%d tokens=%d",
                        rank, uid_val, len(_ck),
                    )
                except Exception as _e:
                    logger.info(
                        "[TP-CACHE] worker(%d) cache store hook FAILED: "
                        "uid=%d err=%s",
                        rank, uid_val, _e,
                    )
        return batch

    batch_generator._process_prompts = _worker_cache_save_hook

    # Monkey-patch _step() to add distributed sampling synchronization.
    # All ranks must use rank 0's sampled token to prevent KV cache divergence.
    if communicator.is_distributed:
        _orig_step = batch_generator._step

        def _synced_step(input_tokens, prompt_cache, samplers, logits_processors, tokens):
            sampled, logprobs = _orig_step(
                input_tokens, prompt_cache, samplers, logits_processors, tokens
            )
            # Force model forward eval (including TP all_sum inside MoE layers).
            # Both ranks must complete the model's internal collectives BEFORE
            # we break the dependency chain with zeros_like.
            mx.eval(sampled)
            # Now sync sampling: zero non-rank-0, all_sum to broadcast rank 0's token.
            if communicator.rank > 0:
                sampled = mx.zeros_like(sampled)
            sampled = mx.distributed.all_sum(sampled, group=communicator.group)
            mx.eval(sampled)
            return sampled, logprobs

        batch_generator._step = _synced_step

    # Track request_id -> uid mapping (mirrors rank 0's scheduler)
    request_id_to_uid: dict[str, int] = {}
    uid_to_request_id: dict[int, str] = {}

    logger.info(f"[Rank {rank}] BatchGenerator created, entering StepPlan loop")

    # Ready barrier: wait for rank 0 to finish server initialization.
    # Rank 0 loads the draft model, starts uvicorn, and creates the engine
    # loop before signaling readiness. Without this wait, we'd call
    # receive_step_plan() while rank 0 isn't broadcasting, causing JACCL timeout.
    communicator.barrier()
    logger.info(f"[Rank {rank}] Ready barrier passed — rank 0 scheduler loop starting")

    # Shutdown handling
    shutdown_event = threading.Event()

    def _shutdown_handler(signum: int, _frame: Any) -> None:
        logger.info(f"[Rank {rank}] Received signal {signum}, shutting down")
        shutdown_event.set()

    original_sigterm = signal.getsignal(signal.SIGTERM)
    original_sigint = signal.getsignal(signal.SIGINT)
    signal.signal(signal.SIGTERM, _shutdown_handler)
    signal.signal(signal.SIGINT, _shutdown_handler)

    try:
        import time as _tp_time

        # Step 3: StepPlan processing loop
        while not shutdown_event.is_set():
            try:
                _tp_loop_t0 = _tp_time.perf_counter()

                # Receive StepPlan from rank 0
                plan = communicator.receive_step_plan()
                _tp_recv_t = _tp_time.perf_counter() - _tp_loop_t0

                if plan is None:
                    # Shutdown signal
                    logger.info(f"[Rank {rank}] Received None plan, shutting down")
                    break

                logger.debug(
                    "[TP-DEBUG] worker(%d) RECEIVED step_plan: "
                    "step_id=%d inserts=%d removes=%d fp=%s "
                    "insert_ids=[%s] remove_ids=[%s] "
                    "spec=%s recv_time=%.3fs",
                    rank, plan.step_id,
                    len(plan.inserts), len(plan.removes),
                    plan.fingerprint,
                    ",".join(op.request_id[:12] for op in plan.inserts),
                    ",".join(r[:12] for r in plan.removes),
                    "yes" if plan.spec_decode_plan else "no",
                    _tp_recv_t,
                )

                # Apply removes first
                for request_id in plan.removes:
                    if request_id in request_id_to_uid:
                        uid = request_id_to_uid[request_id]
                        logger.debug(
                            "[TP-DEBUG] worker(%d) REMOVE: req=%s uid=%d",
                            rank, request_id[:12], uid,
                        )
                        batch_generator.remove([uid])
                        if batch_generator.active_batch is not None:
                            from vllm_mlx.spec_decode.cache_utils import fixup_cache_after_filter
                            fixup_cache_after_filter(batch_generator.active_batch.cache)
                        del request_id_to_uid[request_id]
                        del uid_to_request_id[uid]

                # Apply inserts with per-rank cache validation
                if plan.inserts:
                    import mlx.core as mx

                    # Phase 1: Probe local cache for each insert
                    cache_masks = []
                    local_caches = []    # local KV cache per insert (or None)
                    local_remainings = []  # local remaining tokens per insert (or None)

                    for insert_op in plan.inserts:
                        if insert_op.use_cache and insert_op.full_tokens and _worker_cache is not None:
                            local_cache, local_remaining = _worker_cache.fetch(insert_op.full_tokens)
                            local_cached = (
                                len(insert_op.full_tokens) - len(local_remaining)
                                if local_cache else 0
                            )
                            # Strict: must match rank 0's cached count exactly
                            cache_ok = 1 if local_cached == insert_op.cached_tokens else 0
                            local_caches.append(local_cache)
                            local_remainings.append(local_remaining)
                        elif insert_op.use_cache:
                            # use_cache=True but missing full_tokens or worker cache
                            # → force fallback (vote 0)
                            cache_ok = 0
                            local_caches.append(None)
                            local_remainings.append(None)
                        else:
                            cache_ok = 1  # No cache needed, always OK
                            local_caches.append(None)
                            local_remainings.append(None)
                        cache_masks.append(cache_ok)

                    # Phase 2: Batched all_sum validation (single collective op)
                    any_cache_used = any(op.use_cache for op in plan.inserts)
                    if any_cache_used:
                        local_mask = mx.array(cache_masks, dtype=mx.int32)
                        global_mask = mx.distributed.all_sum(local_mask, group=communicator.group)
                        mx.eval(global_mask)
                        global_mask_list = global_mask.tolist()
                    else:
                        global_mask_list = cache_masks  # All 1s, no collective needed

                    # Phase 3: Insert with validated cache decisions
                    _inserted_tokens_map = {}  # request_id -> tokens actually inserted
                    for i, insert_op in enumerate(plan.inserts):
                        all_agree = global_mask_list[i] == world_size
                        use_local_cache = (
                            insert_op.use_cache and all_agree and local_caches[i] is not None
                        )

                        if use_local_cache:
                            # Use rank 0's authoritative token suffix for consistency
                            tokens_for_insert = insert_op.tokens
                            cache_for_insert = [local_caches[i]]
                        else:
                            # Fallback: full tokens, no cache
                            tokens_for_insert = insert_op.full_tokens if insert_op.full_tokens else insert_op.tokens
                            cache_for_insert = None

                            if insert_op.use_cache:
                                logger.info(
                                    "[TP-CACHE] worker(%d) cache FALLBACK: "
                                    "req=%s global_mask=%d (need %d)",
                                    rank, insert_op.request_id[:12],
                                    global_mask_list[i], world_size,
                                )

                        logger.debug(
                            "[TP-DEBUG] worker(%d) BEFORE insert: "
                            "req=%s tokens=%d max_tokens=%d use_cache=%s all_agree=%s",
                            rank, insert_op.request_id[:12],
                            len(tokens_for_insert), insert_op.max_tokens,
                            use_local_cache, all_agree,
                        )
                        uids = batch_generator.insert(
                            [tokens_for_insert],
                            max_tokens=[insert_op.max_tokens],
                            caches=cache_for_insert,
                        )
                        if uids:
                            uid = uids[0]
                            request_id_to_uid[insert_op.request_id] = uid
                            uid_to_request_id[uid] = insert_op.request_id
                            _inserted_tokens_map[insert_op.request_id] = tokens_for_insert

                            # Record cache key for _process_prompts hook
                            if _worker_cache is not None:
                                if insert_op.full_tokens:
                                    _ck = insert_op.full_tokens
                                elif not insert_op.use_cache:
                                    _ck = tokens_for_insert
                                else:
                                    _ck = None
                                if _ck:
                                    _worker_cache_keys[uid] = _ck

                            logger.debug(
                                "[TP-DEBUG] worker(%d) AFTER insert: "
                                "req=%s uid=%d success=True cached=%s",
                                rank, insert_op.request_id[:12], uid, use_local_cache,
                            )
                        else:
                            logger.debug(
                                "[TP-DEBUG] worker(%d) AFTER insert: "
                                "req=%s uids=None success=False",
                                rank, insert_op.request_id[:12],
                            )

                # Verify fingerprint for batch sync detection
                if plan.fingerprint:
                    import hashlib

                    local_running_ids = sorted(request_id_to_uid.keys())
                    local_batch_size = (
                        len(batch_generator.active_batch.uids)
                        if batch_generator.active_batch
                        else 0
                    )
                    local_fp_data = (
                        ",".join(local_running_ids)
                        + f"|step={plan.step_id}|batch={local_batch_size}"
                    )
                    local_fingerprint = hashlib.md5(
                        local_fp_data.encode()
                    ).hexdigest()[:16]
                    logger.debug(
                        "[TP-DEBUG] worker(%d) fingerprint check: "
                        "local=%s remote=%s match=%s "
                        "local_reqs=%d batch_size=%d "
                        "local_ids=[%s]",
                        rank, local_fingerprint, plan.fingerprint,
                        local_fingerprint == plan.fingerprint,
                        len(local_running_ids), local_batch_size,
                        ",".join(rid[:12] for rid in local_running_ids),
                    )
                    if local_fingerprint != plan.fingerprint:
                        logger.error(
                            f"[Rank {rank}] BATCH DESYNC DETECTED! "
                            f"local={local_fingerprint} rank0={plan.fingerprint} "
                            f"local_reqs={len(local_running_ids)} batch_size={local_batch_size}"
                        )
                        raise RuntimeError(
                            f"[Rank {rank}] Batch desync at step {plan.step_id}. "
                            f"local_fp={local_fingerprint} rank0_fp={plan.fingerprint} "
                            f"Terminating to prevent deadlock."
                        )
                # Check if this step is a speculative decoding step
                if plan.spec_decode_plan is not None:
                    # Spec decode path: build input tensor + model forward + receive result
                    finished_spec_ids = _worker_spec_decode_step(
                        spec_plan=plan.spec_decode_plan,
                        batch_generator=batch_generator,
                        model=model,
                        communicator=communicator,
                        request_id_to_uid=request_id_to_uid,
                        uid_to_request_id=uid_to_request_id,
                    )
                    # Remove finished requests
                    for fid in finished_spec_ids:
                        if fid in request_id_to_uid:
                            uid = request_id_to_uid[fid]
                            batch_generator.remove([uid])
                            del request_id_to_uid[fid]
                            del uid_to_request_id[uid]
                    # Fix stale _idx after filter and evaluate cache metadata
                    if finished_spec_ids and batch_generator.active_batch is not None:
                        from vllm_mlx.spec_decode.cache_utils import fixup_cache_after_filter
                        fixup_cache_after_filter(batch_generator.active_batch.cache)
                else:
                    # Normal decode path
                    # Execute forward pass (model computation with all_sum).
                    # This must happen on all ranks simultaneously.
                    if plan.__dict__.get("should_step", bool(request_id_to_uid)):  # Sync with rank 0's next() decision
                        _w_active_uids = (
                            list(batch_generator.active_batch.uids)
                            if batch_generator.active_batch else []
                        )
                        _cache_info = ""
                        if batch_generator.active_batch and batch_generator.active_batch.cache:
                            try:
                                _c0 = batch_generator.active_batch.cache[0]
                                _cache_info = f"cache_idx={_c0._idx} offset={_c0.offset.tolist()} lpad={_c0.left_padding.tolist()}"
                            except Exception:
                                _cache_info = "unavailable"
                        logger.debug(
                            "[TP-DEBUG] worker(%d) BEFORE next(): "
                            "step_id=%d batch_size=%d active_uids=%s "
                            "req_map=%s %s t=%.3f",
                            rank, plan.step_id,
                            len(request_id_to_uid), _w_active_uids,
                            {k[:12]: v for k, v in request_id_to_uid.items()},
                            _cache_info,
                            _tp_time.perf_counter() - _tp_loop_t0,
                        )
                        _batch_size_before = len(batch_generator.active_batch.y) if batch_generator.active_batch else 0
                        _responses = batch_generator.next()
                        _cache_info_after = ""
                        if batch_generator.active_batch and batch_generator.active_batch.cache:
                            try:
                                _c0 = batch_generator.active_batch.cache[0]
                                _cache_info_after = f"cache_idx={_c0._idx} offset={_c0.offset.tolist()} lpad={_c0.left_padding.tolist()}"
                            except Exception:
                                _cache_info_after = "unavailable"
                        logger.debug(
                            "[TP-DEBUG] worker(%d) AFTER next(): "
                            "step_id=%d responses=%d %s t=%.3f",
                            rank, plan.step_id,
                            len(_responses) if _responses else 0,
                            _cache_info_after,
                            _tp_time.perf_counter() - _tp_loop_t0,
                        )

                        # Force lazy all_sum computation to complete.
                        # batch_generator.next() with dist_group triggers all_sum
                        # lazily; mx.eval materializes the result.
                        if batch_generator.active_batch is not None:
                            # Only run cache fixup when batch.filter() removed a request.
                            # fixup_cache_after_filter iterates all layers (expensive for
                            # deep models like Kimi K2.5).  mx.eval(batch.y) is also
                            # unnecessary per-token since _synced_step already evals.
                            _batch_size_after = len(batch_generator.active_batch.y)
                            if _batch_size_after < _batch_size_before:
                                from vllm_mlx.spec_decode.cache_utils import fixup_cache_after_filter
                                fixup_cache_after_filter(batch_generator.active_batch.cache)
                                logger.debug(
                                    "[TP-DEBUG] worker(%d) fixup_cache_after_filter: "
                                    "batch %d -> %d step_id=%d",
                                    rank, _batch_size_before, _batch_size_after, plan.step_id,
                                )

            except Exception as e:
                if shutdown_event.is_set():
                    break
                logger.error(
                    f"[Rank {rank}] Fatal error in StepPlan loop: {e}",
                    exc_info=True,
                )
                # Worker cannot continue -- any further broadcast will
                # deadlock.  The only safe action is to exit and let
                # rank 0's next all_sum operation fail with a
                # timeout/error.
                break

    finally:
        signal.signal(signal.SIGTERM, original_sigterm)
        signal.signal(signal.SIGINT, original_sigint)
        logger.info(f"[Rank {rank}] Worker loop exited")


# ---------------------------------------------------------------------------
# Distributed server entry point
# ---------------------------------------------------------------------------


def distributed_main(server_args: list[str] | None = None) -> None:
    """Entry point for distributed vllm-mlx server.

    This function is called by each rank process spawned by
    ``mlx.launch``.

    - **Rank 0**: Runs the full API server (FastAPI + EngineCore +
      Scheduler).
    - **Rank N>=1**: Runs ``worker_loop()`` which receives
      ``StepPlan`` messages and executes model forward passes.

    The function:

    1. Initializes ``mx.distributed``
    2. Creates ``MLXCommunicator``
    3. If rank == 0: imports and runs the server ``main()`` with
       distributed mode
    4. If rank > 0: enters ``worker_loop()``

    Args:
        server_args: CLI arguments to pass to the server (rank 0
            only).  If ``None``, reads from ``sys.argv``.
    """
    import mlx.core as mx

    # Determine backend from environment or default
    backend = os.environ.get("MLX_DISTRIBUTED_BACKEND", "any")

    # Initialize the distributed group
    group = mx.distributed.init(backend=backend, strict=True)
    rank = group.rank()
    world_size = group.size()

    # Configure logging with rank prefix
    logging.basicConfig(
        level=logging.DEBUG,
        format=f"%(asctime)s [Rank {rank}] %(name)s %(levelname)s: %(message)s",
    )

    logger.info(
        f"Distributed process initialized: "
        f"rank={rank}, world_size={world_size}, backend={backend}"
    )

    # Lazy import to avoid circular dependencies and heavy imports
    # on worker ranks that may not need the full server.
    try:
        from .distributed import MLXCommunicator
    except ImportError:
        logger.error(
            "Failed to import MLXCommunicator. "
            "Ensure vllm_mlx.distributed module is available."
        )
        raise

    communicator = MLXCommunicator(group=group)

    # Store as global singleton so get_communicator() reuses it
    # instead of calling mx.distributed.init() again (which crashes).
    import vllm_mlx.distributed as _dist_mod
    _dist_mod._global_communicator = communicator

    # --- Set distributed environment vars BEFORE any collective ops ---
    os.environ["VLLM_MLX_DISTRIBUTED"] = "1"
    os.environ["VLLM_MLX_WORLD_SIZE"] = str(world_size)
    os.environ["VLLM_MLX_BACKEND"] = backend

    # --- Shared model loading (ALL ranks, collective operation) ---
    # Extract model name from server_args or environment
    model_name = os.environ.get("VLLM_MLX_MODEL", "")
    if not model_name and server_args:
        for i, arg in enumerate(server_args):
            if arg == "--model" and i + 1 < len(server_args):
                model_name = server_args[i + 1]
                break

    if not model_name:
        logger.error(f"[Rank {rank}] No model name provided")
        return

    # Set env for other components
    os.environ["VLLM_MLX_MODEL"] = model_name

    # ALL ranks: load model with sharded_load (collective operation).
    # This MUST be called on all ranks simultaneously because it uses
    # all_sum internally for weight distribution.
    logger.info(f"[Rank {rank}] Loading sharded model: {model_name}")
    from mlx_lm.utils import sharded_load

    model, tokenizer = sharded_load(
        model_name,
        pipeline_group=None,
        tensor_group=communicator.group,
    )
    logger.info(f"[Rank {rank}] Sharded model loaded successfully")

    # ALL ranks: synchronize after model loading
    communicator.barrier()
    logger.info(f"[Rank {rank}] Post-load barrier passed")

    if rank == 0:
        # Store pre-loaded model globally for server's BatchedEngine to pick up.
        # Must store on BOTH __main__ (this process) and the package module
        # (which batched.py imports from) since python -m loads as __main__.
        global _preloaded_model
        _preloaded_model = (model, tokenizer)
        import vllm_mlx.distributed_launcher as _launcher_mod
        _launcher_mod._preloaded_model = (model, tokenizer)
        _run_rank0_server(communicator, server_args)
    else:
        worker_loop(communicator, model_name=model_name, model=model, tokenizer=tokenizer)


def _run_rank0_server(
    communicator: Any,
    server_args: list[str] | None = None,
) -> None:
    """Run the API server on rank 0.

    This imports the server module and calls its ``main()`` function
    after injecting the communicator into the server's configuration.

    Args:
        communicator: An ``MLXCommunicator`` instance.
        server_args: CLI arguments for the server.
    """
    logger.info(
        f"[Rank 0] Starting API server "
        f"(world_size={communicator.world_size})"
    )

    # Phase 0: Just start the server normally.
    # Phase 2 will integrate the communicator with EngineCore/Scheduler
    # so that StepPlans are broadcast to worker ranks.
    from .server import main as server_main

    server_main(argv=server_args)


# ---------------------------------------------------------------------------
# CLI interface
# ---------------------------------------------------------------------------


def create_parser() -> argparse.ArgumentParser:
    """Create argument parser for the distributed launcher.

    Returns:
        An :class:`argparse.ArgumentParser` configured with all
        launcher options.
    """
    parser = argparse.ArgumentParser(
        description="Launch distributed vllm-mlx across multiple Mac nodes",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""\
Examples:
  # Launch on 2 local ranks with JACCL backend
  python -m vllm_mlx.distributed_launcher -n 2 --backend jaccl -- \\
      --model mlx-community/Llama-3.2-3B-Instruct-4bit

  # Launch with a hostfile for multi-node
  python -m vllm_mlx.distributed_launcher --hostfile hosts.json -- \\
      --model mlx-community/Llama-3.2-3B-Instruct-4bit

  # Launch with explicit hosts
  python -m vllm_mlx.distributed_launcher \\
      --hosts 10.254.0.1 10.254.0.2 --backend ring -- \\
      --model mlx-community/Llama-3.2-3B-Instruct-4bit
""",
    )
    parser.add_argument(
        "--backend",
        default="jaccl",
        choices=list(_SUPPORTED_BACKENDS),
        help="Distributed backend: jaccl (Thunderbolt 5 RDMA), "
        "ring (TCP), or any (auto-detect). Default: jaccl",
    )
    parser.add_argument(
        "--num-ranks",
        "-n",
        type=int,
        default=None,
        help="Number of ranks (processes). If not set, inferred from "
        "hostfile or hosts list.",
    )
    parser.add_argument(
        "--hostfile",
        type=str,
        default=None,
        help="Path to hostfile JSON for multi-node setup.",
    )
    parser.add_argument(
        "--hosts",
        type=str,
        nargs="+",
        default=None,
        help="List of host IPs (alternative to --hostfile for simple cases).",
    )
    parser.add_argument(
        "--env",
        type=str,
        nargs="*",
        default=[],
        help="Additional environment variables as KEY=VALUE pairs.",
    )
    # Everything after "--" is passed through to the server
    parser.add_argument(
        "server_args",
        nargs=argparse.REMAINDER,
        help="Arguments to pass to the vllm-mlx server (after --).",
    )
    return parser


def _parse_env_pairs(env_list: list[str]) -> dict[str, str]:
    """Parse ``KEY=VALUE`` strings into a dict."""
    result: dict[str, str] = {}
    for item in env_list:
        if "=" not in item:
            raise ValueError(
                f"Invalid environment variable format: {item!r}. "
                f"Expected KEY=VALUE."
            )
        key, _, value = item.partition("=")
        result[key] = value
    return result


def main() -> None:
    """Entry point for ``python -m vllm_mlx.distributed_launcher``.

    If ``MLX_RANK`` is set in the environment, this process was
    spawned by ``mlx.launch`` and should enter ``distributed_main``
    as a worker.  Otherwise, this is the launcher process and should
    call ``launch_distributed``.
    """
    # If MLX_RANK is set, we are a spawned worker process
    if "MLX_RANK" in os.environ:
        # Everything in sys.argv after the script name is server args
        server_args = sys.argv[1:] if len(sys.argv) > 1 else None

        # Strip leading "--" separator if present
        if server_args and server_args[0] == "--":
            server_args = server_args[1:]

        distributed_main(server_args=server_args or None)
        return

    # Otherwise, we are the launcher
    parser = create_parser()
    args = parser.parse_args()

    # Parse environment variable pairs
    envs = _parse_env_pairs(args.env) if args.env else None

    # Clean up server_args: remove leading "--" separator
    server_args = args.server_args
    if server_args and server_args[0] == "--":
        server_args = server_args[1:]

    # Determine the target script: ourselves, so spawned processes
    # will re-enter with MLX_RANK set and call distributed_main.
    script = os.path.abspath(__file__)

    launch_distributed(
        script_or_module=script,
        args=server_args or None,
        backend=args.backend,
        num_ranks=args.num_ranks,
        hostfile=args.hostfile,
        hosts=args.hosts,
        envs=envs,
    )


if __name__ == "__main__":
    main()
