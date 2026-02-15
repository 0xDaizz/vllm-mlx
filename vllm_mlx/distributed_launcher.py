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
    logger.info(f"[TP-DEBUG] rank{rank} BEFORE model forward, input_tokens.shape={input_tokens.shape}, batch_uids={len(batch.uids)}")
    logits = model(input_tokens, cache=batch.cache)
    # Eager eval is CRITICAL: completes the TP all_sum synchronization
    # with rank 0 before we wait for SpecDecodeResult.
    mx.eval(logits)
    logger.info(f"[TP-DEBUG] rank{rank} _worker_spec_decode_step: AFTER model forward complete, logits.shape={logits.shape}")

    # 3. Receive SpecDecodeResult from rank 0
    spec_result = communicator.receive_spec_decode_result()
    logger.info(f"[TP-DEBUG] rank{rank} _worker_spec_decode_step: received spec_result, finished_ids={spec_result.finished_ids}")

    # 4. Apply KV cache trim
    if spec_result.trim_amounts and any(t > 0 for t in spec_result.trim_amounts):
        from vllm_mlx.spec_decode.cache_utils import batch_variable_trim
        trim_array = mx.array(spec_result.trim_amounts, dtype=mx.int32)
        batch_variable_trim(batch.cache, trim_array)
        # Materialize trim results before further cache operations
        if batch.cache:
            mx.eval(batch.cache[0].offset, batch.cache[0].left_padding)

    # 5. Update batch state
    # Set new batch.y
    batch.y = mx.array(spec_result.new_y, dtype=mx.int32)

    # Update tokens and num_tokens for each request
    for rid, accepted in spec_result.accepted_tokens.items():
        if rid not in request_id_to_uid:
            continue
        uid = request_id_to_uid[rid]
        # Find batch index for this uid
        batch_idx = None
        for idx, b_uid in enumerate(batch.uids):
            if b_uid == uid:
                batch_idx = idx
                break
        if batch_idx is None:
            continue

        # Find original y for this position
        orig_y = batch_y[batch_idx] if batch_idx < len(batch_y) else 0

        # Batch all new tokens into a single concatenation
        tokens_for_cache = accepted[:-1] if accepted else []
        new_tokens = [orig_y] + tokens_for_cache
        if new_tokens:
            batch.tokens[batch_idx] = mx.concatenate(
                (batch.tokens[batch_idx], mx.array(new_tokens))
            )
        # Update num_tokens
        n_committed = len(accepted)
        batch.num_tokens[batch_idx] += n_committed

    # Materialize all updated tensors to prevent lazy graph accumulation
    mx.eval(batch.y, *batch.tokens)

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

    batch_generator = BatchGenerator(
        model=model,
        max_tokens=4096,
        sampler=sampler,
    )

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
        # Step 3: StepPlan processing loop
        while not shutdown_event.is_set():
            try:
                logger.info(f"[TP-DEBUG] rank{rank} worker_loop: waiting for StepPlan, batch_size={len(request_id_to_uid)}")
                # Receive StepPlan from rank 0
                plan = communicator.receive_step_plan()

                if plan is None:
                    # Shutdown signal
                    logger.info(f"[Rank {rank}] Received None plan, shutting down")
                    break

                # Apply removes first
                removes_processed = 0
                for request_id in plan.removes:
                    if request_id in request_id_to_uid:
                        uid = request_id_to_uid[request_id]
                        batch_generator.remove([uid])
                        del request_id_to_uid[request_id]
                        del uid_to_request_id[uid]
                        removes_processed += 1
                if removes_processed > 0:
                    logger.info(f"[TP-DEBUG] rank{rank} worker_loop: processed {removes_processed} removes, batch_size={len(request_id_to_uid)}")

                # Apply inserts
                for insert_op in plan.inserts:
                    uids = batch_generator.insert(
                        [insert_op.tokens],
                        max_tokens=[insert_op.max_tokens],
                    )
                    if uids:
                        uid = uids[0]
                        request_id_to_uid[insert_op.request_id] = uid
                        uid_to_request_id[uid] = insert_op.request_id

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
                    if local_fingerprint != plan.fingerprint:
                        logger.error(
                            f"[Rank {rank}] BATCH DESYNC DETECTED! "
                            f"Fingerprint mismatch: local={local_fingerprint}, "
                            f"rank0={plan.fingerprint}, "
                            f"local_requests={len(local_running_ids)}, "
                            f"batch_size={local_batch_size}"
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
                    logger.info(f"[TP-DEBUG] rank{rank} worker_loop: spec_decode_step done, finished={len(finished_spec_ids)}, batch_size={len(request_id_to_uid)}")
                else:
                    # Normal decode path
                    # Execute forward pass (model computation with all_sum).
                    # This must happen on all ranks simultaneously.
                    if request_id_to_uid:  # Only if there are active requests
                        _responses = batch_generator.next()

                        # Force async computation to complete before the
                        # broadcast overwrites batch.y.  batch_generator.next()
                        # calls mx.async_eval(batch.y, ...) internally; without
                        # this explicit eval, the broadcast below may overwrite
                        # batch.y before the async computation finishes.
                        if batch_generator.active_batch is not None:
                            mx.eval(batch_generator.active_batch.y)

                    # Receive broadcast token IDs from rank 0.
                    # Protocol: rank 0 sends broadcast_int(count) then
                    # broadcast_tokens(token_ids). Workers must match
                    # this exact sequence to stay synchronized.
                    if request_id_to_uid:
                        # Receive expected token count (matches rank 0's broadcast_int)
                        num_tokens = communicator.broadcast_int(0, src=0)

                        if num_tokens > 0:
                            token_array = mx.zeros((num_tokens,), dtype=mx.int32)
                            token_array = communicator.broadcast_tensor(token_array, src=0)
                            mx.eval(token_array)

                            # Apply broadcast tokens to batch state
                            if batch_generator.active_batch is not None:
                                batch = batch_generator.active_batch
                                # Use actual batch size for shape validation
                                num_active = len(batch.uids)
                                if token_array.shape[0] != num_active:
                                    logger.error(
                                        f"[Rank {rank}] Token broadcast shape mismatch: "
                                        f"received {token_array.shape[0]}, "
                                        f"batch has {num_active}"
                                    )
                                batch.y = token_array

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
        level=logging.INFO,
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
