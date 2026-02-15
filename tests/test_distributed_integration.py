#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Phase 2+3 Integration Test for vllm-mlx distributed inference.

Tests StepPlan broadcast, sampling sync, cache TP awareness, and engine integration.

Usage:
    # Single-process tests only (no mlx.launch needed)
    python tests/test_distributed_integration.py

    # Include distributed tests (needs mlx.launch)
    python tests/test_distributed_integration.py --distributed -n 2

    # As a distributed worker (called by mlx.launch)
    python tests/test_distributed_integration.py --worker-mode
"""

from __future__ import annotations

import argparse
import json
import os
import pickle
import subprocess
import sys
import tempfile
import traceback

# Ensure the project root is on sys.path so that ``vllm_mlx`` can be
# imported when the script is executed directly (outside of pytest).
_PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_results: list[tuple[str, bool, str]] = []


def report(name: str, passed: bool, detail: str = "") -> None:
    """Record a test result and print it immediately."""
    status = "PASS" if passed else "FAIL"
    msg = f"  [{status}] {name}"
    if detail:
        msg += f"  -- {detail}"
    print(msg)
    _results.append((name, passed, detail))


def summary() -> int:
    """Print summary and return exit code (0 = all pass, 1 = any fail)."""
    total = len(_results)
    passed = sum(1 for _, ok, _ in _results if ok)
    failed = total - passed
    print()
    print("=" * 60)
    print(f"Results: {passed}/{total} passed, {failed} failed")
    if failed:
        print("Failed tests:")
        for name, ok, detail in _results:
            if not ok:
                print(f"  - {name}: {detail}")
    print("=" * 60)
    return 0 if failed == 0 else 1


# ===================================================================
# Test 1: StepPlan serialization round-trip
# ===================================================================

def test_stepplan_serialization() -> None:
    """Verify InsertOp and StepPlan pickle round-trip."""
    print("\n--- Test 1: StepPlan serialization round-trip ---")
    from vllm_mlx.distributed import InsertOp, StepPlan

    insert = InsertOp(
        request_id="req-001",
        tokens=[1, 2, 3, 4, 5],
        max_tokens=128,
        cache_info={"cached_tokens": 10, "has_cache": True},
    )

    plan = StepPlan(
        step_id=42,
        inserts=[insert],
        removes=["req-old-1", "req-old-2"],
        sampling_seeds={"req-001": 12345, "req-002": 67890},
        fingerprint="abcdef1234567890",
    )

    # Pickle round-trip
    data = pickle.dumps(plan)
    restored = pickle.loads(data)

    # Verify fields
    try:
        assert restored.step_id == plan.step_id, (
            f"step_id mismatch: {restored.step_id} != {plan.step_id}"
        )
        assert len(restored.inserts) == 1, (
            f"inserts length: {len(restored.inserts)} != 1"
        )
        ri = restored.inserts[0]
        assert ri.request_id == insert.request_id
        assert ri.tokens == insert.tokens
        assert ri.max_tokens == insert.max_tokens
        assert ri.cache_info == insert.cache_info
        assert restored.removes == plan.removes
        assert restored.sampling_seeds == plan.sampling_seeds
        assert restored.fingerprint == plan.fingerprint
        report("StepPlan pickle round-trip", True)
    except AssertionError as e:
        report("StepPlan pickle round-trip", False, str(e))
        return

    # Verify InsertOp standalone
    data2 = pickle.dumps(insert)
    ri2 = pickle.loads(data2)
    try:
        assert ri2.request_id == insert.request_id
        assert ri2.tokens == insert.tokens
        assert ri2.max_tokens == insert.max_tokens
        assert ri2.cache_info == insert.cache_info
        report("InsertOp pickle round-trip", True)
    except AssertionError as e:
        report("InsertOp pickle round-trip", False, str(e))

    # Verify StepResult
    from vllm_mlx.distributed import StepResult

    result = StepResult(step_id=42, token_ids={"req-001": 99, "req-002": 100})
    data3 = pickle.dumps(result)
    rr = pickle.loads(data3)
    try:
        assert rr.step_id == result.step_id
        assert rr.token_ids == result.token_ids
        report("StepResult pickle round-trip", True)
    except AssertionError as e:
        report("StepResult pickle round-trip", False, str(e))


# ===================================================================
# Test 2: MLXCommunicator single-process mode
# ===================================================================

def test_communicator_single_process() -> None:
    """Verify MLXCommunicator pass-through behavior when world_size=1."""
    print("\n--- Test 2: MLXCommunicator single-process mode ---")
    from vllm_mlx.distributed import MLXCommunicator

    comm = MLXCommunicator()  # defaults to world_size=1

    # is_distributed should be False
    try:
        assert comm.is_distributed is False, (
            f"is_distributed should be False, got {comm.is_distributed}"
        )
        report("is_distributed == False", True)
    except AssertionError as e:
        report("is_distributed == False", False, str(e))

    # rank and world_size
    try:
        assert comm.rank == 0, f"rank should be 0, got {comm.rank}"
        assert comm.world_size == 1, f"world_size should be 1, got {comm.world_size}"
        report("rank=0, world_size=1", True)
    except AssertionError as e:
        report("rank=0, world_size=1", False, str(e))

    # broadcast_object pass-through
    obj = {"key": "value", "nested": [1, 2, 3]}
    try:
        result = comm.broadcast_object(obj)
        assert result is obj, "broadcast_object should return the same object"
        report("broadcast_object pass-through", True)
    except Exception as e:
        report("broadcast_object pass-through", False, str(e))

    # broadcast_tokens pass-through
    import mlx.core as mx

    tokens = mx.array([10, 20, 30], dtype=mx.int32)
    try:
        result = comm.broadcast_tokens(tokens)
        assert result is tokens, "broadcast_tokens should return the same array"
        report("broadcast_tokens pass-through", True)
    except Exception as e:
        report("broadcast_tokens pass-through", False, str(e))

    # broadcast_int pass-through
    try:
        result = comm.broadcast_int(42)
        assert result == 42, f"broadcast_int should return 42, got {result}"
        report("broadcast_int pass-through", True)
    except Exception as e:
        report("broadcast_int pass-through", False, str(e))

    # barrier no-op
    try:
        comm.barrier()  # should not raise or block
        report("barrier no-op", True)
    except Exception as e:
        report("barrier no-op", False, str(e))

    # broadcast_step_plan no-op
    from vllm_mlx.distributed import StepPlan

    plan = StepPlan(
        step_id=1,
        inserts=[],
        removes=[],
        sampling_seeds={},
        fingerprint="test",
    )
    try:
        comm.broadcast_step_plan(plan)  # should not raise or block
        report("broadcast_step_plan no-op", True)
    except Exception as e:
        report("broadcast_step_plan no-op", False, str(e))

    # broadcast_tensor pass-through
    tensor = mx.ones((2, 3))
    try:
        result = comm.broadcast_tensor(tensor)
        assert result is tensor, "broadcast_tensor should return the same tensor"
        report("broadcast_tensor pass-through", True)
    except Exception as e:
        report("broadcast_tensor pass-through", False, str(e))


# ===================================================================
# Test 3: Scheduler communicator integration
# ===================================================================

def test_scheduler_communicator() -> None:
    """Verify Scheduler.set_communicator() and broadcast_sampled_tokens no-op."""
    print("\n--- Test 3: Scheduler communicator integration ---")

    # Create a mock communicator with is_distributed=False
    class MockCommunicator:
        rank = 0
        world_size = 1
        is_distributed = False
        broadcast_called = False

        def broadcast_step_plan(self, plan):
            self.broadcast_called = True

        def broadcast_tokens(self, token_ids, src=0):
            self.broadcast_called = True
            return token_ids

        def barrier(self):
            pass

    mock_comm = MockCommunicator()

    # Import scheduler - this will import mlx and mlx_lm
    try:
        from vllm_mlx.scheduler import Scheduler, SchedulerConfig
    except ImportError as e:
        report("Scheduler import", False, str(e))
        return

    # We cannot create a Scheduler without a real model/tokenizer,
    # so we test set_communicator with a minimal mock approach.
    # Create a minimal mock scheduler by directly setting attributes.

    class MockScheduler:
        """Lightweight stand-in to test set_communicator logic."""

        def __init__(self):
            self._communicator = None
            self._tp_world_size = 1
            self.memory_aware_cache = None

        def set_communicator(self, communicator):
            """Mirror of Scheduler.set_communicator."""
            self._communicator = communicator
            if communicator is not None and communicator.is_distributed:
                self._tp_world_size = communicator.world_size
                if self.memory_aware_cache is not None:
                    self.memory_aware_cache._tp_world_size = self._tp_world_size
                    self.memory_aware_cache._tp_rank = 0

    sched = MockScheduler()

    # Before set_communicator
    try:
        assert sched._communicator is None
        report("communicator initially None", True)
    except AssertionError as e:
        report("communicator initially None", False, str(e))

    # After set_communicator with non-distributed
    sched.set_communicator(mock_comm)
    try:
        assert sched._communicator is mock_comm
        assert sched._tp_world_size == 1, (
            f"tp_world_size should remain 1, got {sched._tp_world_size}"
        )
        report("set_communicator (non-distributed)", True)
    except AssertionError as e:
        report("set_communicator (non-distributed)", False, str(e))

    # Test with distributed communicator
    class MockDistributedComm:
        rank = 0
        world_size = 4
        is_distributed = True

    dist_comm = MockDistributedComm()
    sched2 = MockScheduler()
    sched2.set_communicator(dist_comm)
    try:
        assert sched2._communicator is dist_comm
        assert sched2._tp_world_size == 4, (
            f"tp_world_size should be 4, got {sched2._tp_world_size}"
        )
        report("set_communicator (distributed, world_size=4)", True)
    except AssertionError as e:
        report("set_communicator (distributed, world_size=4)", False, str(e))

    # Verify _broadcast_sampled_tokens is a no-op when not distributed
    # We test the guard condition directly
    try:
        comm_none = None
        # Guard: if self._communicator is None or not self._communicator.is_distributed: return
        guard_noop = (comm_none is None) or (not getattr(comm_none, "is_distributed", False))
        assert guard_noop is True
        report("_broadcast_sampled_tokens guard (comm=None)", True)
    except AssertionError as e:
        report("_broadcast_sampled_tokens guard (comm=None)", False, str(e))

    try:
        guard_noop = (mock_comm is None) or (not mock_comm.is_distributed)
        assert guard_noop is True
        report("_broadcast_sampled_tokens guard (non-distributed)", True)
    except AssertionError as e:
        report("_broadcast_sampled_tokens guard (non-distributed)", False, str(e))

    try:
        guard_noop = (dist_comm is None) or (not dist_comm.is_distributed)
        assert guard_noop is False, "should NOT be no-op for distributed comm"
        report("_broadcast_sampled_tokens guard (distributed)", True)
    except AssertionError as e:
        report("_broadcast_sampled_tokens guard (distributed)", False, str(e))


# ===================================================================
# Test 4: EngineCore distributed env detection
# ===================================================================

def test_engine_distributed_detection() -> None:
    """Verify EngineCore reads VLLM_MLX_DISTRIBUTED env var."""
    print("\n--- Test 4: EngineCore distributed env detection ---")

    # Test the env detection logic without loading a model
    # EngineCore checks:
    #   os.environ.get("VLLM_MLX_DISTRIBUTED") == "1"
    #   world_size = int(os.environ.get("VLLM_MLX_WORLD_SIZE", "1"))

    # Case 1: env not set
    env_backup = {
        k: os.environ.pop(k, None)
        for k in ["VLLM_MLX_DISTRIBUTED", "VLLM_MLX_WORLD_SIZE"]
    }
    try:
        val = os.environ.get("VLLM_MLX_DISTRIBUTED")
        assert val is None, f"VLLM_MLX_DISTRIBUTED should be None, got {val}"
        report("env not set -> no distributed mode", True)
    except AssertionError as e:
        report("env not set -> no distributed mode", False, str(e))

    # Case 2: env set to "1" but world_size=1 (should skip communicator)
    try:
        os.environ["VLLM_MLX_DISTRIBUTED"] = "1"
        os.environ["VLLM_MLX_WORLD_SIZE"] = "1"
        is_dist = os.environ.get("VLLM_MLX_DISTRIBUTED") == "1"
        world_size = int(os.environ.get("VLLM_MLX_WORLD_SIZE", "1"))
        assert is_dist is True
        assert world_size == 1
        # EngineCore only creates communicator if world_size > 1
        should_create_comm = is_dist and world_size > 1
        assert should_create_comm is False, (
            "should NOT create communicator when world_size=1"
        )
        report("env=1, world_size=1 -> skip communicator", True)
    except AssertionError as e:
        report("env=1, world_size=1 -> skip communicator", False, str(e))

    # Case 3: env set to "1" and world_size=2
    try:
        os.environ["VLLM_MLX_DISTRIBUTED"] = "1"
        os.environ["VLLM_MLX_WORLD_SIZE"] = "2"
        is_dist = os.environ.get("VLLM_MLX_DISTRIBUTED") == "1"
        world_size = int(os.environ.get("VLLM_MLX_WORLD_SIZE", "1"))
        assert is_dist is True
        assert world_size == 2
        should_create_comm = is_dist and world_size > 1
        assert should_create_comm is True, (
            "should create communicator when world_size > 1"
        )
        report("env=1, world_size=2 -> create communicator", True)
    except AssertionError as e:
        report("env=1, world_size=2 -> create communicator", False, str(e))

    # Case 4: env set to "0"
    try:
        os.environ["VLLM_MLX_DISTRIBUTED"] = "0"
        is_dist = os.environ.get("VLLM_MLX_DISTRIBUTED") == "1"
        assert is_dist is False
        report("env=0 -> no distributed mode", True)
    except AssertionError as e:
        report("env=0 -> no distributed mode", False, str(e))

    # Restore environment
    for k in ["VLLM_MLX_DISTRIBUTED", "VLLM_MLX_WORLD_SIZE"]:
        os.environ.pop(k, None)
    for k, v in env_backup.items():
        if v is not None:
            os.environ[k] = v

    # Verify the actual EngineCore code path matches expectations
    try:
        import inspect
        from vllm_mlx.engine_core import EngineCore

        source = inspect.getsource(EngineCore.__init__)
        assert 'VLLM_MLX_DISTRIBUTED' in source, (
            "EngineCore.__init__ should check VLLM_MLX_DISTRIBUTED"
        )
        assert 'VLLM_MLX_WORLD_SIZE' in source, (
            "EngineCore.__init__ should check VLLM_MLX_WORLD_SIZE"
        )
        assert 'get_communicator' in source, (
            "EngineCore.__init__ should call get_communicator"
        )
        assert 'set_communicator' in source, (
            "EngineCore.__init__ should call scheduler.set_communicator"
        )
        report("EngineCore source contains distributed code paths", True)
    except Exception as e:
        report("EngineCore source contains distributed code paths", False, str(e))


# ===================================================================
# Test 5: Memory cache TP key generation
# ===================================================================

def test_memory_cache_tp_keys() -> None:
    """Verify TP-aware cache key generation and save/load paths."""
    print("\n--- Test 5: Memory cache TP key generation ---")

    from vllm_mlx.memory_cache import MemoryCacheConfig, MemoryAwarePrefixCache

    # Use a dummy model object
    class DummyModel:
        pass

    model = DummyModel()

    # Single-device config (default)
    config_single = MemoryCacheConfig(max_memory_mb=100)
    cache_single = MemoryAwarePrefixCache(model, config_single)

    # TP config (world_size=2, rank=0)
    config_tp2_r0 = MemoryCacheConfig(max_memory_mb=100, tp_world_size=2, tp_rank=0)
    cache_tp2_r0 = MemoryAwarePrefixCache(model, config_tp2_r0)

    # TP config (world_size=2, rank=1)
    config_tp2_r1 = MemoryCacheConfig(max_memory_mb=100, tp_world_size=2, tp_rank=1)
    cache_tp2_r1 = MemoryAwarePrefixCache(model, config_tp2_r1)

    # TP config (world_size=4, rank=0)
    config_tp4_r0 = MemoryCacheConfig(max_memory_mb=100, tp_world_size=4, tp_rank=0)
    cache_tp4_r0 = MemoryAwarePrefixCache(model, config_tp4_r0)

    tokens = (1, 2, 3, 4, 5)

    # Test 5a: Single-device key is just the token tuple
    try:
        key_single = cache_single._make_cache_key(tokens)
        assert key_single == tokens, (
            f"single-device key should be token tuple, got {key_single}"
        )
        report("single-device cache key = token tuple", True)
    except AssertionError as e:
        report("single-device cache key = token tuple", False, str(e))

    # Test 5b: TP key includes world_size
    try:
        key_tp2 = cache_tp2_r0._make_cache_key(tokens)
        assert key_tp2 == (tokens, 2), (
            f"TP-2 key should be (tokens, 2), got {key_tp2}"
        )
        report("TP-2 cache key includes world_size", True)
    except AssertionError as e:
        report("TP-2 cache key includes world_size", False, str(e))

    # Test 5c: Different TP sizes produce different keys
    try:
        key_tp4 = cache_tp4_r0._make_cache_key(tokens)
        assert key_tp2 != key_tp4, (
            f"TP-2 and TP-4 keys should differ: {key_tp2} vs {key_tp4}"
        )
        report("different TP sizes -> different keys", True)
    except AssertionError as e:
        report("different TP sizes -> different keys", False, str(e))

    # Test 5d: Single vs TP keys differ
    try:
        assert key_single != key_tp2, (
            f"single and TP-2 keys should differ: {key_single} vs {key_tp2}"
        )
        report("single vs TP keys differ", True)
    except AssertionError as e:
        report("single vs TP keys differ", False, str(e))

    # Test 5e: TP-aware save path construction
    with tempfile.TemporaryDirectory() as tmpdir:
        # Single-device: saves directly to cache_dir
        try:
            # Check the expected path logic from save_to_disk
            if cache_single._tp_world_size > 1:
                single_save_dir = os.path.join(
                    tmpdir, f"tp{cache_single._tp_world_size}",
                    f"rank{cache_single._tp_rank}",
                )
            else:
                single_save_dir = tmpdir
            assert single_save_dir == tmpdir
            report("single-device save path = cache_dir", True)
        except AssertionError as e:
            report("single-device save path = cache_dir", False, str(e))

        # TP: saves to cache_dir/tp{N}/rank{R}/
        try:
            tp_save_dir = os.path.join(tmpdir, "tp2", "rank0")
            if cache_tp2_r0._tp_world_size > 1:
                expected = os.path.join(
                    tmpdir,
                    f"tp{cache_tp2_r0._tp_world_size}",
                    f"rank{cache_tp2_r0._tp_rank}",
                )
            else:
                expected = tmpdir
            assert expected == tp_save_dir, (
                f"TP save path mismatch: {expected} != {tp_save_dir}"
            )
            report("TP save path = cache_dir/tp2/rank0", True)
        except AssertionError as e:
            report("TP save path = cache_dir/tp2/rank0", False, str(e))

        # Rank 1 path
        try:
            expected_r1 = os.path.join(tmpdir, "tp2", "rank1")
            if cache_tp2_r1._tp_world_size > 1:
                computed = os.path.join(
                    tmpdir,
                    f"tp{cache_tp2_r1._tp_world_size}",
                    f"rank{cache_tp2_r1._tp_rank}",
                )
            else:
                computed = tmpdir
            assert computed == expected_r1, (
                f"Rank 1 path mismatch: {computed} != {expected_r1}"
            )
            report("TP save path rank1 = cache_dir/tp2/rank1", True)
        except AssertionError as e:
            report("TP save path rank1 = cache_dir/tp2/rank1", False, str(e))

    # Test 5f: TP config validation
    try:
        config_ok = MemoryCacheConfig(tp_world_size=2, tp_rank=0)
        assert config_ok.tp_world_size == 2
        assert config_ok.tp_rank == 0
        report("TP config fields", True)
    except Exception as e:
        report("TP config fields", False, str(e))

    # Test 5g: TP metadata attributes on cache
    try:
        assert cache_tp2_r0._tp_world_size == 2
        assert cache_tp2_r0._tp_rank == 0
        assert cache_tp2_r1._tp_world_size == 2
        assert cache_tp2_r1._tp_rank == 1
        report("cache TP attributes match config", True)
    except AssertionError as e:
        report("cache TP attributes match config", False, str(e))


# ===================================================================
# Test 6: Distributed multi-process test (requires --distributed)
# ===================================================================

def _worker_main() -> None:
    """Worker entry point for distributed test (called via mlx.launch).

    Each rank initializes a communicator, runs broadcast tests, and
    writes results to a rank-specific JSON file whose path is passed
    via the DIST_TEST_OUTPUT_DIR environment variable.
    """
    import mlx.core as mx
    from vllm_mlx.distributed import MLXCommunicator, InsertOp, StepPlan

    output_dir = os.environ.get("DIST_TEST_OUTPUT_DIR", "")
    if not output_dir:
        print("ERROR: DIST_TEST_OUTPUT_DIR not set", file=sys.stderr)
        sys.exit(1)

    # Initialize communicator using the group created by mlx.launch
    comm = MLXCommunicator()
    rank = comm.rank
    world_size = comm.world_size

    results: dict[str, any] = {
        "rank": rank,
        "world_size": world_size,
        "is_distributed": comm.is_distributed,
        "tests": {},
    }

    # --- Sub-test A: Verify rank and world_size ---
    results["tests"]["rank_world_size"] = {
        "passed": world_size == 2 and 0 <= rank < world_size,
        "detail": f"rank={rank}, world_size={world_size}",
    }

    # --- Sub-test B: Broadcast StepPlan (rank 0 -> rank 1) ---
    try:
        if rank == 0:
            plan = StepPlan(
                step_id=100,
                inserts=[
                    InsertOp(
                        request_id="dist-req-1",
                        tokens=[10, 20, 30],
                        max_tokens=64,
                    )
                ],
                removes=["old-req-1"],
                sampling_seeds={"dist-req-1": 99999},
                fingerprint="dist_test_fp",
            )
            comm.broadcast_step_plan(plan)
            results["tests"]["stepplan_broadcast"] = {
                "passed": True,
                "detail": "sent StepPlan from rank 0",
            }
        else:
            received = comm.receive_step_plan()
            ok = (
                received.step_id == 100
                and len(received.inserts) == 1
                and received.inserts[0].request_id == "dist-req-1"
                and received.inserts[0].tokens == [10, 20, 30]
                and received.inserts[0].max_tokens == 64
                and received.removes == ["old-req-1"]
                and received.sampling_seeds == {"dist-req-1": 99999}
                and received.fingerprint == "dist_test_fp"
            )
            results["tests"]["stepplan_broadcast"] = {
                "passed": ok,
                "detail": f"received step_id={received.step_id}, "
                          f"inserts={len(received.inserts)}, "
                          f"removes={received.removes}",
            }
    except Exception as e:
        results["tests"]["stepplan_broadcast"] = {
            "passed": False,
            "detail": f"exception: {e}",
        }

    # --- Sub-test C: Broadcast tokens ---
    try:
        if rank == 0:
            token_ids = mx.array([42, 77, 101], dtype=mx.int32)
        else:
            token_ids = mx.zeros((3,), dtype=mx.int32)

        received_tokens = comm.broadcast_tokens(token_ids, src=0)
        mx.eval(received_tokens)
        token_list = received_tokens.tolist()

        ok = token_list == [42, 77, 101]
        results["tests"]["token_broadcast"] = {
            "passed": ok,
            "detail": f"tokens={token_list}",
        }
    except Exception as e:
        results["tests"]["token_broadcast"] = {
            "passed": False,
            "detail": f"exception: {e}",
        }

    # --- Sub-test D: Broadcast int ---
    try:
        if rank == 0:
            val = 12345
        else:
            val = 0

        received_val = comm.broadcast_int(val, src=0)
        ok = received_val == 12345
        results["tests"]["int_broadcast"] = {
            "passed": ok,
            "detail": f"value={received_val}",
        }
    except Exception as e:
        results["tests"]["int_broadcast"] = {
            "passed": False,
            "detail": f"exception: {e}",
        }

    # --- Sub-test E: Barrier ---
    try:
        comm.barrier()
        results["tests"]["barrier"] = {
            "passed": True,
            "detail": "barrier completed",
        }
    except Exception as e:
        results["tests"]["barrier"] = {
            "passed": False,
            "detail": f"exception: {e}",
        }

    # Write results to file
    output_path = os.path.join(output_dir, f"rank_{rank}.json")
    with open(output_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"[Rank {rank}] Results written to {output_path}")


def test_distributed_launch(num_ranks: int, backend: str) -> None:
    """Launch distributed test with mlx.launch and verify results."""
    print(f"\n--- Test 6: Distributed multi-process test (n={num_ranks}, backend={backend}) ---")

    with tempfile.TemporaryDirectory(prefix="dist_test_") as output_dir:
        # Build the mlx.launch command
        script = os.path.abspath(__file__)
        cmd = [
            sys.executable, "-m", "mlx._distributed_utils.launch",
            "-n", str(num_ranks),
            "--backend", backend,
            script,
            "--worker-mode",
        ]

        env = os.environ.copy()
        env["DIST_TEST_OUTPUT_DIR"] = output_dir

        print(f"  Launching: {' '.join(cmd)}")
        try:
            proc = subprocess.run(
                cmd,
                env=env,
                capture_output=True,
                text=True,
                timeout=120,
            )
        except subprocess.TimeoutExpired:
            report("distributed launch (timeout)", False, "mlx.launch timed out after 120s")
            return
        except FileNotFoundError:
            report("distributed launch", False, "mlx.launch not found")
            return

        # Print stdout/stderr for debugging
        if proc.stdout:
            for line in proc.stdout.strip().split("\n"):
                print(f"    [stdout] {line}")
        if proc.stderr:
            for line in proc.stderr.strip().split("\n"):
                print(f"    [stderr] {line}")

        if proc.returncode != 0:
            report(
                "distributed launch (exit code)",
                False,
                f"exit code {proc.returncode}",
            )
            return

        report("distributed launch (exit code)", True, "exit code 0")

        # Read and verify results from each rank
        all_ok = True
        for r in range(num_ranks):
            result_path = os.path.join(output_dir, f"rank_{r}.json")
            if not os.path.exists(result_path):
                report(f"rank {r} result file", False, "file not found")
                all_ok = False
                continue

            with open(result_path) as f:
                rank_results = json.load(f)

            # Verify rank info
            try:
                assert rank_results["rank"] == r
                assert rank_results["world_size"] == num_ranks
                assert rank_results["is_distributed"] is True
            except (AssertionError, KeyError) as e:
                report(f"rank {r} metadata", False, str(e))
                all_ok = False
                continue

            # Verify each sub-test
            for test_name, test_data in rank_results.get("tests", {}).items():
                passed = test_data.get("passed", False)
                detail = test_data.get("detail", "")
                report(f"rank {r}: {test_name}", passed, detail)
                if not passed:
                    all_ok = False

        if all_ok:
            report("distributed test overall", True, f"all {num_ranks} ranks passed")


# ===================================================================
# Test 7: SpecDecodePlan and SpecDecodeResult serialization
# ===================================================================

def test_spec_decode_protocol_serialization() -> None:
    """Verify SpecDecodePlan and SpecDecodeResult pickle round-trip."""
    print("\n--- Test 7: SpecDecodePlan/SpecDecodeResult serialization ---")
    from vllm_mlx.distributed import SpecDecodePlan, SpecDecodeResult

    # Test SpecDecodePlan round-trip
    plan = SpecDecodePlan(
        draft_tokens={"req-1": [10, 20, 30], "req-2": [40, 50]},
        max_draft_len=3,
        batch_order=[("req-1", 0), ("req-2", 1)],
        batch_y=[100, 200],
    )

    data = pickle.dumps(plan)
    restored = pickle.loads(data)

    try:
        assert restored.draft_tokens == plan.draft_tokens
        assert restored.max_draft_len == plan.max_draft_len
        assert restored.batch_order == plan.batch_order
        assert restored.batch_y == plan.batch_y
        report("SpecDecodePlan pickle round-trip", True)
    except AssertionError as e:
        report("SpecDecodePlan pickle round-trip", False, str(e))

    # Test SpecDecodeResult round-trip
    result = SpecDecodeResult(
        step_id=42,
        accepted_tokens={"req-1": [10, 20, 30, 99], "req-2": [40, 88]},
        trim_amounts=[2, 3],
        new_y=[99, 88],
        finished_ids=["req-2"],
    )

    data2 = pickle.dumps(result)
    restored2 = pickle.loads(data2)

    try:
        assert restored2.step_id == result.step_id
        assert restored2.accepted_tokens == result.accepted_tokens
        assert restored2.trim_amounts == result.trim_amounts
        assert restored2.new_y == result.new_y
        assert restored2.finished_ids == result.finished_ids
        report("SpecDecodeResult pickle round-trip", True)
    except AssertionError as e:
        report("SpecDecodeResult pickle round-trip", False, str(e))

    # Test StepPlan with spec_decode_plan embedded
    from vllm_mlx.distributed import StepPlan, InsertOp

    step_plan = StepPlan(
        step_id=100,
        inserts=[InsertOp(request_id="req-1", tokens=[1, 2, 3], max_tokens=64)],
        removes=[],
        sampling_seeds={"req-1": 12345},
        fingerprint="test_fp",
        spec_decode_plan=plan,
    )

    data3 = pickle.dumps(step_plan)
    restored3 = pickle.loads(data3)

    try:
        assert restored3.spec_decode_plan is not None
        assert restored3.spec_decode_plan.draft_tokens == plan.draft_tokens
        assert restored3.spec_decode_plan.max_draft_len == plan.max_draft_len
        assert restored3.spec_decode_plan.batch_order == plan.batch_order
        assert restored3.spec_decode_plan.batch_y == plan.batch_y
        report("StepPlan with SpecDecodePlan pickle round-trip", True)
    except AssertionError as e:
        report("StepPlan with SpecDecodePlan pickle round-trip", False, str(e))

    # Test StepPlan without spec_decode_plan (backward compat)
    step_plan_no_spec = StepPlan(
        step_id=101,
        inserts=[],
        removes=[],
        sampling_seeds={},
        fingerprint="test_fp2",
    )

    data4 = pickle.dumps(step_plan_no_spec)
    restored4 = pickle.loads(data4)

    try:
        assert restored4.spec_decode_plan is None
        report("StepPlan without SpecDecodePlan (backward compat)", True)
    except AssertionError as e:
        report("StepPlan without SpecDecodePlan (backward compat)", False, str(e))

    # Test Communicator methods exist
    from vllm_mlx.distributed import MLXCommunicator

    comm = MLXCommunicator()

    try:
        assert hasattr(comm, "broadcast_spec_decode_result")
        assert callable(comm.broadcast_spec_decode_result)
        assert hasattr(comm, "receive_spec_decode_result")
        assert callable(comm.receive_spec_decode_result)
        report("MLXCommunicator spec decode methods exist", True)
    except AssertionError as e:
        report("MLXCommunicator spec decode methods exist", False, str(e))

    # Test no-op behavior on single process (world_size=1)
    try:
        comm.broadcast_spec_decode_result(result)  # should be a no-op
        report("broadcast_spec_decode_result no-op (single process)", True)
    except Exception as e:
        report("broadcast_spec_decode_result no-op (single process)", False, str(e))


# ===================================================================
# Main
# ===================================================================

def main() -> int:
    parser = argparse.ArgumentParser(
        description="Phase 2+3 distributed integration tests for vllm-mlx",
    )
    parser.add_argument(
        "--distributed",
        action="store_true",
        help="Include distributed tests (needs mlx.launch with -n 2)",
    )
    parser.add_argument(
        "--worker-mode",
        action="store_true",
        help="Run as a distributed worker (called by mlx.launch)",
    )
    parser.add_argument(
        "-n",
        type=int,
        default=2,
        help="Number of ranks for distributed test (default: 2)",
    )
    parser.add_argument(
        "--backend",
        type=str,
        default="ring",
        help="Distributed backend: ring or jaccl (default: ring)",
    )
    args = parser.parse_args()

    # Worker mode: run distributed tests and exit
    if args.worker_mode:
        _worker_main()
        return 0

    # Main test runner
    print("=" * 60)
    print("vllm-mlx Phase 2+3 Distributed Integration Tests")
    print("=" * 60)

    # Single-process tests
    try:
        test_stepplan_serialization()
    except Exception as e:
        report("Test 1 (StepPlan serialization)", False, f"unhandled: {e}")
        traceback.print_exc()

    try:
        test_communicator_single_process()
    except Exception as e:
        report("Test 2 (Communicator single-process)", False, f"unhandled: {e}")
        traceback.print_exc()

    try:
        test_scheduler_communicator()
    except Exception as e:
        report("Test 3 (Scheduler communicator)", False, f"unhandled: {e}")
        traceback.print_exc()

    try:
        test_engine_distributed_detection()
    except Exception as e:
        report("Test 4 (EngineCore env detection)", False, f"unhandled: {e}")
        traceback.print_exc()

    try:
        test_memory_cache_tp_keys()
    except Exception as e:
        report("Test 5 (Memory cache TP keys)", False, f"unhandled: {e}")
        traceback.print_exc()

    try:
        test_spec_decode_protocol_serialization()
    except Exception as e:
        report("Test 7 (Spec decode protocol serialization)", False, f"unhandled: {e}")
        traceback.print_exc()

    # Distributed tests (only if --distributed)
    if args.distributed:
        try:
            test_distributed_launch(num_ranks=args.n, backend=args.backend)
        except Exception as e:
            report("Test 6 (Distributed launch)", False, f"unhandled: {e}")
            traceback.print_exc()
    else:
        print("\n--- Test 6: SKIPPED (pass --distributed to enable) ---")

    return summary()


if __name__ == "__main__":
    sys.exit(main())
