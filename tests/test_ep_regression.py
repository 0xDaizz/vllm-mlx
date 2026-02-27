# SPDX-License-Identifier: Apache-2.0
"""Regression tests: EP code must not affect existing TP behavior."""

import argparse

import pytest

from vllm_mlx.cli_args import (
    add_all_serve_args,
    add_distributed_args,
    rebuild_server_args_from_namespace,
    validate_serve_args,
)


def _make_parser(*, distributed: bool = True):
    parser = argparse.ArgumentParser()
    add_all_serve_args(parser, positional_model=False)
    if distributed:
        add_distributed_args(parser)
    return parser


def _parse(argv, **kwargs):
    return _make_parser(**kwargs).parse_args(argv)


class TestEPCLIValidation:
    """EP-specific CLI validation."""

    def test_ep_requires_distributed(self):
        args = _parse([
            "--model", "m",
            "--continuous-batching",
            "--expert-parallel",
            "--dist-num-ranks", "2",
        ])
        # expert_parallel is True but distributed is False
        args.distributed = False
        with pytest.raises(SystemExit):
            validate_serve_args(args)

    def test_ep_requires_continuous_batching(self):
        args = _parse([
            "--model", "m",
            "--distributed",
            "--expert-parallel",
            "--dist-num-ranks", "2",
        ])
        with pytest.raises(SystemExit):
            validate_serve_args(args)

    def test_ep_valid_configuration(self):
        args = _parse([
            "--model", "m",
            "--continuous-batching",
            "--distributed",
            "--expert-parallel",
            "--dist-num-ranks", "2",
        ])
        # Should not raise
        validate_serve_args(args)

    def test_ep_custom_capacity_factor(self):
        args = _parse([
            "--model", "m",
            "--continuous-batching",
            "--distributed",
            "--expert-parallel",
            "--ep-capacity-factor", "1.5",
            "--ep-kernel-backend", "metal",
            "--dist-num-ranks", "2",
        ])
        assert args.ep_capacity_factor == 1.5
        assert args.ep_kernel_backend == "metal"
        validate_serve_args(args)


class TestEPRebuildArgs:
    """EP flags must round-trip through rebuild_server_args_from_namespace."""

    def test_ep_flags_roundtrip(self):
        args1 = _parse([
            "--model", "m",
            "--continuous-batching",
            "--distributed",
            "--expert-parallel",
            "--ep-capacity-factor", "1.5",
            "--ep-kernel-backend", "metal",
            "--dist-num-ranks", "2",
        ])
        rebuilt = rebuild_server_args_from_namespace(args1)

        assert "--expert-parallel" in rebuilt
        assert "--ep-capacity-factor" in rebuilt
        assert "1.5" in rebuilt
        assert "--ep-kernel-backend" in rebuilt
        assert "metal" in rebuilt

        # Re-parse and verify
        args2 = _parse(rebuilt)
        assert args2.expert_parallel is True
        assert args2.ep_capacity_factor == 1.5
        assert args2.ep_kernel_backend == "metal"

    def test_ep_defaults_not_in_rebuilt(self):
        """Default EP values should not appear in rebuilt args."""
        args = _parse([
            "--model", "m",
            "--continuous-batching",
            "--distributed",
            "--dist-num-ranks", "2",
        ])
        rebuilt = rebuild_server_args_from_namespace(args)
        assert "--expert-parallel" not in rebuilt
        assert "--ep-capacity-factor" not in rebuilt
        assert "--ep-kernel-backend" not in rebuilt


class TestTPRegression:
    """Ensure TP-only mode is unaffected by EP additions."""

    def test_tp_only_validates(self):
        """Standard TP config (without EP) should still validate."""
        args = _parse([
            "--model", "m",
            "--continuous-batching",
            "--distributed",
            "--dist-num-ranks", "2",
        ])
        validate_serve_args(args)

    def test_tp_rebuild_unchanged(self):
        """TP-only rebuild should not include EP flags."""
        args = _parse([
            "--model", "m",
            "--continuous-batching",
            "--distributed",
            "--dist-num-ranks", "2",
        ])
        rebuilt = rebuild_server_args_from_namespace(args)
        assert "--expert-parallel" not in rebuilt
        assert "--ep-capacity-factor" not in rebuilt

    def test_ep_imports_dont_break(self):
        """EP modules should import without errors."""
        import vllm_mlx.check_mlx_ep
        import vllm_mlx.ep_adapter
        import vllm_mlx.ep_loader

    def test_ep_files_removable(self):
        """EP classify_weight and apply_ep_adapter should work standalone."""
        from vllm_mlx.ep_loader import classify_weight
        from vllm_mlx.ep_adapter import apply_ep_adapter

        # These are independent of any distributed runtime
        assert classify_weight("model.layers.0.mlp.gate.weight") == "replicate"
        assert classify_weight("model.layers.0.mlp.switch_mlp.w.weight") == "slice_expert"
