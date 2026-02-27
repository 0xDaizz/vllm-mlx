# SPDX-License-Identifier: Apache-2.0
"""Tests for vllm_mlx.ep_loader module."""

import pytest

from vllm_mlx.ep_loader import classify_weight, slice_expert_weight


class TestClassifyWeight:
    """Tests for weight classification."""

    def test_switch_mlp_gate_proj(self):
        key = "model.layers.0.mlp.switch_mlp.gate_proj.weight"
        assert classify_weight(key) == "slice_expert"

    def test_switch_mlp_up_proj(self):
        key = "model.layers.5.mlp.switch_mlp.up_proj.weight"
        assert classify_weight(key) == "slice_expert"

    def test_switch_mlp_down_proj(self):
        key = "model.layers.10.mlp.switch_mlp.down_proj.weight"
        assert classify_weight(key) == "slice_expert"

    def test_gate_replicated(self):
        key = "model.layers.0.mlp.gate.weight"
        assert classify_weight(key) == "replicate"

    def test_shared_expert_replicated(self):
        key = "model.layers.0.mlp.shared_experts.gate_proj.weight"
        assert classify_weight(key) == "replicate"

    def test_attention_replicated(self):
        key = "model.layers.0.self_attn.q_proj.weight"
        assert classify_weight(key) == "replicate"

    def test_embedding_replicated(self):
        key = "model.embed_tokens.weight"
        assert classify_weight(key) == "replicate"

    def test_lm_head_replicated(self):
        key = "lm_head.weight"
        assert classify_weight(key) == "replicate"

    def test_layer_norm_replicated(self):
        key = "model.layers.0.input_layernorm.weight"
        assert classify_weight(key) == "replicate"


class TestSliceExpertWeight:
    """Tests for expert weight slicing."""

    def test_basic_slice_rank0(self):
        import mlx.core as mx

        # 8 experts, 2 ranks → 4 experts per rank
        tensor = mx.ones((8, 64, 128))
        sliced = slice_expert_weight(tensor, rank=0, world_size=2, num_experts=8)
        assert sliced.shape == (4, 64, 128)

    def test_basic_slice_rank1(self):
        import mlx.core as mx

        tensor = mx.arange(8).reshape(8, 1)
        sliced = slice_expert_weight(tensor, rank=1, world_size=2, num_experts=8)
        assert sliced.shape == (4, 1)
        # Rank 1 should get experts [4, 5, 6, 7]
        expected = mx.arange(4, 8).reshape(4, 1)
        assert mx.array_equal(sliced, expected)

    def test_single_rank(self):
        import mlx.core as mx

        # Single rank keeps all experts
        tensor = mx.ones((16, 32))
        sliced = slice_expert_weight(tensor, rank=0, world_size=1, num_experts=16)
        assert sliced.shape == (16, 32)

    def test_dim0_mismatch_returns_original(self):
        import mlx.core as mx

        # If dim-0 doesn't match num_experts, return unchanged
        tensor = mx.ones((10, 32))
        sliced = slice_expert_weight(tensor, rank=0, world_size=2, num_experts=8)
        assert sliced.shape == (10, 32)  # Unchanged

    def test_many_experts(self):
        import mlx.core as mx

        # 512 experts, 2 ranks → 256 per rank
        tensor = mx.zeros((512, 16))
        sliced = slice_expert_weight(tensor, rank=0, world_size=2, num_experts=512)
        assert sliced.shape == (256, 16)

    def test_four_ranks(self):
        import mlx.core as mx

        # 512 experts, 4 ranks → 128 per rank
        tensor = mx.arange(512).reshape(512, 1)
        sliced = slice_expert_weight(tensor, rank=2, world_size=4, num_experts=512)
        assert sliced.shape == (128, 1)
        # Rank 2 should get experts [256..383]
        assert sliced[0, 0].item() == 256
        assert sliced[-1, 0].item() == 383
