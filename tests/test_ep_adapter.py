# SPDX-License-Identifier: Apache-2.0
"""Tests for vllm_mlx.ep_adapter module."""

import pytest

from vllm_mlx.ep_adapter import EPMoEAdapter, apply_ep_adapter


class TestEPMoEAdapterInit:
    """Test EPMoEAdapter initialization."""

    def test_attributes_stored(self):
        """EPMoEAdapter should store all configuration."""
        import mlx.nn as nn

        class FakeGate(nn.Module):
            pass

        class FakeSwitchMLP(nn.Module):
            pass

        class FakeMoE(nn.Module):
            def __init__(self):
                super().__init__()
                self.gate = FakeGate()
                self.switch_mlp = FakeSwitchMLP()
                self.shared_experts = None

        moe = FakeMoE()
        adapter = EPMoEAdapter(
            original_moe=moe,
            ep_group=None,  # No real group in unit test
            rank=0,
            world_size=2,
            num_experts_total=64,
            capacity_factor=1.5,
            kernel_backend="cpu",
        )

        assert adapter.E_total == 64
        assert adapter.rank == 0
        assert adapter.world_size == 2
        assert adapter.capacity_factor == 1.5
        assert adapter.kernel_backend == "cpu"
        assert adapter.gate is moe.gate
        assert adapter.switch_mlp is moe.switch_mlp
        assert adapter.shared_experts is None

    def test_shared_expert_variants(self):
        """EPMoEAdapter should find shared_experts or shared_expert."""
        import mlx.nn as nn

        class FakeSharedExpert(nn.Module):
            pass

        # Test with shared_expert (singular)
        class FakeMoE1(nn.Module):
            def __init__(self):
                super().__init__()
                self.gate = nn.Linear(64, 8)
                self.switch_mlp = nn.Linear(64, 64)
                self.shared_expert = FakeSharedExpert()

        moe1 = FakeMoE1()
        adapter1 = EPMoEAdapter(
            moe1, ep_group=None, rank=0, world_size=2, num_experts_total=8
        )
        assert adapter1.shared_experts is not None

        # Test with shared_experts (plural)
        class FakeMoE2(nn.Module):
            def __init__(self):
                super().__init__()
                self.gate = nn.Linear(64, 8)
                self.switch_mlp = nn.Linear(64, 64)
                self.shared_experts = FakeSharedExpert()

        moe2 = FakeMoE2()
        adapter2 = EPMoEAdapter(
            moe2, ep_group=None, rank=0, world_size=2, num_experts_total=8
        )
        assert adapter2.shared_experts is not None


class TestApplyEPAdapter:
    """Test apply_ep_adapter function."""

    def test_replaces_moe_layers(self):
        """apply_ep_adapter should replace layers with switch_mlp."""
        import mlx.nn as nn

        class FakeMoE(nn.Module):
            def __init__(self):
                super().__init__()
                self.gate = nn.Linear(64, 8)
                self.switch_mlp = nn.Linear(64, 64)

        class FakeLayer(nn.Module):
            def __init__(self, has_moe=False):
                super().__init__()
                self.mlp = FakeMoE() if has_moe else nn.Linear(64, 64)

        class FakeModelInner(nn.Module):
            def __init__(self):
                super().__init__()
                self.layers = [
                    FakeLayer(has_moe=True),
                    FakeLayer(has_moe=False),  # Dense layer
                    FakeLayer(has_moe=True),
                ]

        class FakeModel(nn.Module):
            def __init__(self):
                super().__init__()
                self.model = FakeModelInner()

        model = FakeModel()
        replaced = apply_ep_adapter(
            model,
            ep_group=None,
            rank=0,
            world_size=2,
            num_experts_total=8,
        )

        assert replaced == 2  # Only 2 MoE layers
        assert isinstance(model.model.layers[0].mlp, EPMoEAdapter)
        assert not isinstance(model.model.layers[1].mlp, EPMoEAdapter)  # Dense
        assert isinstance(model.model.layers[2].mlp, EPMoEAdapter)

    def test_no_layers_returns_zero(self):
        """Model without model.layers should return 0."""
        import mlx.nn as nn

        class FakeModel(nn.Module):
            pass

        model = FakeModel()
        replaced = apply_ep_adapter(
            model,
            ep_group=None,
            rank=0,
            world_size=2,
            num_experts_total=8,
        )
        assert replaced == 0
