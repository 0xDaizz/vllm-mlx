# SPDX-License-Identifier: Apache-2.0
"""EP MoE Adapter — wraps mlx-lm MoE modules for Expert Parallelism.

Replaces each MoE layer's ``mlp`` with an ``EPMoEAdapter`` that:
1. Runs the global gate (router) to get expert assignments
2. Uses ``mx.distributed.moe_dispatch_exchange`` to scatter tokens
3. Runs local expert FFN on received tokens
4. Uses ``mx.distributed.moe_combine_exchange`` to gather results
5. Adds the shared expert output (computed locally)
"""

from __future__ import annotations

import logging
import math
from typing import Any

import mlx.core as mx
import mlx.nn as nn

logger = logging.getLogger(__name__)


class EPMoEAdapter(nn.Module):
    """Adapter that wraps an mlx-lm MoE module for EP execution.

    The gate (router), shared expert, and switch_mlp are taken from
    the original MoE module. The switch_mlp holds only this rank's
    local expert subset (sliced by ``ep_loader``).

    Args:
        original_moe: The original MoE module (e.g. ``Qwen3_5MoeSparseMoeBlock``).
        ep_group: ``mx.distributed`` group for EP communication.
        rank: This process's rank.
        world_size: Total number of EP ranks.
        num_experts_total: Total number of experts across all ranks
            (``config.num_experts``).
        capacity_factor: Capacity factor for dispatch buffer sizing.
        kernel_backend: EP kernel backend (``"auto"``, ``"cpu"``, ``"metal"``).
    """

    def __init__(
        self,
        original_moe: nn.Module,
        ep_group: Any,
        rank: int,
        world_size: int,
        num_experts_total: int,
        capacity_factor: float = 1.25,
        kernel_backend: str = "auto",
    ):
        super().__init__()
        self.gate = original_moe.gate
        # shared_experts may be named differently across architectures
        self.shared_experts = getattr(original_moe, "shared_experts", None)
        if self.shared_experts is None:
            self.shared_experts = getattr(original_moe, "shared_expert", None)
        self.switch_mlp = original_moe.switch_mlp

        self.E_total = num_experts_total
        self.ep_group = ep_group
        self.rank = rank
        self.world_size = world_size
        self.capacity_factor = capacity_factor
        self.kernel_backend = kernel_backend

    def __call__(self, x: mx.array) -> mx.array:
        """EP-aware MoE forward pass.

        Args:
            x: Input tensor of shape ``[B, N, D]`` or ``[N, D]``.

        Returns:
            Output tensor of the same shape as *x*.
        """
        # 1. Global gate — replicated, same routing on all ranks
        indices, weights = self.gate(x)

        # 2. Compute dispatch capacity
        # N = number of tokens in this batch
        if x.ndim == 3:
            N = x.shape[0] * x.shape[1]
        else:
            N = x.shape[0]
        top_k = indices.shape[-1]
        capacity = max(
            1, math.ceil(N * top_k * self.capacity_factor / self.E_total)
        )

        # 3. Dispatch — scatter tokens to expert-owning ranks
        dispatched, route_idx = mx.distributed.moe_dispatch_exchange(
            x,
            indices,
            num_experts=self.E_total,
            capacity=capacity,
            group=self.ep_group,
            backend=self.kernel_backend,
        )

        # 4. Local expert FFN
        expert_out = self.switch_mlp(dispatched)

        # 5. Combine — gather results back to original positions
        routed_out = mx.distributed.moe_combine_exchange(
            expert_out,
            route_idx,
            weights,
            x,
            num_experts=self.E_total,
            capacity=capacity,
            group=self.ep_group,
            backend=self.kernel_backend,
        )

        # 6. Shared expert (local, no communication)
        if self.shared_experts is not None:
            shared_out = self.shared_experts(x)
            return routed_out + shared_out

        return routed_out


def apply_ep_adapter(
    model: Any,
    ep_group: Any,
    rank: int,
    world_size: int,
    num_experts_total: int,
    capacity_factor: float = 1.25,
    kernel_backend: str = "auto",
) -> int:
    """Replace all MoE layers in the model with EPMoEAdapter.

    Walks ``model.model.layers`` and replaces any layer whose ``mlp``
    has a ``switch_mlp`` attribute (indicating it's a MoE block).

    Args:
        model: The loaded model (e.g. from ``mlx_lm.load()``).
        ep_group: ``mx.distributed`` group for EP communication.
        rank: This process's rank.
        world_size: Total number of EP ranks.
        num_experts_total: Total number of experts across all ranks.
        capacity_factor: Capacity factor for dispatch buffer sizing.
        kernel_backend: EP kernel backend.

    Returns:
        Number of MoE layers replaced.
    """
    replaced = 0
    layers = getattr(getattr(model, "model", None), "layers", None)
    if layers is None:
        logger.warning("Model has no model.layers — cannot apply EP adapter")
        return 0

    for i, layer in enumerate(layers):
        mlp = getattr(layer, "mlp", None)
        if mlp is not None and hasattr(mlp, "switch_mlp"):
            adapter = EPMoEAdapter(
                original_moe=mlp,
                ep_group=ep_group,
                rank=rank,
                world_size=world_size,
                num_experts_total=num_experts_total,
                capacity_factor=capacity_factor,
                kernel_backend=kernel_backend,
            )
            layer.mlp = adapter
            replaced += 1

    logger.info(
        "[Rank %d] EP adapter applied to %d MoE layers (E_total=%d)",
        rank,
        replaced,
        num_experts_total,
    )
    return replaced
