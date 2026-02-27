# SPDX-License-Identifier: Apache-2.0
"""EP-aware model loading for Expert Parallelism.

Loads the full model via mlx-lm, then slices routed expert weights
so each rank holds only its local expert subset. Attention, gate,
shared expert, embedding, and LM head weights are replicated.
"""

from __future__ import annotations

import logging
import math
from typing import Any

import mlx.core as mx

logger = logging.getLogger(__name__)


def classify_weight(key: str) -> str:
    """Classify a model weight key as 'replicate' or 'slice_expert'.

    Routed expert weights (inside switch_mlp) are sliced across ranks.
    Everything else (gate, shared_expert, attention, embedding, etc.)
    is replicated on all ranks.

    Args:
        key: Dot-separated weight name, e.g.
            ``"model.layers.0.mlp.switch_mlp.gate_proj.weight"``.

    Returns:
        ``"slice_expert"`` if the weight belongs to a routed expert,
        ``"replicate"`` otherwise.
    """
    # switch_mlp contains the routed expert parameters (stacked [E, ...])
    if ".switch_mlp." in key:
        return "slice_expert"
    return "replicate"


def slice_expert_weight(
    tensor: mx.array,
    rank: int,
    world_size: int,
    num_experts: int,
) -> mx.array:
    """Slice an expert-stacked tensor for the given rank.

    Expert weights are expected to have shape ``[E_total, ...]`` on
    dim-0.  This function extracts the contiguous block belonging to
    *rank*.

    Args:
        tensor: Full expert weight with shape ``[E_total, ...]``.
        rank: This process's rank.
        world_size: Total number of EP ranks.
        num_experts: Total number of experts (``config.num_experts``).

    Returns:
        Sliced tensor with shape ``[E_local, ...]``.
    """
    if tensor.shape[0] != num_experts:
        logger.warning(
            "Weight dim-0 (%d) != num_experts (%d), skipping slice",
            tensor.shape[0],
            num_experts,
        )
        return tensor

    e_local = num_experts // world_size
    start = rank * e_local
    end = start + e_local
    return tensor[start:end]


def ep_load(
    model_name: str,
    rank: int,
    world_size: int,
    tokenizer_config: dict[str, Any] | None = None,
) -> tuple[Any, Any]:
    """Load a model with EP expert slicing.

    1. Loads the full model using ``mlx_lm.load()``.
    2. Reads ``num_experts`` from the model config.
    3. Walks all parameters, slicing expert weights for this rank.

    Args:
        model_name: HuggingFace model name or local path.
        rank: This process's rank.
        world_size: Total number of EP ranks.
        tokenizer_config: Optional tokenizer configuration overrides.

    Returns:
        ``(model, tokenizer)`` tuple with expert weights sliced.
    """
    from mlx_lm import load

    logger.info("[Rank %d] EP loading model: %s", rank, model_name)

    # Load full model on every rank
    model, tokenizer = load(model_name, tokenizer_config=tokenizer_config or {})

    # Extract num_experts from model config
    num_experts = _get_num_experts(model)
    if num_experts is None:
        raise ValueError(
            f"Cannot determine num_experts from model config. "
            f"Model may not be a MoE model: {model_name}"
        )

    if num_experts % world_size != 0:
        raise ValueError(
            f"num_experts ({num_experts}) must be divisible by "
            f"world_size ({world_size})"
        )

    e_local = num_experts // world_size
    logger.info(
        "[Rank %d] EP slicing experts: E_total=%d, E_local=%d",
        rank,
        num_experts,
        e_local,
    )

    # Walk model parameters and slice expert weights
    sliced_count = 0
    for name, param in model.named_parameters():
        action = classify_weight(name)
        if action == "slice_expert":
            sliced = slice_expert_weight(param, rank, world_size, num_experts)
            _set_param_by_name(model, name, sliced)
            sliced_count += 1

    # Force evaluation of sliced weights
    mx.eval(model.parameters())

    logger.info(
        "[Rank %d] EP loading complete: %d weights sliced, E_local=%d",
        rank,
        sliced_count,
        e_local,
    )
    return model, tokenizer


def _get_num_experts(model: Any) -> int | None:
    """Extract num_experts from model config or architecture.

    Searches common config attribute names used by MoE models.
    """
    # Check model.config if available
    config = getattr(model, "config", None)
    if config is not None:
        for attr in ("num_experts", "num_local_experts", "n_routed_experts"):
            val = getattr(config, attr, None)
            if val is not None:
                return val

    # Check model.args (mlx-lm style)
    args = getattr(model, "args", None)
    if args is not None:
        for attr in ("num_experts", "num_local_experts", "n_routed_experts"):
            val = getattr(args, attr, None)
            if val is not None:
                return val

    # Try to infer from first MoE layer's switch_mlp
    if hasattr(model, "model") and hasattr(model.model, "layers"):
        for layer in model.model.layers:
            mlp = getattr(layer, "mlp", None)
            if mlp is not None:
                switch_mlp = getattr(mlp, "switch_mlp", None)
                if switch_mlp is not None:
                    ne = getattr(switch_mlp, "num_experts", None)
                    if ne is not None:
                        return ne

    return None


def _set_param_by_name(module: Any, name: str, value: mx.array) -> None:
    """Set a parameter on a module by dot-separated name.

    E.g. ``_set_param_by_name(model, "model.layers.0.mlp.switch_mlp.gate_proj.weight", tensor)``
    navigates to the correct submodule and sets the attribute.
    """
    parts = name.split(".")
    for part in parts[:-1]:
        module = getattr(module, part)
    setattr(module, parts[-1], value)
