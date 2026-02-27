# SPDX-License-Identifier: Apache-2.0
"""Custom MLX EP API availability check.

Validates that the custom MLX build (~/mlx) with Expert Parallelism
primitives is installed. Called at runtime on EP worker nodes only
(not on the development Mac).
"""

import sys

REQUIRED_EP_OPS = [
    "moe_dispatch_exchange",
    "moe_combine_exchange",
    "moe_ep_warmup",
    "moe_ep_stats",
]


def check_ep_api() -> None:
    """Verify that all required EP operations exist in mx.distributed.

    Raises SystemExit with installation instructions if any are missing.
    """
    import mlx.core as mx

    missing = [op for op in REQUIRED_EP_OPS if not hasattr(mx.distributed, op)]
    if missing:
        print(f"ERROR: mx.distributed is missing EP operations: {missing}")
        print("Install the custom MLX build: pip install -e ~/mlx")
        sys.exit(1)
    print(f"OK: MLX EP API available (mlx {mx.__version__})")
