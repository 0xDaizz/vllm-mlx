# MTP (Multi-Token Prediction) Speculative Decoding

MTP is a self-speculative decoding technique that uses lightweight prediction heads built into the model to propose draft tokens. Unlike n-gram or draft model approaches, MTP requires no external model — it reuses the target model's hidden states and built-in MTP weights.

## Supported Models

| Model Family | model_type | Status |
|---|---|---|
| DeepSeek V3 | `deepseek_v3` | Supported |
| DeepSeek V3.2 | `deepseek_v32` | Supported |
| GLM-5 (GLM MoE DSA) | `glm_moe_dsa` | Supported |
| GLM-4 MoE Lite | `glm4_moe_lite` | Supported |

## Requirements

**MTP weights must be present in the model checkpoint.** Standard quantized MLX conversions (e.g., `mlx-community/DeepSeek-V3-4bit`) do **not** include MTP weights — mlx-lm's `sanitize()` strips them during conversion.

You must use one of:
- The original HuggingFace checkpoint (FP8/BF16)
- A custom MLX conversion that preserves MTP weights

## Usage

```bash
vllm-mlx serve deepseek-ai/DeepSeek-V3-0324 \
    --continuous-batching \
    --speculative-method mtp \
    --num-speculative-tokens 1
```

### CLI Arguments

| Argument | Description | Default |
|---|---|---|
| `--speculative-method mtp` | Enable MTP speculative decoding | (disabled) |
| `--num-speculative-tokens N` | Number of draft tokens per step | 3 |
| `--spec-decode-disable-batch-size N` | Disable spec decode above this batch size | (none) |
| `--spec-decode-auto-disable-threshold F` | Auto-disable below this acceptance rate | 0.4 |

## How It Works

1. **Bootstrap**: On the first decode step, the model runs a normal forward pass. Hidden states are captured for the next step's MTP proposal.

2. **Propose**: The MTP module takes the target model's hidden states and the last predicted token, then:
   - Normalizes both via `enorm` and `hnorm` (RMSNorm)
   - Projects their concatenation via `eh_proj` (Linear)
   - Runs through a full decoder layer (same architecture as main model layers)
   - Applies the shared `lm_head` to produce logits
   - Greedily selects the draft token

3. **Verify**: The target model processes `[last_token, draft_1, ..., draft_k]` in a single forward pass, producing logits for all positions.

4. **Accept/Reject**: Greedy rejection sampling compares target model's argmax with draft tokens. Accepted tokens are committed; rejected ones cause KV cache rollback.

5. **Hidden State Update**: After rejection sampling, the hidden state at the accepted position is stored for the next step's MTP proposal.

## Architecture

```
Target Model Hidden States ──┐
                              ├──> hnorm ──┐
                              │            ├──> concat ──> eh_proj ──> DecoderLayer ──> lm_head ──> draft token
Last Predicted Token ─────────┤            │
                              └──> embed ──> enorm ──┘
```

The MTP module shares `embed_tokens` and `lm_head` with the target model (no additional memory for these).

## Limitations

- **Single node only**: MTP does not yet support Tensor Parallel (TP) mode. Hidden state passing across TP ranks is planned for a future release.
- **Quantized models**: Standard mlx-community quantized models do not include MTP weights. You need the original checkpoint.
- **k=1 recommended**: Most MTP-equipped models have `num_nextn_predict_layers=1`, which means they predict only 1 token ahead. Using `--num-speculative-tokens 1` is recommended.
- **Memory**: Loading MTP weights adds one additional decoder layer to memory usage (including MoE experts if the model uses them).

## File Structure

```
vllm_mlx/spec_decode/
├── mtp_module.py              # Abstract MTP base class + weight loading
├── mtp_proposer.py            # MTPProposer implementation
└── mtp_adapters/
    ├── __init__.py             # Adapter registry
    └── deepseek_v3_mtp.py     # DeepSeek V3/V3.2/GLM-5 adapter
```
