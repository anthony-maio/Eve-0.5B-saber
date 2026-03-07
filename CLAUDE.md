# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Eve-3-SABER-1B is a 1-billion-parameter decoder-only language model with three novel architectural components built on a standard transformer skeleton. It is a HuggingFace-compatible model requiring `trust_remote_code=True`. The model uses GPT-2's tokenizer (vocab 50,257) and is designed for the HuggingFace ecosystem (Trainer, SFTTrainer, PEFT, pipelines).

## Commands

```bash
# Install dependencies
pip install -r requirements.txt

# Count parameters (verify 1B target)
python param_counter.py
python param_counter.py --tune-dff          # binary-search d_ff to hit exactly 1.0B
python param_counter.py --dff 2855          # count with specific d_ff

# Run ablation study
python ablation_runner.py --ablation all --dry-run   # validate configs without training
python ablation_runner.py --ablation baseline         # single ablation run
python ablation_runner.py --ablation all --tokens 1e9 --wandb  # full 5-run suite with logging

# Resume a specific ablation run
python ablation_runner.py --ablation anchors --resume-from ./ablation_results/anchors/checkpoint-last
```

## Architecture

The model is split across two core files that HuggingFace auto-loads via `trust_remote_code`:

- **configuration_saber.py** — `SABERConfig` (extends `PretrainedConfig`, `model_type = "saber"`)
- **modeling_saber.py** — Full model implementation

### Key dimensions

| Parameter | Value |
|-----------|-------|
| d_model | 2048 |
| n_layers | 24 |
| n_heads | 16 |
| head_dim | 128 |
| d_ff | 2855 (tuned to hit 1B params) |
| Total params | 1,000,001,548 |

### Three novel components (~73M params total, ~7.3% of model)

1. **Slip-Anchors** (`SlipAnchors` class, ~7.2M params) — Learnable codebook of 64 prototypes (d_anchor=128) that biases K and V *after* RoPE. Applied in all 24 layers. Controlled by `config.enable_anchors`.

2. **Experience Stream** (`ExperienceStream` class, ~15.8M params) — Per-token state (d_exp=256) flowing layer-to-layer (NOT token-to-token, NOT an RNN). Includes curiosity auxiliary loss with stop-gradient. Applied in all 24 layers. Controlled by `config.enable_experience`.

3. **Resonant FFN** (`ResonantFFN` class, ~50.4M params) — Sinusoidal modulation of SwiGLU output with learned alpha blend (init ~0.95 = near-pure SwiGLU). Applied on 12 even-indexed layers (0,2,4,...,22). Controlled by `config.resonant_layers` (empty list = disabled).

### Model class hierarchy (modeling_saber.py)

```
SABERForCausalLM (top-level, registered for AutoModelForCausalLM)
  └── SABERModel (base model: embeddings -> blocks -> final norm)
        └── SABERBlock x24 (pre-norm transformer block)
              ├── SABERAttention (QKV + RoPE + SlipAnchors + SDPA)
              │     ├── SABERRotaryEmbedding
              │     └── SlipAnchors
              ├── StandardFFN or ResonantFFN (alternating by layer index)
              └── ExperienceStream
```

### Loss computation

`SABERForCausalLM.forward()` returns: `loss = CrossEntropy + config.curiosity_coeff * mean_curiosity_loss`

The curiosity loss is averaged across all 24 layers, weighted by `curiosity_coeff` (default 0.01).

### Ablation system

Component toggling uses config flags (`enable_anchors`, `enable_experience`) and `resonant_layers=[]` rather than zeroing dimensions, to avoid tensor shape mismatches. The five ablation runs are: baseline, +anchors, +experience, +resonant, full.

## Important design decisions

- Slip-anchors score from raw hidden state `h` (not Q) to avoid circularity
- RoPE is applied *before* anchor bias to prevent positional corruption of the bias
- Stop-gradient on `s` in curiosity loss is critical to prevent representational collapse
- The LM head is weight-tied to the embedding table
- All attention uses `F.scaled_dot_product_attention` for FlashAttention-2 compatibility
- The `d_ff` spec value of 5461 yields ~1.38B params; 2855 is the tuned value for exactly 1B
