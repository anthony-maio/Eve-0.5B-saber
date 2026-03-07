#!/bin/bash
# Eve-3-SABER-1B Single-Node Training Launch Script
# Usage: bash launch.sh

set -euo pipefail

NPROC="${NPROC:-8}"

echo "=== Eve-3-SABER-1B Training ==="
echo "  GPUs: ${NPROC}"
echo ""

# Install deps if needed
pip install -q torch transformers accelerate datasets safetensors \
    tokenizers wandb numpy tqdm huggingface_hub zstandard 2>/dev/null

# Try to install flash-attn (optional, will fallback to SDPA)
pip install -q flash-attn --no-build-isolation 2>/dev/null || true

# NCCL tuning
export NCCL_IB_DISABLE=0
export NCCL_NET_GDR_LEVEL=5
export NCCL_DEBUG=WARN

# Login to HuggingFace if token is set
if [ -n "${HF_TOKEN:-}" ]; then
    huggingface-cli login --token "$HF_TOKEN" 2>/dev/null || true
fi

# Login to WandB if key is set
if [ -n "${WANDB_API_KEY:-}" ]; then
    wandb login "$WANDB_API_KEY" 2>/dev/null || true
fi

# Launch training
torchrun \
    --nproc_per_node="$NPROC" \
    --nnodes=1 \
    --standalone \
    train_full.py
