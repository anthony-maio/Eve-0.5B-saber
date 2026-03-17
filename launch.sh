#!/bin/bash
# Eve-3-SABER-0.5B Training Launch Script
# Usage: bash launch.sh
# NOTE: Do NOT install torch on RunPod — it ships with a working CUDA build.

set -euo pipefail

echo "=== Eve-3-SABER-0.5B Training ==="
echo ""

# Install deps (NO torch — RunPod pre-installs it with correct CUDA)
pip install -q transformers accelerate datasets safetensors \
    tokenizers wandb numpy tqdm huggingface_hub zstandard 2>/dev/null

# Login to HuggingFace if token is set
if [ -n "${HF_TOKEN:-}" ]; then
    huggingface-cli login --token "$HF_TOKEN" 2>/dev/null || true
fi

# Login to WandB if key is set
if [ -n "${WANDB_API_KEY:-}" ]; then
    wandb login "$WANDB_API_KEY" 2>/dev/null || true
fi

# Launch training (single GPU — no torchrun needed)
python train_full.py
