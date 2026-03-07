#!/bin/bash
# Eve-3-SABER-1B Multi-Node Training Launch Script
# Run this on BOTH nodes. Set MASTER_ADDR before running.
#
# Usage:
#   Node 0 (master): MASTER_ADDR=<node0-ip> NODE_RANK=0 bash launch_multinode.sh
#   Node 1:          MASTER_ADDR=<node0-ip> NODE_RANK=1 bash launch_multinode.sh
#
# On RunPod, MASTER_ADDR is typically set automatically, or use the
# internal IP of the master pod.

set -euo pipefail

MASTER_ADDR="${MASTER_ADDR:?Set MASTER_ADDR to the IP of node 0}"
MASTER_PORT="${MASTER_PORT:-29500}"
NODE_RANK="${NODE_RANK:?Set NODE_RANK (0 for master, 1 for worker)}"
NNODES="${NNODES:-2}"
NPROC="${NPROC:-8}"

echo "=== Eve-3-SABER-1B Training ==="
echo "  Master: ${MASTER_ADDR}:${MASTER_PORT}"
echo "  Node rank: ${NODE_RANK} / ${NNODES}"
echo "  GPUs per node: ${NPROC}"
echo "  Total GPUs: $((NNODES * NPROC))"
echo ""

# Install deps if needed
pip install -q torch transformers accelerate datasets safetensors \
    tokenizers wandb numpy tqdm huggingface_hub zstandard 2>/dev/null

# Try to install flash-attn (optional, will fallback to SDPA)
pip install -q flash-attn --no-build-isolation 2>/dev/null || true

# Set NCCL env vars for optimal multi-node performance
export NCCL_IB_DISABLE=0          # Enable InfiniBand
export NCCL_NET_GDR_LEVEL=5       # GPUDirect RDMA
export NCCL_IB_GID_INDEX=3        # InfiniBand GID
export NCCL_SOCKET_IFNAME=eth0    # Network interface (adjust if needed)
export NCCL_DEBUG=WARN            # Set to INFO for debugging

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
    --nnodes="$NNODES" \
    --node_rank="$NODE_RANK" \
    --master_addr="$MASTER_ADDR" \
    --master_port="$MASTER_PORT" \
    train_full.py
