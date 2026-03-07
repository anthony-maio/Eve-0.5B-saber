#!/usr/bin/env python3
"""Eve-3-SABER-1B: 20B-token 3-stage curriculum pretraining.
Launch: bash launch.sh
"""

# ============================================================
# CELL 2: IMPORTS & HARDWARE DETECTION
# ============================================================
import os, sys, json, math, time, random, shutil, logging
from pathlib import Path
from typing import Optional, Dict, List, Tuple, Any

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, IterableDataset

import numpy as np
from tqdm.auto import tqdm

from transformers import GPT2Tokenizer
from datasets import load_dataset
from accelerate import Accelerator
from accelerate.utils import set_seed
import wandb

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s", datefmt="%H:%M:%S")
logger = logging.getLogger("train_full")
logging.getLogger("httpx").setLevel(logging.WARNING)
logging.getLogger("httpcore").setLevel(logging.WARNING)

# ---- Hardware detection ----
def detect_hardware():
    if not torch.cuda.is_available():
        return "cpu", 0, False
    gpu_name = torch.cuda.get_device_name(0).lower()
    vram_gb = torch.cuda.get_device_properties(0).total_memory / 1024**3
    if "h100" in gpu_name:
        return "h100", vram_gb, True
    elif "a100" in gpu_name:
        return "a100", vram_gb, True
    elif "3090" in gpu_name or "4090" in gpu_name:
        return "3090", vram_gb, True
    elif "l4" in gpu_name or "t4" in gpu_name:
        return "l4_t4", vram_gb, True
    return "unknown", vram_gb, vram_gb < 32

GPU_TIER, GPU_VRAM_GB, _ = detect_hardware()
N_GPUS = torch.cuda.device_count() if torch.cuda.is_available() else 0

if torch.cuda.is_available():
    for i in range(N_GPUS):
        name = torch.cuda.get_device_name(i)
        vram = torch.cuda.get_device_properties(i).total_memory / 1024**3
        print(f"  GPU {i}: {name} ({vram:.0f} GB)")

print(f"\nGPU tier: {GPU_TIER}, VRAM: {GPU_VRAM_GB:.1f} GB, GPUs: {N_GPUS}")
print(f"PyTorch: {torch.__version__}, CUDA: {torch.version.cuda}")

# ============================================================
# CELL 3: TRAINING CONFIGURATION
# ============================================================

# ---- Paths ----
MODEL_NAME = "eve-3-saber-1b"
HF_REPO    = "anthonym21/Eve-3-SABER-1B"
MODEL_DIR  = Path("./eve-3-saber-1b")
CKPT_DIR   = Path("./checkpoints")
LOG_DIR    = Path("./logs")
for _d in [MODEL_DIR, CKPT_DIR, LOG_DIR]:
    _d.mkdir(parents=True, exist_ok=True)

# ---- Training ----
TOTAL_TOKENS          = 20_000_000_000    # 20B
SEQ_LEN               = 2048
EFFECTIVE_BATCH_TOKENS = 500_000          # ~0.5M tokens per optimizer step
SEED                  = 42

# Auto-size micro batch by GPU tier
_MICRO_BATCH_MAP = {"h100": 6, "a100": 4, "3090": 1, "l4_t4": 1, "unknown": 1, "cpu": 1}
MICRO_BATCH = _MICRO_BATCH_MAP.get(GPU_TIER, 1)

# Gradient accumulation to hit effective batch
_world_size = max(N_GPUS, 1)
GRAD_ACCUM = max(1, EFFECTIVE_BATCH_TOKENS // (MICRO_BATCH * _world_size * SEQ_LEN))
_actual_batch_tokens = MICRO_BATCH * _world_size * GRAD_ACCUM * SEQ_LEN
TOTAL_STEPS = TOTAL_TOKENS // _actual_batch_tokens

# ---- Optimizer ----
LR           = 3e-4
LR_MIN       = 3e-5
BETAS        = (0.9, 0.95)
WEIGHT_DECAY = 0.1
GRAD_CLIP    = 1.0
WARMUP_STEPS = 2000

# ---- Loss ----
CURIOSITY_WEIGHT = 0.01

# ---- Precision ----
DTYPE_STR   = "bf16"
DTYPE_TORCH = torch.bfloat16

# ---- Gradient checkpointing ----
GRADIENT_CHECKPOINTING = (GPU_TIER in ("3090", "l4_t4", "unknown", "cpu"))

# ---- Checkpointing ----
CHECKPOINT_EVERY = 1000
KEEP_LAST_N      = 3
LOG_EVERY        = 10
EVAL_EVERY       = 500

# ---- WandB ----
WANDB_PROJECT  = "eve-3-saber-1b"
WANDB_ENTITY   = None
WANDB_RUN_NAME = f"{MODEL_NAME}-full-{N_GPUS}gpu"

# ===========================================================
# 3-STAGE CURRICULUM DATA CONFIGURATION
# ===========================================================
# Stage 1: Foundation (10B tokens) — build language + code fundamentals
# Stage 2: Specialization (6B tokens) — heavy code + technical docs
# Stage 3: Anneal (4B tokens) — highest quality, LR → 0

STAGE_1_TOKENS = 10_000_000_000   # 10B
STAGE_2_TOKENS =  6_000_000_000   #  6B
STAGE_3_TOKENS =  4_000_000_000   #  4B

# Token boundaries (cumulative)
STAGE_1_END = STAGE_1_TOKENS                         # 10B
STAGE_2_END = STAGE_1_TOKENS + STAGE_2_TOKENS        # 16B
STAGE_3_END = TOTAL_TOKENS                           # 20B

# Step boundaries
STAGE_1_STEPS = STAGE_1_TOKENS // _actual_batch_tokens
STAGE_2_STEPS = STAGE_2_TOKENS // _actual_batch_tokens
STAGE_3_STEPS = STAGE_3_TOKENS // _actual_batch_tokens

# Format: (dataset_id, config_or_split, weight, streaming)
DATA_MIX_STAGE1 = [
    ("HuggingFaceFW/fineweb-edu", "sample-350BT", 0.35, True),
    ("bigcode/starcoderdata",     "default",       0.35, True),
    ("mlfoundations/dclm-baseline-1.0", None,        0.15, True),
    ("open-web-math/open-web-math", "default",     0.10, True),
    ("HuggingFaceTB/cosmopedia",  "web_samples_v2", 0.05, True),
]

DATA_MIX_STAGE2 = [
    ("bigcode/starcoderdata",     "default",       0.45, True),
    ("HuggingFaceFW/fineweb-edu", "sample-10BT",   0.15, True),
    ("HuggingFaceTB/finemath",    "finemath-4+",   0.10, True),
    ("HuggingFaceTB/cosmopedia",  "web_samples_v2", 0.10, True),
    ("HuggingFaceFW/finepdfs",    "default",       0.20, True),
]

DATA_MIX_STAGE3 = [
    ("HuggingFaceTB/smollm-corpus", "python-edu",  0.35, True),
    ("HuggingFaceTB/finemath",    "finemath-4+",   0.20, True),
    ("HuggingFaceFW/finepdfs",    "default",       0.15, True),
    ("HuggingFaceTB/cosmopedia",  "web_samples_v2", 0.15, True),
    ("wikimedia/wikipedia",       "20231101.en",   0.10, True),
    ("mlfoundations/dclm-baseline-1.0", None,        0.05, True),
]

STAGES = [
    {"name": "Stage 1: Foundation",     "tokens": STAGE_1_TOKENS, "steps": STAGE_1_STEPS, "data_mix": DATA_MIX_STAGE1, "lr_schedule": "cosine"},
    {"name": "Stage 2: Specialization", "tokens": STAGE_2_TOKENS, "steps": STAGE_2_STEPS, "data_mix": DATA_MIX_STAGE2, "lr_schedule": "cosine"},
    {"name": "Stage 3: Anneal",         "tokens": STAGE_3_TOKENS, "steps": STAGE_3_STEPS, "data_mix": DATA_MIX_STAGE3, "lr_schedule": "linear_decay"},
]

# ---- Print config ----
print(f"Batch config: micro={MICRO_BATCH}, world={_world_size}, accum={GRAD_ACCUM}")
print(f"Effective batch: {_actual_batch_tokens:,} tokens/step")
print(f"Total steps: {TOTAL_STEPS:,}")
print(f"\n3-Stage Curriculum:")
cumulative_steps = 0
for s in STAGES:
    cumulative_steps += s["steps"]
    print(f"  {s['name']}: {s['tokens']/1e9:.0f}B tokens, {s['steps']:,} steps (cumulative: {cumulative_steps:,}), LR: {s['lr_schedule']}")
print(f"\nGradient checkpointing: {GRADIENT_CHECKPOINTING}")

set_seed(SEED)

# ============================================================
# CELL 4: MODEL & TOKENIZER
# ============================================================
from configuration_saber import SABERConfig
from modeling_saber import SABERForCausalLM

# Tokenizer
tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
tokenizer.pad_token = tokenizer.eos_token
EOS_ID = tokenizer.eos_token_id
print(f"Tokenizer: vocab={tokenizer.vocab_size}, eos={EOS_ID}")

# Model (full config — all components enabled)
config = SABERConfig(d_ff=2855)
model = SABERForCausalLM(config)

n_params = sum(p.numel() for p in model.parameters())
print(f"Model: {n_params:,} parameters ({n_params/1e9:.4f}B)")

if GRADIENT_CHECKPOINTING:
    model.gradient_checkpointing_enable()
    print("Gradient checkpointing enabled")

# ============================================================
# CELL 5: DATASET — STREAMING + SEQUENCE PACKING
# ============================================================

def tokenise_example(example):
    text = (example.get("text") or example.get("content") or
            example.get("passage") or example.get("document") or "")
    if not isinstance(text, str) or len(text.strip()) < 50:
        return None
    ids = tokenizer.encode(text, add_special_tokens=False, truncation=False)
    return ids + [EOS_ID]


class PackedDataset(IterableDataset):
    def __init__(self, hf_datasets_weighted, seq_len=SEQ_LEN, seed=SEED, max_tokens=None):
        self.datasets = hf_datasets_weighted
        self.seq_len = seq_len
        self.seed = seed
        self.max_tokens = max_tokens

    def _interleave_streams(self):
        rng = random.Random(self.seed)
        iterators = [iter(ds) for ds, _ in self.datasets]
        weights = [w for _, w in self.datasets]
        while True:
            idx = rng.choices(range(len(iterators)), weights=weights, k=1)[0]
            try:
                yield next(iterators[idx])
            except StopIteration:
                iterators[idx] = iter(self.datasets[idx][0])
                yield next(iterators[idx])

    def __iter__(self):
        buffer = []
        tokens_emitted = 0
        for example in self._interleave_streams():
            if self.max_tokens and tokens_emitted >= self.max_tokens:
                break
            ids = tokenise_example(example)
            if ids is None:
                continue
            buffer.extend(ids)
            while len(buffer) >= self.seq_len:
                chunk = buffer[:self.seq_len]
                buffer = buffer[self.seq_len:]
                input_ids = torch.tensor(chunk, dtype=torch.long)
                yield {"input_ids": input_ids, "labels": input_ids.clone()}
                tokens_emitted += self.seq_len


def build_streaming_datasets(data_mix, tag="train"):
    loaded = []
    total_weight = sum(w for _, _, w, _ in data_mix)
    for ds_id, cfg, weight, streaming in data_mix:
        try:
            logger.info(f"Loading {ds_id}/{cfg} (weight={weight/total_weight:.0%})...")
            ds = load_dataset(ds_id, cfg, split=tag, streaming=streaming) if cfg else load_dataset(ds_id, split=tag, streaming=streaming)
            ds = ds.shuffle(seed=SEED, buffer_size=10_000)
            loaded.append((ds, weight / total_weight))
        except Exception as e:
            logger.warning(f"Failed to load {ds_id}/{cfg}: {e}")
    # Re-normalise
    total_w = sum(w for _, w in loaded)
    loaded = [(ds, w / total_w) for ds, w in loaded]
    return loaded


def build_stage_dataloader(stage_idx, seed_offset=0):
    """Build a DataLoader for a specific curriculum stage."""
    stage = STAGES[stage_idx]
    logger.info(f"\n{'='*58}")
    logger.info(f"  Building dataloader: {stage['name']}")
    logger.info(f"  Tokens: {stage['tokens']/1e9:.0f}B | Steps: {stage['steps']:,}")
    logger.info(f"{'='*58}")

    datasets = build_streaming_datasets(stage["data_mix"], tag="train")
    packed = PackedDataset(
        hf_datasets_weighted=datasets,
        seq_len=SEQ_LEN,
        seed=SEED + seed_offset,
        max_tokens=stage["tokens"],
    )
    loader = DataLoader(
        packed, batch_size=MICRO_BATCH,
        num_workers=4, pin_memory=True, prefetch_factor=2,
    )
    return loader


# Build Stage 1 dataloader (stages 2 & 3 built on-the-fly during training)
logger.info("Building Stage 1 dataloader...")
stage1_loader = build_stage_dataloader(0)
print("Stage 1 dataloader ready.")

# ============================================================
# CELL 6: TRAINING UTILITIES (LR, checkpoints, metrics, WandB)
# ============================================================

# ---- LR schedule: 3-stage curriculum ----
# Stages 1-2: cosine with linear warmup (LR → LR_MIN)
# Stage 3: linear decay from LR_MIN → 0

def get_lr(global_step):
    """Compute LR based on global step across all 3 stages."""
    stages_12_steps = STAGE_1_STEPS + STAGE_2_STEPS

    if global_step < stages_12_steps:
        # Stages 1-2: warmup then cosine decay
        if global_step < WARMUP_STEPS:
            return LR * global_step / max(1, WARMUP_STEPS)
        progress = (global_step - WARMUP_STEPS) / max(1, stages_12_steps - WARMUP_STEPS)
        cosine = 0.5 * (1.0 + math.cos(math.pi * progress))
        return LR_MIN + (LR - LR_MIN) * cosine
    else:
        # Stage 3: linear decay from LR_MIN → 0
        stage3_step = global_step - stages_12_steps
        return LR_MIN * max(0.0, 1.0 - stage3_step / max(1, STAGE_3_STEPS))

def set_lr(optimizer, lr):
    for group in optimizer.param_groups:
        group["lr"] = lr

def get_current_stage(global_step):
    """Return stage index (0, 1, 2) based on global step."""
    if global_step < STAGE_1_STEPS:
        return 0
    elif global_step < STAGE_1_STEPS + STAGE_2_STEPS:
        return 1
    else:
        return 2

# ---- Custom metrics ----
def compute_anchor_entropy(anchor_scores_list):
    if not anchor_scores_list:
        return 0.0
    entropies = []
    for scores in anchor_scores_list:
        ent = -(scores * (scores + 1e-9).log()).sum(-1)
        entropies.append(ent.mean().item())
    return float(np.mean(entropies))

def compute_experience_magnitude(exp_state):
    if exp_state is None:
        return 0.0
    return exp_state.float().norm(dim=-1).mean().item()

# ---- WandB ----
def init_wandb(resume_run_id=None):
    run = wandb.init(
        project=WANDB_PROJECT, entity=WANDB_ENTITY,
        name=WANDB_RUN_NAME if not resume_run_id else None,
        id=resume_run_id, resume="allow" if resume_run_id else None,
        config={
            "d_model": config.d_model, "n_layers": config.n_layers,
            "n_heads": config.n_heads, "d_ff": config.d_ff,
            "d_exp": config.d_exp, "n_anchors": config.n_anchors,
            "total_tokens": TOTAL_TOKENS, "seq_len": SEQ_LEN,
            "micro_batch": MICRO_BATCH, "grad_accum": GRAD_ACCUM,
            "effective_batch_tokens": _actual_batch_tokens,
            "lr": LR, "lr_min": LR_MIN, "warmup_steps": WARMUP_STEPS,
            "weight_decay": WEIGHT_DECAY, "dtype": DTYPE_STR,
            "gpu_tier": GPU_TIER, "n_gpus": N_GPUS,
            "stage_1_tokens": STAGE_1_TOKENS,
            "stage_2_tokens": STAGE_2_TOKENS,
            "stage_3_tokens": STAGE_3_TOKENS,
            "focus": "code + tool use",
        },
    )
    logger.info(f"WandB run: {run.url}")
    return run

# ---- Checkpoint save/load ----
def save_checkpoint(model, optimizer, step, tokens_seen, metrics, wandb_run_id=None, stage_idx=0):
    ckpt_path = CKPT_DIR / f"step_{step:08d}"
    ckpt_path.mkdir(parents=True, exist_ok=True)

    torch.save(model.state_dict(), ckpt_path / "model.pt")
    torch.save(optimizer.state_dict(), ckpt_path / "optimizer.pt")

    meta = {
        "step": step, "tokens_seen": tokens_seen,
        "stage_idx": stage_idx,
        "metrics": metrics, "wandb_run_id": wandb_run_id,
    }
    with open(ckpt_path / "meta.json", "w") as f:
        json.dump(meta, f, indent=2)
    with open(CKPT_DIR / "latest.txt", "w") as f:
        f.write(str(ckpt_path))

    logger.info(f"[Step {step}] Checkpoint saved -> {ckpt_path}")

    # Prune old checkpoints
    all_ckpts = sorted(CKPT_DIR.glob("step_*"), key=lambda p: int(p.name.split("_")[1]))
    for old in all_ckpts[:-KEEP_LAST_N]:
        shutil.rmtree(old)
        logger.info(f"  Pruned: {old.name}")


def load_checkpoint(model, optimizer=None, ckpt_path=None):
    if ckpt_path is None:
        latest_file = CKPT_DIR / "latest.txt"
        if not latest_file.exists():
            logger.info("No checkpoint found. Starting from scratch.")
            return {"step": 0, "tokens_seen": 0, "stage_idx": 0, "wandb_run_id": None}
        ckpt_path = Path(latest_file.read_text().strip())

    logger.info(f"Loading checkpoint from {ckpt_path}...")
    state_dict = torch.load(ckpt_path / "model.pt", map_location="cpu")
    model.load_state_dict(state_dict)

    if optimizer is not None and (ckpt_path / "optimizer.pt").exists():
        opt_state = torch.load(ckpt_path / "optimizer.pt", map_location="cpu")
        optimizer.load_state_dict(opt_state)

    with open(ckpt_path / "meta.json") as f:
        meta = json.load(f)
    logger.info(f"  Resuming from step {meta['step']:,} | tokens {meta['tokens_seen']:,} | stage {meta.get('stage_idx', 0)}")
    return meta

print("Training utilities ready.")

# ============================================================
# CELL 7: PRETRAINING LOOP (3-STAGE CURRICULUM)
# ============================================================

def train(model, initial_loader, resume=True):
    """
    Main pretraining loop with 3-stage curriculum and Accelerate DDP.

    Stage transitions happen automatically at the step boundaries:
      - Stage 1 → 2 at STAGE_1_STEPS
      - Stage 2 → 3 at STAGE_1_STEPS + STAGE_2_STEPS

    At each transition, the dataloader is rebuilt with the new data mix.
    LR follows cosine decay for stages 1-2 and linear→0 for stage 3.
    """

    accelerator = Accelerator(
        mixed_precision=DTYPE_STR,
        gradient_accumulation_steps=GRAD_ACCUM,
    )
    device = accelerator.device
    logger.info(f"Accelerator: {accelerator.num_processes} processes, device={device}")

    # Optimizer (separate weight decay groups)
    decay = [p for n, p in model.named_parameters() if p.dim() >= 2]
    no_decay = [p for n, p in model.named_parameters() if p.dim() < 2]
    optimizer = torch.optim.AdamW([
        {"params": decay, "weight_decay": WEIGHT_DECAY},
        {"params": no_decay, "weight_decay": 0.0},
    ], lr=LR, betas=BETAS, fused=(device.type == "cuda"))

    # Resume
    start_step = 0
    tokens_seen = 0
    wandb_run_id = None
    current_stage_idx = 0
    if resume:
        meta = load_checkpoint(model, optimizer)
        start_step = meta.get("step", 0)
        tokens_seen = meta.get("tokens_seen", 0)
        wandb_run_id = meta.get("wandb_run_id", None)
        current_stage_idx = meta.get("stage_idx", get_current_stage(start_step))

    # WandB
    wandb_run = None
    if accelerator.is_main_process:
        try:
            wandb_run = init_wandb(resume_run_id=wandb_run_id)
            wandb_run_id = wandb_run.id
        except Exception as e:
            logger.warning(f"WandB init failed: {e}")

    # If resuming into a later stage, rebuild the right dataloader
    if current_stage_idx > 0:
        logger.info(f"Resuming into {STAGES[current_stage_idx]['name']}, rebuilding dataloader...")
        initial_loader = build_stage_dataloader(current_stage_idx, seed_offset=current_stage_idx)

    # Prepare with accelerate
    model, optimizer, dataloader = accelerator.prepare(model, optimizer, initial_loader)

    # Training loop
    model.train()
    optimizer.zero_grad()
    step = start_step
    micro_idx = 0
    log_ce, log_cur, log_ent, log_mag, log_count = 0.0, 0.0, 0.0, 0.0, 0
    t0 = time.time()

    pbar = tqdm(total=TOTAL_STEPS, initial=start_step,
                desc=f"Training [{STAGES[current_stage_idx]['name']}]",
                disable=not accelerator.is_main_process, dynamic_ncols=True)

    def run_stage(dataloader, stage_idx):
        """Train through a single stage's dataloader. Returns when exhausted or step limit hit."""
        nonlocal step, micro_idx, tokens_seen, log_ce, log_cur, log_ent, log_mag, log_count, t0

        for batch in dataloader:
            if step >= TOTAL_STEPS:
                return

            input_ids = batch["input_ids"].to(device)
            labels = batch["labels"].to(device)

            with accelerator.accumulate(model):
                outputs = model(input_ids=input_ids, labels=labels)
                loss = outputs["loss"]
                ce_loss = outputs["ce_loss"]
                curiosity_loss = outputs["curiosity_loss"]

                log_ce += ce_loss.item() if ce_loss is not None else 0.0
                log_cur += curiosity_loss.item() if curiosity_loss is not None else 0.0
                if outputs.get("anchor_scores") is not None:
                    log_ent += compute_anchor_entropy(outputs["anchor_scores"])
                if outputs.get("experience_state") is not None:
                    log_mag += compute_experience_magnitude(outputs["experience_state"])
                log_count += 1

                accelerator.backward(loss)

            micro_idx += 1

            if micro_idx % GRAD_ACCUM == 0:
                accelerator.clip_grad_norm_(model.parameters(), GRAD_CLIP)
                lr = get_lr(step)
                set_lr(optimizer, lr)
                optimizer.step()
                optimizer.zero_grad()

                step += 1
                tokens_seen += _actual_batch_tokens

                # Logging
                if step % LOG_EVERY == 0 and accelerator.is_main_process and log_count > 0:
                    elapsed = time.time() - t0
                    tok_per_s = (LOG_EVERY * _actual_batch_tokens) / elapsed
                    avg_ce = log_ce / log_count

                    alpha_dict = {}
                    raw_model = accelerator.unwrap_model(model)
                    for block in raw_model.model.layers:
                        if hasattr(block, 'ffn') and hasattr(block.ffn, 'alpha'):
                            alpha_dict[f"alpha/layer_{block.layer_idx}"] = block.ffn.alpha.item()

                    log_dict = {
                        "train/loss": avg_ce + log_cur / log_count,
                        "train/ce_loss": avg_ce,
                        "train/curiosity_loss": log_cur / log_count,
                        "train/lr": lr,
                        "train/tokens_seen": tokens_seen,
                        "train/tok_per_sec": tok_per_s,
                        "train/ppl": math.exp(min(avg_ce, 20)),
                        "train/stage": stage_idx,
                        "metrics/anchor_entropy": log_ent / log_count,
                        "metrics/experience_mag": log_mag / log_count,
                        **alpha_dict,
                    }
                    pbar.set_postfix(
                        stage=stage_idx + 1, loss=f"{avg_ce:.4f}",
                        ppl=f"{math.exp(min(avg_ce, 20)):.2f}",
                        lr=f"{lr:.2e}", tps=f"{tok_per_s:.0f}"
                    )
                    if wandb_run:
                        wandb.log(log_dict, step=step)

                    log_ce = log_cur = log_ent = log_mag = 0.0
                    log_count = 0
                    t0 = time.time()

                # Checkpoint
                if step % CHECKPOINT_EVERY == 0 and accelerator.is_main_process:
                    raw_model = accelerator.unwrap_model(model)
                    save_checkpoint(raw_model, optimizer, step, tokens_seen,
                                    {"ce_loss": log_ce / max(1, log_count)},
                                    wandb_run_id, stage_idx=stage_idx)

                pbar.update(1)

                # Check for stage boundary
                new_stage = get_current_stage(step)
                if new_stage != stage_idx:
                    return  # Signal stage transition

    # Run through stages starting from current_stage_idx
    for stage_idx in range(current_stage_idx, 3):
        if step >= TOTAL_STEPS:
            break

        stage = STAGES[stage_idx]
        logger.info(f"\n{'='*58}")
        logger.info(f"  ENTERING {stage['name'].upper()}")
        logger.info(f"  Global step: {step:,} | Tokens: {tokens_seen:,}")
        logger.info(f"{'='*58}")
        pbar.set_description(f"Training [{stage['name']}]")

        if wandb_run:
            wandb.log({"train/stage_transition": stage_idx, "train/stage_name": stage["name"]}, step=step)

        # Save checkpoint at stage boundary (except first stage start)
        if stage_idx > current_stage_idx and accelerator.is_main_process:
            raw_model = accelerator.unwrap_model(model)
            save_checkpoint(raw_model, optimizer, step, tokens_seen, {},
                            wandb_run_id, stage_idx=stage_idx)

        # Build new dataloader for stages 2+
        if stage_idx > 0 and stage_idx > current_stage_idx:
            new_loader = build_stage_dataloader(stage_idx, seed_offset=stage_idx)
            dataloader = accelerator.prepare(new_loader)

        run_stage(dataloader, stage_idx)

    pbar.close()

    # Final checkpoint
    if accelerator.is_main_process:
        raw_model = accelerator.unwrap_model(model)
        save_checkpoint(raw_model, optimizer, step, tokens_seen, {},
                        wandb_run_id, stage_idx=2)
        if wandb_run:
            wandb.finish()

    logger.info(f"Training complete. Steps: {step:,} | Tokens: {tokens_seen:,}")
    return accelerator.unwrap_model(model)

print("Training loop ready (3-stage curriculum).")


if __name__ == "__main__":
    logger.info("Building Stage 1 dataloader...")
    stage1_loader = build_stage_dataloader(0)
    model = train(model=model, initial_loader=stage1_loader, resume=True)
