#!/usr/bin/env python3
"""
Eve-3-SABER-0.5B -- Standalone Training Script (50B tokens)

Combines the Colab/RunPod notebook cells into a single script.
Supports automatic resume from workspace checkpoints.

Usage:
    python train_full.py

Environment variables:
    EVE_WORKSPACE_ROOT     Base directory (default: /workspace)
    EVE_RUN_NAME           Run name (default: eve-saber-05b)
    EVE_RUN_DIR            Override run directory
    EVE_SESSION_TOKENS     Tokens per session (default: 5B)
    EVE_MICRO_BATCH        Override micro batch size
    EVE_EFFECTIVE_BATCH_TOKENS  Override effective batch tokens (default: 524288)
    EVE_GRADIENT_CHECKPOINTING  Force on/off (default: auto by VRAM)
    EVE_CHECKPOINT_EVERY   Steps between checkpoints (default: 500)
    EVE_KEEP_LAST_N        Checkpoints to keep (default: 3)
    EVE_LOG_EVERY          Steps between log prints (default: 25)
    EVE_DATA_LOADER_WORKERS  DataLoader workers (default: auto)
"""

# ============================================================
# WORKSPACE SETUP
# ============================================================
import os
from pathlib import Path

WORKSPACE_ROOT = Path(os.environ.get('EVE_WORKSPACE_ROOT', '/workspace'))
RUN_NAME = os.environ.get('EVE_RUN_NAME', 'eve-saber-05b')
RUN_DIR = Path(os.environ.get('EVE_RUN_DIR', str(WORKSPACE_ROOT / RUN_NAME)))
CKPT_DIR = RUN_DIR / 'checkpoints'

cache_root = WORKSPACE_ROOT / '.cache' / 'huggingface'
os.environ.setdefault('HF_HOME', str(cache_root))
os.environ.setdefault('HF_DATASETS_CACHE', str(cache_root / 'datasets'))
os.environ.setdefault('HF_HUB_CACHE', str(cache_root / 'hub'))

for path in [
    WORKSPACE_ROOT, RUN_DIR, CKPT_DIR,
    Path(os.environ['HF_HOME']),
    Path(os.environ['HF_DATASETS_CACHE']),
    Path(os.environ['HF_HUB_CACHE']),
]:
    path.mkdir(parents=True, exist_ok=True)


# ============================================================
# CONFIG + MODEL
# ============================================================
import torch
import math
import time
import json
import shutil
import random
import logging
from typing import Optional, Tuple

import numpy as np
from tqdm.auto import tqdm

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s | %(levelname)s | %(message)s',
    datefmt='%H:%M:%S',
)
logger = logging.getLogger('train')
logging.getLogger('httpx').setLevel(logging.WARNING)
logging.getLogger('httpcore').setLevel(logging.WARNING)

if torch.cuda.is_available():
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    try:
        torch.set_float32_matmul_precision('high')
    except AttributeError:
        pass

# ---- GPU detection ----
gpu_name = torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'cpu'
vram_gb = torch.cuda.get_device_properties(0).total_memory / 1024**3 if torch.cuda.is_available() else 0

# bf16 requires compute capability >= 8.0 (Ampere+: A100, RTX 3090/4090, RTX 6000, L4, H100, etc.)
compute_capability = torch.cuda.get_device_capability(0) if torch.cuda.is_available() else (0, 0)
is_bf16_capable = compute_capability[0] >= 8

# Precision: bf16 on Ampere+, fp16 on T4/older
if is_bf16_capable:
    DTYPE_STR = 'bf16'
    DTYPE_TORCH = torch.bfloat16
else:
    DTYPE_STR = 'fp16'
    DTYPE_TORCH = torch.float16

print(f'GPU: {gpu_name} ({vram_gb:.0f} GB) | Compute: {compute_capability[0]}.{compute_capability[1]} | Precision: {DTYPE_STR}')

# ---- Architecture: 0.5B ----
from configuration_saber import SABERConfig
from modeling_saber import SABERForCausalLM
from transformers import GPT2TokenizerFast

config = SABERConfig(
    vocab_size=50257,
    d_model=1536,
    n_heads=12,
    head_dim=128,
    n_layers=20,
    d_ff=2164,         # tuned for ~500M params
    max_position_embeddings=2048,
    d_exp=192,
    d_anchor=96,
    n_anchors=64,
    curiosity_coeff=0.01,
    tie_word_embeddings=True,
)

model = SABERForCausalLM(config)
n_params = sum(p.numel() for p in model.parameters())
print(f'Model: {n_params:,} params ({n_params/1e6:.1f}M)')

use_gradient_checkpointing = os.environ.get('EVE_GRADIENT_CHECKPOINTING')
if use_gradient_checkpointing is None:
    use_gradient_checkpointing = vram_gb < 70
else:
    use_gradient_checkpointing = use_gradient_checkpointing.lower() in {'1', 'true', 'yes'}
if use_gradient_checkpointing:
    model.gradient_checkpointing_enable()

tokenizer = GPT2TokenizerFast.from_pretrained('gpt2')
tokenizer.pad_token = tokenizer.eos_token
EOS_ID = tokenizer.eos_token_id

# ---- Training hyperparameters ----
SEQ_LEN = 2048
TOTAL_TOKENS = 50_000_000_000
SESSION_TOKENS = int(os.environ.get('EVE_SESSION_TOKENS', 5_000_000_000))
EFFECTIVE_BATCH_TOKENS = int(os.environ.get('EVE_EFFECTIVE_BATCH_TOKENS', 524_288))

# Micro batch by VRAM (override with EVE_MICRO_BATCH if needed)
micro_batch_override = os.environ.get('EVE_MICRO_BATCH')
if micro_batch_override is not None:
    MICRO_BATCH = int(micro_batch_override)
elif vram_gb >= 140:
    MICRO_BATCH = 16   # B200 / H200 tier
elif vram_gb >= 70:
    MICRO_BATCH = 12
elif vram_gb >= 35:
    MICRO_BATCH = 6
elif vram_gb >= 20:
    MICRO_BATCH = 2
else:
    MICRO_BATCH = 1

GRAD_ACCUM = max(1, EFFECTIVE_BATCH_TOKENS // (MICRO_BATCH * SEQ_LEN))
ACTUAL_BATCH_TOKENS = MICRO_BATCH * GRAD_ACCUM * SEQ_LEN
TOTAL_STEPS = TOTAL_TOKENS // ACTUAL_BATCH_TOKENS
SESSION_STEPS = SESSION_TOKENS // ACTUAL_BATCH_TOKENS
DATA_LOADER_WORKERS = int(os.environ.get('EVE_DATA_LOADER_WORKERS', min(8, max(2, (os.cpu_count() or 4) // 2))))

# LR
LR = 3e-4
LR_MIN = 3e-5
BETAS = (0.9, 0.95)
WEIGHT_DECAY = 0.1
GRAD_CLIP = 1.0
WARMUP_STEPS = 2000
SEED = 42

# Checkpointing
CHECKPOINT_EVERY = int(os.environ.get('EVE_CHECKPOINT_EVERY', 500))
KEEP_LAST_N = int(os.environ.get('EVE_KEEP_LAST_N', 3))
LOG_EVERY = int(os.environ.get('EVE_LOG_EVERY', 25))

# ---- 3-Stage Curriculum (50B tokens) ----
STAGE_1_TOKENS = 25_000_000_000
STAGE_2_TOKENS = 15_000_000_000
STAGE_3_TOKENS = 10_000_000_000

STAGE_1_STEPS = STAGE_1_TOKENS // ACTUAL_BATCH_TOKENS
STAGE_2_STEPS = STAGE_2_TOKENS // ACTUAL_BATCH_TOKENS
STAGE_3_STEPS = STAGE_3_TOKENS // ACTUAL_BATCH_TOKENS

print(f'\nBatch: micro={MICRO_BATCH}, grad_accum={GRAD_ACCUM}, effective={ACTUAL_BATCH_TOKENS:,} tok/step')
print(f'Total: {TOTAL_STEPS:,} steps | Session: {SESSION_STEPS:,} steps ({SESSION_TOKENS/1e9:.0f}B tokens)')
print(f'Stages: Foundation={STAGE_1_STEPS:,} | Specialization={STAGE_2_STEPS:,} | Anneal={STAGE_3_STEPS:,}')
print(f'Checkpoint every {CHECKPOINT_EVERY} steps to {CKPT_DIR}')
print(f'Gradient checkpointing: {use_gradient_checkpointing} | DataLoader workers: {DATA_LOADER_WORKERS}')


# ============================================================
# DATA PIPELINE + TRAINING UTILITIES
# ============================================================
from torch.utils.data import DataLoader, IterableDataset
from datasets import load_dataset
from accelerate import Accelerator
from accelerate.utils import set_seed

set_seed(SEED)

# ---- Data mixes ----
# Note: bigcode/starcoderdata and codeparrot/github-code-clean both fail
# (config issues / legacy dataset scripts). Using smollm-corpus python-edu
# as primary code source -- it's curated, high-quality Python/ML code.
#
# Stage 1: Foundation (25B) -- broad language + code
DATA_MIX_STAGE1 = [
    ('HuggingFaceTB/smollm-corpus',     'python-edu',     0.35, True),
    ('HuggingFaceFW/fineweb-edu',       'sample-350BT',   0.25, True),
    ('mlfoundations/dclm-baseline-1.0', None,            0.10, True),
    ('open-web-math/open-web-math',     'default',       0.15, True),
    ('HuggingFaceTB/cosmopedia',        'web_samples_v2', 0.10, True),
    ('HuggingFaceFW/finepdfs',          'eng_Latn',      0.05, True),
]

# Stage 2: Specialization (15B) -- heavy code + technical
DATA_MIX_STAGE2 = [
    ('HuggingFaceTB/smollm-corpus',     'python-edu',     0.50, True),
    ('HuggingFaceTB/finemath',          'finemath-4+',    0.15, True),
    ('HuggingFaceFW/finepdfs',          'eng_Latn',       0.15, True),
    ('open-web-math/open-web-math',     'default',       0.10, True),
    ('HuggingFaceTB/cosmopedia',        'web_samples_v2', 0.05, True),
    ('HuggingFaceFW/fineweb-edu',       'sample-10BT',    0.05, True),
]

# Stage 3: Anneal (10B) -- highest quality, LR -> 0
DATA_MIX_STAGE3 = [
    ('HuggingFaceTB/smollm-corpus',     'python-edu',     0.35, True),
    ('HuggingFaceTB/finemath',          'finemath-4+',    0.20, True),
    ('HuggingFaceTB/cosmopedia',        'web_samples_v2', 0.15, True),
    ('HuggingFaceFW/finepdfs',          'eng_Latn',       0.10, True),
    ('HuggingFaceFW/fineweb-edu',       'sample-10BT',    0.05, True),
    ('wikimedia/wikipedia',             '20231101.en',    0.10, True),
    ('open-web-math/open-web-math',     'default',       0.05, True),
]

STAGES = [
    {'name': 'Stage 1: Foundation',     'tokens': STAGE_1_TOKENS, 'steps': STAGE_1_STEPS, 'data_mix': DATA_MIX_STAGE1},
    {'name': 'Stage 2: Specialization', 'tokens': STAGE_2_TOKENS, 'steps': STAGE_2_STEPS, 'data_mix': DATA_MIX_STAGE2},
    {'name': 'Stage 3: Anneal',         'tokens': STAGE_3_TOKENS, 'steps': STAGE_3_STEPS, 'data_mix': DATA_MIX_STAGE3},
]


# ---- Tokenization + packing ----
def tokenise_example(example):
    text = (example.get('text') or example.get('content') or
            example.get('passage') or example.get('document') or '')
    if not isinstance(text, str) or len(text.strip()) < 50:
        return None
    try:
        ids = tokenizer.encode(text, add_special_tokens=False, truncation=False)
        return ids + [EOS_ID]
    except Exception:
        return None


class PackedDataset(IterableDataset):
    """Streams from weighted HF datasets, packs into fixed-length sequences."""

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
            except Exception:
                continue

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
                yield {'input_ids': input_ids, 'labels': input_ids.clone()}
                tokens_emitted += self.seq_len


def build_streaming_datasets(data_mix):
    loaded = []
    total_weight = sum(w for _, _, w, _ in data_mix)
    for ds_id, cfg, weight, streaming in data_mix:
        try:
            logger.info(f'Loading {ds_id}/{cfg} ({weight/total_weight:.0%})...')
            if cfg:
                ds = load_dataset(ds_id, cfg, split='train', streaming=streaming)
            else:
                ds = load_dataset(ds_id, split='train', streaming=streaming)
            ds = ds.shuffle(seed=SEED, buffer_size=10_000)
            loaded.append((ds, weight / total_weight))
        except Exception as e:
            logger.warning(f'Failed to load {ds_id}/{cfg}: {e}')
    total_w = sum(w for _, w in loaded)
    loaded = [(ds, w / total_w) for ds, w in loaded]
    return loaded


def build_stage_dataloader(stage_idx, seed_offset=0, max_tokens=None):
    stage = STAGES[stage_idx]
    logger.info(f'Building dataloader: {stage["name"]} ({stage["tokens"]/1e9:.0f}B tokens)')
    datasets = build_streaming_datasets(stage['data_mix'])
    packed = PackedDataset(
        hf_datasets_weighted=datasets,
        seq_len=SEQ_LEN,
        seed=SEED + seed_offset,
        max_tokens=max_tokens or stage['tokens'],
    )
    loader_kwargs = {
        'batch_size': MICRO_BATCH,
        'num_workers': DATA_LOADER_WORKERS,
        'pin_memory': torch.cuda.is_available(),
    }
    if DATA_LOADER_WORKERS > 0:
        loader_kwargs['prefetch_factor'] = 4
        loader_kwargs['persistent_workers'] = True
    return DataLoader(packed, **loader_kwargs)


# ---- LR schedule ----
def get_lr(global_step):
    stages_12_steps = STAGE_1_STEPS + STAGE_2_STEPS
    if global_step < stages_12_steps:
        if global_step < WARMUP_STEPS:
            return LR * global_step / max(1, WARMUP_STEPS)
        progress = (global_step - WARMUP_STEPS) / max(1, stages_12_steps - WARMUP_STEPS)
        cosine = 0.5 * (1.0 + math.cos(math.pi * progress))
        return LR_MIN + (LR - LR_MIN) * cosine
    stage3_step = global_step - stages_12_steps
    return LR_MIN * max(0.0, 1.0 - stage3_step / max(1, STAGE_3_STEPS))


def get_current_stage(global_step):
    if global_step < STAGE_1_STEPS:
        return 0
    if global_step < STAGE_1_STEPS + STAGE_2_STEPS:
        return 1
    return 2


# ---- Checkpoint save/load (workspace) ----
def save_checkpoint(model, optimizer, step, tokens_seen, stage_idx, wandb_run_id=None):
    ckpt_path = Path(CKPT_DIR) / f'step_{step:08d}'
    ckpt_path.mkdir(parents=True, exist_ok=True)

    torch.save(model.state_dict(), ckpt_path / 'model.pt')
    torch.save(optimizer.state_dict(), ckpt_path / 'optimizer.pt')

    meta = {
        'step': step,
        'tokens_seen': tokens_seen,
        'stage_idx': stage_idx,
        'wandb_run_id': wandb_run_id,
        'total_target': TOTAL_TOKENS,
    }
    with open(ckpt_path / 'meta.json', 'w') as f:
        json.dump(meta, f, indent=2)
    with open(Path(CKPT_DIR) / 'latest.txt', 'w') as f:
        f.write(str(ckpt_path))

    logger.info(f'[Step {step:,}] Saved checkpoint: {ckpt_path.name}')

    all_ckpts = sorted(Path(CKPT_DIR).glob('step_*'), key=lambda p: int(p.name.split('_')[1]))
    stage_boundary_steps = {0, STAGE_1_STEPS, STAGE_1_STEPS + STAGE_2_STEPS}
    regular = [c for c in all_ckpts if int(c.name.split('_')[1]) not in stage_boundary_steps]
    for old in regular[:-KEEP_LAST_N]:
        shutil.rmtree(old)
        logger.info(f'  Pruned: {old.name}')


def load_checkpoint(model, optimizer=None):
    latest_file = Path(CKPT_DIR) / 'latest.txt'
    if not latest_file.exists():
        logger.info('No checkpoint found in workspace. Starting from scratch.')
        return {'step': 0, 'tokens_seen': 0, 'stage_idx': 0, 'wandb_run_id': None}

    ckpt_path = Path(latest_file.read_text().strip())
    if not ckpt_path.exists():
        logger.warning(f'Checkpoint path {ckpt_path} not found. Starting fresh.')
        return {'step': 0, 'tokens_seen': 0, 'stage_idx': 0, 'wandb_run_id': None}

    logger.info(f'Loading checkpoint: {ckpt_path.name}')
    model.load_state_dict(torch.load(ckpt_path / 'model.pt', map_location='cpu'))

    if optimizer and (ckpt_path / 'optimizer.pt').exists():
        optimizer.load_state_dict(torch.load(ckpt_path / 'optimizer.pt', map_location='cpu'))

    with open(ckpt_path / 'meta.json') as f:
        meta = json.load(f)
    logger.info(f'  Resumed: step {meta["step"]:,} | {meta["tokens_seen"]/1e9:.2f}B tokens | stage {meta["stage_idx"]}')
    return meta


print('Data pipeline + utilities ready')


# ============================================================
# TRAINING LOOP
# ============================================================
if __name__ == '__main__':

    # ---- Optimizer ----
    decay_params = [p for n, p in model.named_parameters() if p.dim() >= 2]
    nodecay_params = [p for n, p in model.named_parameters() if p.dim() < 2]
    optimizer = torch.optim.AdamW([
        {'params': decay_params, 'weight_decay': WEIGHT_DECAY},
        {'params': nodecay_params, 'weight_decay': 0.0},
    ], lr=LR, betas=BETAS, fused=torch.cuda.is_available())

    # ---- Resume from workspace checkpoint ----
    meta = load_checkpoint(model, optimizer)
    start_step = meta['step']
    tokens_seen = meta['tokens_seen']
    wandb_run_id = meta.get('wandb_run_id')
    current_stage_idx = meta.get('stage_idx', get_current_stage(start_step))

    if start_step >= TOTAL_STEPS:
        print(f'Training complete! {tokens_seen/1e9:.1f}B tokens across {start_step:,} steps.')
        raise SystemExit('Training finished.')

    session_end_step = min(start_step + SESSION_STEPS, TOTAL_STEPS)
    print(f'\nSession plan: step {start_step:,} -> {session_end_step:,} ({(session_end_step - start_step) * ACTUAL_BATCH_TOKENS / 1e9:.1f}B tokens)')
    print(f'Global progress: {tokens_seen/1e9:.1f}B / {TOTAL_TOKENS/1e9:.0f}B ({tokens_seen/TOTAL_TOKENS*100:.1f}%)')
    print(f'Checkpoint dir: {CKPT_DIR}')

    # ---- WandB (optional) ----
    try:
        import wandb
        wandb_run = wandb.init(
            project='eve-3-saber-05b', name=f'session-{start_step}',
            id=wandb_run_id, resume='allow' if wandb_run_id else None,
            config={'d_model': 1536, 'n_layers': 20, 'total_tokens': TOTAL_TOKENS,
                    'session_tokens': SESSION_TOKENS, 'micro_batch': MICRO_BATCH,
                    'grad_accum': GRAD_ACCUM, 'dtype': DTYPE_STR},
        )
        wandb_run_id = wandb_run.id
    except Exception:
        wandb_run = None
        print('WandB disabled (login with wandb.login() to enable)')

    # ---- Accelerate ----
    accelerator = Accelerator(
        mixed_precision=DTYPE_STR,
        gradient_accumulation_steps=GRAD_ACCUM,
    )
    device = accelerator.device

    # Build dataloader for current stage
    # Use tokens_seen as seed offset to avoid repeating data across sessions
    stage_loader = build_stage_dataloader(
        current_stage_idx,
        seed_offset=tokens_seen // ACTUAL_BATCH_TOKENS,
    )

    model, optimizer, dataloader = accelerator.prepare(model, optimizer, stage_loader)

    # ---- Training loop ----
    model.train()
    optimizer.zero_grad(set_to_none=True)

    step = start_step
    micro_idx = 0
    log_ce, log_cur, log_count = 0.0, 0.0, 0
    t0 = time.time()
    session_start = time.time()

    pbar = tqdm(total=session_end_step - start_step,
                desc=f'{STAGES[current_stage_idx]["name"]}',
                dynamic_ncols=True)

    try:
        while step < session_end_step:
            for batch in dataloader:
                if step >= session_end_step:
                    break

                input_ids = batch['input_ids'].to(device)
                labels = batch['labels'].to(device)

                with accelerator.accumulate(model):
                    outputs = model(input_ids=input_ids, labels=labels)
                    loss = outputs['loss']
                    ce_loss = outputs.get('ce_loss')
                    curiosity_loss = outputs.get('curiosity_loss')

                    log_ce += (ce_loss.item() if ce_loss is not None else loss.item())
                    log_cur += (curiosity_loss.item() if curiosity_loss is not None else 0.0)
                    log_count += 1

                    accelerator.backward(loss)

                micro_idx += 1

                if micro_idx % GRAD_ACCUM == 0:
                    accelerator.clip_grad_norm_(model.parameters(), GRAD_CLIP)
                    lr = get_lr(step)
                    for g in optimizer.param_groups:
                        g['lr'] = lr
                    optimizer.step()
                    optimizer.zero_grad(set_to_none=True)

                    step += 1
                    tokens_seen += ACTUAL_BATCH_TOKENS

                    if step % LOG_EVERY == 0 and log_count > 0:
                        elapsed = time.time() - t0
                        tok_s = (LOG_EVERY * ACTUAL_BATCH_TOKENS) / max(elapsed, 1)
                        avg_ce = log_ce / log_count
                        ppl = math.exp(min(avg_ce, 20))

                        pbar.set_postfix(
                            loss=f'{avg_ce:.3f}', ppl=f'{ppl:.1f}',
                            lr=f'{lr:.1e}', tps=f'{tok_s/1e3:.0f}K',
                            B=f'{tokens_seen/1e9:.2f}'
                        )

                        if wandb_run:
                            wandb.log({
                                'train/ce_loss': avg_ce,
                                'train/curiosity_loss': log_cur / log_count,
                                'train/ppl': ppl,
                                'train/lr': lr,
                                'train/tokens_B': tokens_seen / 1e9,
                                'train/tok_per_sec': tok_s,
                                'train/stage': current_stage_idx,
                            }, step=step)

                        log_ce = log_cur = 0.0
                        log_count = 0
                        t0 = time.time()

                    if step % CHECKPOINT_EVERY == 0:
                        raw_model = accelerator.unwrap_model(model)
                        save_checkpoint(raw_model, optimizer, step, tokens_seen,
                                        current_stage_idx, wandb_run_id)

                    pbar.update(1)

                    new_stage = get_current_stage(step)
                    if new_stage != current_stage_idx:
                        logger.info(f'\n{"="*50}')
                        logger.info(f'STAGE TRANSITION: {STAGES[current_stage_idx]["name"]} -> {STAGES[new_stage]["name"]}')
                        logger.info(f'{"="*50}')

                        raw_model = accelerator.unwrap_model(model)
                        save_checkpoint(raw_model, optimizer, step, tokens_seen,
                                        new_stage, wandb_run_id)

                        current_stage_idx = new_stage
                        pbar.set_description(STAGES[current_stage_idx]['name'])

                        new_loader = build_stage_dataloader(
                            current_stage_idx,
                            seed_offset=tokens_seen // ACTUAL_BATCH_TOKENS,
                        )
                        dataloader = accelerator.prepare(new_loader)
                        break

    except KeyboardInterrupt:
        logger.info('\nInterrupted! Saving emergency checkpoint...')
    except Exception as e:
        logger.error(f'\nError: {e}. Saving emergency checkpoint...')
        import traceback; traceback.print_exc()

    # ---- End-of-session save ----
    pbar.close()
    raw_model = accelerator.unwrap_model(model)
    save_checkpoint(raw_model, optimizer, step, tokens_seen,
                    current_stage_idx, wandb_run_id)

    elapsed_h = (time.time() - session_start) / 3600
    session_tokens = (step - start_step) * ACTUAL_BATCH_TOKENS

    if wandb_run:
        wandb.finish()

    print(f'\n{"="*50}')
    print('SESSION COMPLETE')
    print(f'  Steps this session: {step - start_step:,}')
    print(f'  Tokens this session: {session_tokens/1e9:.2f}B')
    print(f'  Wall time: {elapsed_h:.1f}h')
    print(f'  Throughput: {session_tokens/elapsed_h/1e9:.2f}B tok/h')
    print(f'\n  GLOBAL PROGRESS: {tokens_seen/1e9:.1f}B / {TOTAL_TOKENS/1e9:.0f}B ({tokens_seen/TOTAL_TOKENS*100:.1f}%)')
    print(f'  Estimated remaining: {(TOTAL_TOKENS - tokens_seen) / (session_tokens / max(elapsed_h, 0.01)):.0f}h')
    print(f'{"="*50}')
