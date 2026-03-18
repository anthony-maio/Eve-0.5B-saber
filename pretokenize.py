#!/usr/bin/env python3
"""
Pre-tokenize HF datasets into binary token files for fast training.

Downloads streaming data, batch-tokenizes, writes flat uint16 .bin files.
Training reads via numpy mmap = zero network I/O, ~1-2s/step.

Usage:
    python pretokenize.py                    # tokenize all sources
    python pretokenize.py --source python-edu  # just one source
    python pretokenize.py --dry-run          # show plan without downloading

Output: /workspace/eve-saber-05b/tokens/<source>.bin (uint16 arrays)
Disk: ~90GB total for 45B tokens across all sources.
"""

import os
import sys
import argparse
import numpy as np
from pathlib import Path
from tqdm.auto import tqdm
from datasets import load_dataset
from transformers import GPT2TokenizerFast

# ---- Config ----
WORKSPACE = Path(os.environ.get('EVE_WORKSPACE_ROOT', '/workspace'))
TOKEN_DIR = WORKSPACE / 'eve-saber-05b' / 'tokens'
TOKEN_DIR.mkdir(parents=True, exist_ok=True)

# Max tokens needed per source across ALL 3 stages (with 10% buffer)
# Calculated from stage weights × stage token counts
SOURCES = {
    'python-edu': {
        'id': 'HuggingFaceTB/smollm-corpus', 'cfg': 'python-edu',
        'need': 22_000_000_000,  # 35%×25B + 50%×15B + 35%×10B
    },
    'fineweb-edu': {
        'id': 'HuggingFaceFW/fineweb-edu', 'cfg': 'sample-350BT',
        'need': 9_000_000_000,   # 25%×25B + 5%×15B + 5%×10B
    },
    'dclm': {
        'id': 'mlfoundations/dclm-baseline-1.0', 'cfg': None,
        'need': 3_000_000_000,   # 10%×25B only (Stage 1)
    },
    'open-web-math': {
        'id': 'open-web-math/open-web-math', 'cfg': 'default',
        'need': 7_000_000_000,   # 15%×25B + 10%×15B + 5%×10B
    },
    'cosmopedia': {
        'id': 'HuggingFaceTB/cosmopedia', 'cfg': 'web_samples_v2',
        'need': 6_000_000_000,   # 10%×25B + 5%×15B + 15%×10B
    },
    'finepdfs': {
        'id': 'HuggingFaceFW/finepdfs', 'cfg': 'eng_Latn',
        'need': 6_000_000_000,   # 5%×25B + 15%×15B + 10%×10B
    },
    'finemath': {
        'id': 'HuggingFaceTB/finemath', 'cfg': 'finemath-4+',
        'need': 5_000_000_000,   # 15%×15B + 20%×10B
    },
    'wikipedia': {
        'id': 'wikimedia/wikipedia', 'cfg': '20231101.en',
        'need': 2_000_000_000,   # 10%×10B (Stage 3 only)
    },
}


def get_existing_tokens(path):
    """Count tokens already in a .bin file."""
    if not path.exists():
        return 0
    return path.stat().st_size // 2  # uint16 = 2 bytes


def tokenize_source(name, info, tokenizer):
    """Stream a dataset, batch-tokenize, append to .bin file. Resumable."""
    out_path = TOKEN_DIR / f'{name}.bin'
    existing = get_existing_tokens(out_path)

    if existing >= info['need']:
        print(f'  {name}: {existing/1e9:.1f}B tokens (done)')
        return existing

    eos_id = tokenizer.eos_token_id
    target = info['need']

    print(f'  {name}: have {existing/1e9:.1f}B, need {target/1e9:.1f}B, downloading...')

    # Stream dataset
    ds_id, cfg = info['id'], info['cfg']
    if cfg:
        ds = load_dataset(ds_id, cfg, split='train', streaming=True)
    else:
        ds = load_dataset(ds_id, split='train', streaming=True)

    total = existing
    batch_texts = []
    BATCH_SIZE = 500  # examples per tokenization batch
    FLUSH_EVERY = 50_000_000  # flush to disk every 50M tokens

    pbar = tqdm(
        total=target, initial=existing,
        desc=name, unit='tok', unit_scale=True, unit_divisor=1_000_000_000,
    )

    flush_buffer = []

    for item in ds:
        if total >= target:
            break

        text = (item.get('text') or item.get('content') or
                item.get('passage') or item.get('document') or '')
        if not isinstance(text, str) or len(text.strip()) < 50:
            continue

        batch_texts.append(text)

        if len(batch_texts) >= BATCH_SIZE:
            # Batch tokenize (Rust backend, fast)
            encoded = tokenizer(
                batch_texts, add_special_tokens=False,
                truncation=False, return_attention_mask=False,
            )
            for ids in encoded['input_ids']:
                flush_buffer.extend(ids)
                flush_buffer.append(eos_id)

            n_new = sum(len(ids) + 1 for ids in encoded['input_ids'])
            total += n_new
            pbar.update(n_new)
            batch_texts = []

            # Periodic flush to disk
            if len(flush_buffer) >= FLUSH_EVERY:
                arr = np.array(flush_buffer, dtype=np.uint16)
                with open(out_path, 'ab') as f:
                    f.write(arr.tobytes())
                flush_buffer = []

    # Final batch
    if batch_texts:
        encoded = tokenizer(
            batch_texts, add_special_tokens=False,
            truncation=False, return_attention_mask=False,
        )
        for ids in encoded['input_ids']:
            flush_buffer.extend(ids)
            flush_buffer.append(eos_id)

    # Final flush
    if flush_buffer:
        arr = np.array(flush_buffer, dtype=np.uint16)
        with open(out_path, 'ab') as f:
            f.write(arr.tobytes())

    pbar.close()
    final = get_existing_tokens(out_path)
    print(f'  {name}: {final/1e9:.1f}B tokens saved ({out_path.stat().st_size/1e9:.1f}GB)')
    return final


def main():
    parser = argparse.ArgumentParser(description='Pre-tokenize datasets for Eve training')
    parser.add_argument('--source', type=str, help='Tokenize just one source (e.g. python-edu)')
    parser.add_argument('--dry-run', action='store_true', help='Show plan without downloading')
    args = parser.parse_args()

    print(f'Token output dir: {TOKEN_DIR}')
    print()

    # Show plan
    sources_to_run = {}
    if args.source:
        if args.source not in SOURCES:
            print(f'Unknown source: {args.source}')
            print(f'Available: {", ".join(SOURCES.keys())}')
            sys.exit(1)
        sources_to_run = {args.source: SOURCES[args.source]}
    else:
        sources_to_run = SOURCES

    total_need = 0
    total_have = 0
    print('Plan:')
    for name, info in sources_to_run.items():
        existing = get_existing_tokens(TOKEN_DIR / f'{name}.bin')
        remaining = max(0, info['need'] - existing)
        total_need += info['need']
        total_have += existing
        disk_gb = remaining * 2 / 1e9  # uint16
        status = 'done' if remaining == 0 else f'{remaining/1e9:.1f}B tokens, ~{disk_gb:.0f}GB'
        print(f'  {name:15s}: {status}')

    print(f'\nTotal: {total_have/1e9:.1f}B / {total_need/1e9:.1f}B tokens')
    print(f'Remaining disk: ~{(total_need - total_have) * 2 / 1e9:.0f}GB')

    if args.dry_run:
        return

    print('\nTokenizing...\n')
    tokenizer = GPT2TokenizerFast.from_pretrained('gpt2')

    for name, info in sources_to_run.items():
        tokenize_source(name, info, tokenizer)

    # Summary
    print('\n' + '=' * 50)
    print('DONE')
    total_tokens = 0
    total_bytes = 0
    for name in sources_to_run:
        p = TOKEN_DIR / f'{name}.bin'
        if p.exists():
            t = get_existing_tokens(p)
            total_tokens += t
            total_bytes += p.stat().st_size
            print(f'  {name:15s}: {t/1e9:.1f}B tokens ({p.stat().st_size/1e9:.1f}GB)')
    print(f'\nTotal: {total_tokens/1e9:.1f}B tokens, {total_bytes/1e9:.1f}GB on disk')
    print(f'Path: {TOKEN_DIR}')
    print('=' * 50)


if __name__ == '__main__':
    main()
