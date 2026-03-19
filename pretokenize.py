#!/usr/bin/env python3
"""
Pre-tokenize HF datasets into binary token files for fast training.

Phase 1: Download full parquet files (parallel, full bandwidth)
Phase 2: Tokenize from disk (multi-proc, no network)
Output: /workspace/eve-saber-05b/tokens/<source>.bin (uint16 arrays)

Usage:
    python pretokenize.py                      # download + tokenize all
    python pretokenize.py --source python-edu  # just one source
    python pretokenize.py --dry-run            # show plan
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

CACHE_DIR = WORKSPACE / '.cache' / 'huggingface' / 'datasets'
CACHE_DIR.mkdir(parents=True, exist_ok=True)

NUM_PROC = max(1, (os.cpu_count() or 4) - 2)  # leave 2 cores for training

# Max tokens needed per source across ALL 3 stages (with 10% buffer)
SOURCES = {
    'python-edu': {
        'id': 'eminorhan/python-edu', 'cfg': None,
        'need': 22_000_000_000,
    },
    'fineweb-edu': {
        'id': 'HuggingFaceFW/fineweb-edu', 'cfg': 'sample-350BT',
        'need': 9_000_000_000,
        'stream_only': True,  # 350BT dataset, don't download fully
    },
    'dclm': {
        'id': 'mlfoundations/dclm-baseline-1.0', 'cfg': None,
        'need': 3_000_000_000,
        'stream_only': True,  # 27K+ shards, don't download fully
    },
    'open-web-math': {
        'id': 'open-web-math/open-web-math', 'cfg': 'default',
        'need': 7_000_000_000,
    },
    'cosmopedia': {
        'id': 'HuggingFaceTB/cosmopedia', 'cfg': 'web_samples_v2',
        'need': 6_000_000_000,
    },
    'finepdfs': {
        'id': 'HuggingFaceFW/finepdfs', 'cfg': 'eng_Latn',
        'need': 6_000_000_000,
    },
    'finemath': {
        'id': 'HuggingFaceTB/finemath', 'cfg': 'finemath-4+',
        'need': 5_000_000_000,
    },
    'wikipedia': {
        'id': 'wikimedia/wikipedia', 'cfg': '20231101.en',
        'need': 2_000_000_000,
    },
}


def get_existing_tokens(path):
    """Count tokens already in a .bin file."""
    if not path.exists():
        return 0
    return path.stat().st_size // 2


def tokenize_source(name, info, tokenizer):
    """Download dataset to disk, then tokenize in parallel to .bin file."""
    out_path = TOKEN_DIR / f'{name}.bin'
    existing = get_existing_tokens(out_path)

    if existing >= info['need']:
        print(f'  {name}: {existing/1e9:.1f}B tokens (done)')
        return existing

    eos_id = tokenizer.eos_token_id
    target = info['need']
    ds_id, cfg = info['id'], info['cfg']

    # Stream-only datasets: too large to download fully
    if info.get('stream_only'):
        print(f'\n  [{name}] Streaming (dataset too large for full download)')
        return _tokenize_streaming(name, info, tokenizer)

    # ---- Phase 1: Download to disk (parallel parquet download) ----
    print(f'\n  [{name}] Phase 1: Downloading to disk...')
    try:
        if cfg:
            ds = load_dataset(ds_id, cfg, split='train', cache_dir=str(CACHE_DIR))
        else:
            ds = load_dataset(ds_id, split='train', cache_dir=str(CACHE_DIR))
        print(f'  [{name}] Downloaded: {len(ds):,} examples')
    except Exception as e:
        print(f'  [{name}] Full download failed ({e}), falling back to streaming...')
        # Fallback: stream if dataset is too large or download fails
        return _tokenize_streaming(name, info, tokenizer)

    # ---- Phase 2: Tokenize from disk (multi-proc, no network) ----
    print(f'  [{name}] Phase 2: Tokenizing with {NUM_PROC} processes...')

    # Figure out the text column
    sample = ds[0]
    text_col = None
    for col in ['text', 'content', 'passage', 'document', 'code', 'body',
                'input', 'output', 'instruction', 'response', 'markdown']:
        if col in sample and isinstance(sample[col], str):
            text_col = col
            break

    if text_col is None:
        # Last resort: find the first string column with substantial content
        for col in sample:
            if isinstance(sample[col], str) and len(sample[col]) > 100:
                text_col = col
                print(f'  [{name}] Auto-detected text column: "{col}"')
                break

    if text_col is None:
        print(f'  [{name}] No text column in {list(sample.keys())}, falling back to streaming...')
        del ds
        return _tokenize_streaming(name, info, tokenizer)

    # Estimate how many examples we need (avg ~500 tokens/example)
    # Take more than needed since some will be filtered
    avg_tokens_per_example = 500
    examples_needed = int(target * 1.2 / avg_tokens_per_example)
    if examples_needed < len(ds):
        ds = ds.select(range(examples_needed))
        print(f'  [{name}] Using {len(ds):,} / {examples_needed:,} examples (capped to save time)')

    total = existing
    FLUSH_EVERY = 50_000_000
    flush_buffer = []

    pbar = tqdm(
        total=target, initial=existing,
        desc=name, unit='tok', unit_scale=True, unit_divisor=1_000_000_000,
    )

    # Process in batches for speed
    BATCH_SIZE = 1000
    for i in range(0, len(ds), BATCH_SIZE):
        if total >= target:
            break

        batch = ds[i:i + BATCH_SIZE]
        texts = batch[text_col]

        # Filter short texts
        texts = [t for t in texts if isinstance(t, str) and len(t.strip()) >= 50]
        if not texts:
            continue

        # Batch tokenize
        encoded = tokenizer(
            texts, add_special_tokens=False,
            truncation=False, return_attention_mask=False,
        )

        for ids in encoded['input_ids']:
            flush_buffer.extend(ids)
            flush_buffer.append(eos_id)

        n_new = sum(len(ids) + 1 for ids in encoded['input_ids'])
        total += n_new
        pbar.update(n_new)

        if len(flush_buffer) >= FLUSH_EVERY:
            arr = np.array(flush_buffer, dtype=np.uint16)
            with open(out_path, 'ab') as f:
                f.write(arr.tobytes())
            flush_buffer = []

    # Final flush
    if flush_buffer:
        arr = np.array(flush_buffer, dtype=np.uint16)
        with open(out_path, 'ab') as f:
            f.write(arr.tobytes())

    pbar.close()
    final = get_existing_tokens(out_path)
    print(f'  [{name}] {final/1e9:.1f}B tokens saved ({out_path.stat().st_size/1e9:.1f}GB)')

    # Free memory + disk — drop the downloaded dataset
    del ds

    # Clean up HF cache for this dataset to reclaim disk space
    # (the .bin file is all we need now)
    import glob
    for cache_subdir in CACHE_DIR.glob('*'):
        if cache_subdir.is_dir() and ds_id.replace('/', '___') in cache_subdir.name:
            import shutil
            cache_size = sum(f.stat().st_size for f in cache_subdir.rglob('*') if f.is_file())
            shutil.rmtree(cache_subdir)
            print(f'  [{name}] Cleaned HF cache: freed {cache_size/1e9:.1f}GB')

    return final


def _tokenize_streaming(name, info, tokenizer):
    """Fallback: stream + tokenize for datasets too large to download fully."""
    out_path = TOKEN_DIR / f'{name}.bin'
    existing = get_existing_tokens(out_path)
    eos_id = tokenizer.eos_token_id
    target = info['need']
    ds_id, cfg = info['id'], info['cfg']

    print(f'  [{name}] Streaming fallback: have {existing/1e9:.1f}B, need {target/1e9:.1f}B')

    if cfg:
        ds = load_dataset(ds_id, cfg, split='train', streaming=True)
    else:
        ds = load_dataset(ds_id, split='train', streaming=True)

    total = existing
    flush_buffer = []
    batch_texts = []
    BATCH_SIZE = 500
    FLUSH_EVERY = 50_000_000

    pbar = tqdm(
        total=target, initial=existing,
        desc=f'{name} (stream)', unit='tok', unit_scale=True, unit_divisor=1_000_000_000,
    )

    for item in ds:
        if total >= target:
            break

        text = (item.get('text') or item.get('content') or
                item.get('passage') or item.get('document') or '')
        if not isinstance(text, str) or len(text.strip()) < 50:
            continue

        batch_texts.append(text)

        if len(batch_texts) >= BATCH_SIZE:
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

            if len(flush_buffer) >= FLUSH_EVERY:
                arr = np.array(flush_buffer, dtype=np.uint16)
                with open(out_path, 'ab') as f:
                    f.write(arr.tobytes())
                flush_buffer = []

    if batch_texts:
        encoded = tokenizer(
            batch_texts, add_special_tokens=False,
            truncation=False, return_attention_mask=False,
        )
        for ids in encoded['input_ids']:
            flush_buffer.extend(ids)
            flush_buffer.append(eos_id)

    if flush_buffer:
        arr = np.array(flush_buffer, dtype=np.uint16)
        with open(out_path, 'ab') as f:
            f.write(arr.tobytes())

    pbar.close()
    final = get_existing_tokens(out_path)
    print(f'  [{name}] {final/1e9:.1f}B tokens saved ({out_path.stat().st_size/1e9:.1f}GB)')
    return final


def main():
    parser = argparse.ArgumentParser(description='Pre-tokenize datasets for Eve training')
    parser.add_argument('--source', type=str, help='Tokenize just one source (e.g. python-edu)')
    parser.add_argument('--dry-run', action='store_true', help='Show plan without downloading')
    args = parser.parse_args()

    print(f'Token output dir: {TOKEN_DIR}')
    print(f'HF cache dir: {CACHE_DIR}')
    print(f'Tokenize workers: {NUM_PROC}')
    print()

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
        disk_gb = remaining * 2 / 1e9
        status = 'done' if remaining == 0 else f'{remaining/1e9:.1f}B tokens, ~{disk_gb:.0f}GB'
        print(f'  {name:15s}: {status}')

    print(f'\nTotal: {total_have/1e9:.1f}B / {total_need/1e9:.1f}B tokens')
    print(f'Remaining disk: ~{(total_need - total_have) * 2 / 1e9:.0f}GB')

    if args.dry_run:
        return

    print('\nStarting download + tokenize...\n')
    tokenizer = GPT2TokenizerFast.from_pretrained('gpt2')

    for name, info in sources_to_run.items():
        tokenize_source(name, info, tokenizer)

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
