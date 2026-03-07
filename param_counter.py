"""
param_counter.py — Eve-3-SABER-1B parameter counter and d_ff tuner.

Usage:
    python param_counter.py                   # count with spec d_ff=5461
    python param_counter.py --tune-dff        # binary-search d_ff to hit exactly 1.0B
    python param_counter.py --dff 5500        # count with a custom d_ff
"""

import argparse
import math
from typing import Optional

# ---------------------------------------------------------------------------
# Architecture hyperparameters (canonical defaults from architecture_spec.md)
# ---------------------------------------------------------------------------
D_MODEL    = 2048       # Hidden / residual dimension
N_HEADS    = 16         # Number of attention heads
HEAD_DIM   = 128        # head_dim = d_model / n_heads
N_LAYERS   = 24         # Transformer depth
D_FF       = 5461       # SwiGLU feed-forward inner dimension (⅔ × 8192, tuned)
VOCAB_SIZE = 50257      # GPT-2 vocabulary
MAX_SEQ    = 2048       # Maximum sequence length (RoPE, no positional embedding table)
D_EXP      = 256        # Experience stream dimension
N_ANCHORS  = 64         # Slip-anchor codebook size
D_ANCHOR   = 128        # Anchor bottleneck dimension
TARGET_B   = 1_000_000_000  # 1.0 B parameter target

# Derived constants (must remain consistent with above)
assert D_MODEL == N_HEADS * HEAD_DIM, "d_model must equal n_heads * head_dim"
RESONANT_LAYERS = [i for i in range(N_LAYERS) if i % 2 == 0]   # even layers: 0,2,4,...
STANDARD_LAYERS = [i for i in range(N_LAYERS) if i % 2 != 0]   # odd layers

# ---------------------------------------------------------------------------
# Per-component parameter formulas
# ---------------------------------------------------------------------------

def embeddings(vocab_size: int = VOCAB_SIZE, d_model: int = D_MODEL) -> int:
    """Token embedding table.  RoPE has no parameter table (closed-form)."""
    return vocab_size * d_model


def qkv_projection(d_model: int = D_MODEL, n_layers: int = N_LAYERS) -> int:
    """Q, K, V linear projections (no bias) per layer × n_layers."""
    per_layer = 3 * d_model * d_model          # Q + K + V, each d_model→d_model
    return per_layer * n_layers


def o_projection(d_model: int = D_MODEL, n_layers: int = N_LAYERS) -> int:
    """Output projection (no bias) per layer × n_layers."""
    per_layer = d_model * d_model
    return per_layer * n_layers


def swiglu_ffn(d_model: int = D_MODEL, d_ff: int = D_FF, n_layers: int = N_LAYERS) -> int:
    """SwiGLU: W1 (gate), W3 (up), W2 (down) per layer × n_layers (no bias)."""
    per_layer = 3 * d_model * d_ff             # W1 + W3: d_model→d_ff; W2: d_ff→d_model
    return per_layer * n_layers


def slip_anchor_params_per_layer(
    d_model: int = D_MODEL,
    n_anchors: int = N_ANCHORS,
    d_anchor: int = D_ANCHOR,
    head_dim: int = HEAD_DIM,
) -> int:
    """
    Per attention layer:
        W_anchor_down : d_model  × d_anchor   (2048 × 128)
        anchors       : n_anchors × d_anchor  (64   × 128)
        U_k           : d_anchor × head_dim   (128  × 128)
        U_v           : d_anchor × head_dim   (128  × 128)
    """
    return (
        d_model  * d_anchor    # W_anchor_down
        + n_anchors * d_anchor # learnable codebook
        + d_anchor * head_dim  # U_k
        + d_anchor * head_dim  # U_v
    )


def slip_anchor_total(
    d_model: int = D_MODEL,
    n_anchors: int = N_ANCHORS,
    d_anchor: int = D_ANCHOR,
    head_dim: int = HEAD_DIM,
    n_layers: int = N_LAYERS,
) -> int:
    return slip_anchor_params_per_layer(d_model, n_anchors, d_anchor, head_dim) * n_layers


def experience_stream_per_layer(d_model: int = D_MODEL, d_exp: int = D_EXP) -> int:
    """
    Per layer:
        W_s    : d_model × d_exp   (2048 × 256)
        W_pred : d_exp   × d_exp   (256  × 256)
        W_e    : d_exp   × d_exp   (256  × 256)
    """
    return (
        d_model * d_exp   # W_s
        + d_exp  * d_exp  # W_pred
        + d_exp  * d_exp  # W_e
    )


def experience_stream_total(
    d_model: int = D_MODEL, d_exp: int = D_EXP, n_layers: int = N_LAYERS
) -> int:
    return experience_stream_per_layer(d_model, d_exp) * n_layers


def resonant_ffn_extra_per_layer(d_model: int = D_MODEL) -> int:
    """
    Extra params in a resonant FFN layer (beyond the standard SwiGLU shared
    by all layers):
        W_freq    : d_model × d_model  (sin-modulation projection)
        alpha_raw : scalar             (1 learnable blending parameter)
    """
    return d_model * d_model + 1


def resonant_ffn_extra_total(
    d_model: int = D_MODEL, n_resonant: int = len(RESONANT_LAYERS)
) -> int:
    return resonant_ffn_extra_per_layer(d_model) * n_resonant


def rmsnorm_params(d_model: int = D_MODEL, n_layers: int = N_LAYERS) -> int:
    """
    2 RMSNorms per block (pre-attn + pre-ffn) × n_layers, plus 1 final norm.
    Each RMSNorm has a learnable scale vector of size d_model.
    """
    return (2 * n_layers + 1) * d_model


def lm_head() -> int:
    """LM head is weight-tied to token embeddings — no additional parameters."""
    return 0


# ---------------------------------------------------------------------------
# Total counter
# ---------------------------------------------------------------------------

def total_params(
    d_model: int = D_MODEL,
    n_heads: int = N_HEADS,
    n_layers: int = N_LAYERS,
    d_ff: int = D_FF,
    vocab_size: int = VOCAB_SIZE,
    d_exp: int = D_EXP,
    n_anchors: int = N_ANCHORS,
    d_anchor: int = D_ANCHOR,
) -> dict:
    """Return an ordered dict of component_name -> param_count."""
    head_dim = d_model // n_heads
    n_resonant = math.ceil(n_layers / 2)   # even layers: 0, 2, 4, …

    emb     = embeddings(vocab_size, d_model)
    qkv     = qkv_projection(d_model, n_layers)
    o_proj  = o_projection(d_model, n_layers)
    ffn     = swiglu_ffn(d_model, d_ff, n_layers)
    anchors = slip_anchor_total(d_model, n_anchors, d_anchor, head_dim, n_layers)
    exp     = experience_stream_total(d_model, d_exp, n_layers)
    res_ex  = resonant_ffn_extra_total(d_model, n_resonant)
    norms   = rmsnorm_params(d_model, n_layers)
    lm      = lm_head()

    grand_total = emb + qkv + o_proj + ffn + anchors + exp + res_ex + norms + lm

    return {
        "embeddings": emb,
        "qkv_projections": qkv,
        "o_projections": o_proj,
        "swiglu_ffn": ffn,
        "slip_anchors": anchors,
        "experience_stream": exp,
        "resonant_ffn_extra": res_ex,
        "rmsnorm": norms,
        "lm_head_tied": lm,
        "TOTAL": grand_total,
    }


# ---------------------------------------------------------------------------
# Pretty-printer
# ---------------------------------------------------------------------------

def print_breakdown(counts: dict, d_ff: int, label: str = "") -> None:
    title = f"Eve-3-SABER-1B Parameter Breakdown{' ' + label if label else ''}"
    width = 60

    print()
    print("=" * width)
    print(f"  {title}")
    print(f"  d_ff = {d_ff:,}")
    print("=" * width)
    print(f"  {'Component':<30} {'Params':>12}  {'M':>8}")
    print("-" * width)

    # Per-component per-layer details
    extras = {
        "embeddings":        f"vocab={VOCAB_SIZE:,} × d={D_MODEL}",
        "qkv_projections":   f"3 × {D_MODEL}² × {N_LAYERS} layers",
        "o_projections":     f"{D_MODEL}² × {N_LAYERS} layers",
        "swiglu_ffn":        f"3 × {D_MODEL} × {d_ff} × {N_LAYERS} layers",
        "slip_anchors":      f"~{slip_anchor_params_per_layer():,} × {N_LAYERS} layers",
        "experience_stream": f"~{experience_stream_per_layer():,} × {N_LAYERS} layers",
        "resonant_ffn_extra":f"({D_MODEL}² + 1) × {len(RESONANT_LAYERS)} even layers",
        "rmsnorm":           f"(2×{N_LAYERS} + 1) × {D_MODEL}",
        "lm_head_tied":      "tied to embeddings",
    }

    for name, val in counts.items():
        if name == "TOTAL":
            continue
        note = extras.get(name, "")
        print(f"  {name:<30} {val:>12,}  {val/1e6:>7.2f}M   {note}")

    grand = counts["TOTAL"]
    print("=" * width)
    print(f"  {'TOTAL':<30} {grand:>12,}  {grand/1e9:>7.4f}B")
    print(f"  {'vs 1.0B target':<30} {grand - TARGET_B:>+12,}")
    print("=" * width)
    print()


# ---------------------------------------------------------------------------
# Binary-search d_ff tuner
# ---------------------------------------------------------------------------

def tune_d_ff(target: int = TARGET_B) -> int:
    """
    Binary-search for the integer d_ff that makes total params closest to
    `target`.  d_ff must stay positive; upper bound is generous.
    """
    print(f"\n  Tuning d_ff to hit {target/1e9:.3f}B params …")

    lo, hi = 1, 20_000
    best_dff = D_FF
    best_err = abs(total_params(d_ff=D_FF)["TOTAL"] - target)

    while lo <= hi:
        mid = (lo + hi) // 2
        t   = total_params(d_ff=mid)["TOTAL"]
        err = abs(t - target)

        if err < best_err:
            best_err = err
            best_dff = mid

        if t < target:
            lo = mid + 1
        elif t > target:
            hi = mid - 1
        else:
            best_dff = mid
            break

    print(f"  → Best d_ff = {best_dff:,}  (error = {best_err:+,} params)\n")
    return best_dff


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(description="Eve-3-SABER-1B parameter counter")
    parser.add_argument("--tune-dff", action="store_true",
                        help="Binary-search for d_ff that hits exactly 1.0B")
    parser.add_argument("--dff", type=int, default=None,
                        help="Override d_ff (default: spec value 5461)")
    args = parser.parse_args()

    if args.tune_dff:
        best = tune_d_ff()
        counts = total_params(d_ff=best)
        print_breakdown(counts, d_ff=best, label=f"[tuned d_ff={best:,}]")

        # Also show the spec value for comparison
        counts_spec = total_params(d_ff=D_FF)
        print_breakdown(counts_spec, d_ff=D_FF, label="[spec d_ff=5461]")
    else:
        d_ff = args.dff if args.dff is not None else D_FF
        counts = total_params(d_ff=d_ff)
        print_breakdown(counts, d_ff=d_ff)

    # -------------------------------------------------------------------
    # Detailed per-layer breakdown
    # -------------------------------------------------------------------
    d_ff_use = args.dff if (args.dff is not None and not args.tune_dff) else D_FF
    if args.tune_dff:
        d_ff_use = tune_d_ff()  # recompute silently — already printed above

    print("  Per-layer parameter summary (one representative block)")
    print("  " + "-" * 56)

    slip_pl  = slip_anchor_params_per_layer()
    exp_pl   = experience_stream_per_layer()
    res_pl   = resonant_ffn_extra_per_layer()
    qkv_pl   = 3 * D_MODEL * D_MODEL
    o_pl     = D_MODEL * D_MODEL
    ffn_pl   = 3 * D_MODEL * d_ff_use
    norm_pl  = 2 * D_MODEL   # pre-attn + pre-ffn

    def row(name: str, val: int) -> None:
        print(f"    {name:<32} {val:>10,}  ({val/1e6:.4f}M)")

    row("QKV projection",      qkv_pl)
    row("O projection",        o_pl)
    row("SwiGLU FFN",          ffn_pl)
    row("Slip-anchor module",  slip_pl)
    row("Experience stream",   exp_pl)
    row("RMSNorm (×2)",        norm_pl)
    row("Resonant extra (even only)", res_pl)
    print()

    total_even = qkv_pl + o_pl + ffn_pl + slip_pl + exp_pl + norm_pl + res_pl
    total_odd  = qkv_pl + o_pl + ffn_pl + slip_pl + exp_pl + norm_pl
    row("Even block total",   total_even)
    row("Odd  block total",   total_odd)
    print()


if __name__ == "__main__":
    main()
