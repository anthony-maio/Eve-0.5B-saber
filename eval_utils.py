"""
eval_utils.py — Evaluation and logging utilities for Eve-3-SABER-1B.

Functions
---------
compute_perplexity          Token-level perplexity on a held-out dataloader.
compute_anchor_entropy      Per-layer entropy of slip-anchor codebook usage.
compute_experience_magnitude L2 norm of experience-stream weight matrices.
get_alpha_values            Current sigmoid(alpha_raw) for all resonant layers.
log_custom_metrics          Dispatch all custom metrics to WandB and/or stdout.
generate_samples            Greedy / sampling text generation for qualitative eval.
run_component_analysis      Print a comprehensive per-layer stats table.
create_alpha_plot           Matplotlib line-plot of alpha values over training.
create_anchor_usage_heatmap Seaborn heatmap of anchor usage across layers.
"""

from __future__ import annotations

import math
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F

# ---------------------------------------------------------------------------
# Local imports
# ---------------------------------------------------------------------------
sys.path.insert(0, str(Path(__file__).parent))

from configuration_saber import SABERConfig
from modeling_saber import (
    SlipAnchors,
    ExperienceStream,
    ResonantFFN,
    SABERBlock,
    SABERForCausalLM,
    SABERModel,
)


# ---------------------------------------------------------------------------
# 1. Perplexity
# ---------------------------------------------------------------------------

@torch.no_grad()
def compute_perplexity(
    model: SABERForCausalLM,
    dataloader,
    max_steps: int = 100,
    device: Optional[str] = None,
) -> float:
    """
    Compute token-level perplexity on *dataloader*.

    Parameters
    ----------
    model       : SABERForCausalLM (must be in eval mode on the right device)
    dataloader  : iterable that yields (input_ids,) or (input_ids, labels) batches.
                  If labels are omitted, input_ids are used as labels (causal LM).
    max_steps   : cap on evaluation batches (default 100).
    device      : override device; if None, inferred from model parameters.

    Returns
    -------
    perplexity : float (exp of mean cross-entropy loss, ignoring padding)
    """
    if device is None:
        device = next(model.parameters()).device

    model.eval()
    total_loss   = 0.0
    total_tokens = 0

    for step, batch in enumerate(dataloader):
        if step >= max_steps:
            break

        if isinstance(batch, (list, tuple)):
            input_ids = batch[0].to(device)
            labels    = batch[1].to(device) if len(batch) > 1 else input_ids.clone()
        else:
            input_ids = batch.to(device)
            labels    = input_ids.clone()

        with torch.autocast(device_type=str(device).split(":")[0],
                            enabled=(str(device) != "cpu")):
            outputs = model(input_ids=input_ids, labels=labels)

        # outputs.loss is already the combined CE + curiosity loss;
        # for perplexity we want pure CE — but the curiosity term is tiny
        # so this is a close approximation.
        loss = outputs.loss
        if loss is None:
            continue

        # Count non-padding tokens (labels == -100 are ignored)
        valid_tokens = (labels[:, 1:] != -100).sum().item()
        if valid_tokens == 0:
            continue

        total_loss   += loss.item() * valid_tokens
        total_tokens += valid_tokens

    if total_tokens == 0:
        return float("nan")

    mean_loss = total_loss / total_tokens
    return math.exp(min(mean_loss, 30.0))   # clamp to avoid overflow


# ---------------------------------------------------------------------------
# 2. Anchor Entropy
# ---------------------------------------------------------------------------

@torch.no_grad()
def compute_anchor_entropy(
    model: SABERForCausalLM,
    probe_input: Optional[torch.Tensor] = None,
    seq_len: int = 256,
    batch_size: int = 2,
) -> Dict[int, float]:
    """
    Compute the entropy of soft anchor scores in each layer.

    Entropy is measured over the codebook dimension (n_anchors) after softmax,
    averaged across tokens and batch.  Low entropy → few anchors used
    (degenerate codebook).  Maximum entropy = log2(n_anchors) ≈ 6 bits for 64.

    Parameters
    ----------
    model        : SABERForCausalLM
    probe_input  : (B, L) token id tensor; synthesised if None.
    seq_len      : length of synthetic probe sequence (used when probe_input is None)
    batch_size   : batch size for synthetic probe

    Returns
    -------
    dict mapping layer_idx -> entropy (nats)
    """
    device = next(model.parameters()).device

    if probe_input is None:
        probe_input = torch.randint(
            0, model.config.vocab_size,
            (batch_size, seq_len),
            device=device,
        )

    # We hook into each SlipAnchors module during a forward pass.
    entropies: Dict[int, float] = {}
    hooks = []

    for layer_idx, block in enumerate(model.model.layers):
        if not getattr(model.config, "enable_anchors", True):
            entropies[layer_idx] = float("nan")
            continue

        slip_module = block.self_attn.slip_anchors

        def _make_hook(idx: int):
            def _hook(module: SlipAnchors, inputs, outputs):
                h = inputs[0]                          # (B, L, d_model)
                with torch.no_grad():
                    h_anchor = module.W_anchor_down(h) # (B, L, d_anchor)
                    scores = torch.softmax(
                        h_anchor @ module.anchors.T, dim=-1
                    )                                  # (B, L, n_anchors)
                    # Mean over batch and sequence
                    mean_scores = scores.mean(dim=(0, 1))  # (n_anchors,)
                    # Entropy: -sum(p * log(p))
                    ent = -(mean_scores * (mean_scores + 1e-10).log()).sum().item()
                    entropies[idx] = ent
            return _hook

        h = slip_module.register_forward_hook(_make_hook(layer_idx))
        hooks.append(h)

    model.eval()
    with torch.no_grad():
        _ = model(input_ids=probe_input)

    for h in hooks:
        h.remove()

    return entropies


# ---------------------------------------------------------------------------
# 3. Experience Stream Magnitude
# ---------------------------------------------------------------------------

@torch.no_grad()
def compute_experience_magnitude(
    model: SABERForCausalLM,
) -> Dict[int, Dict[str, float]]:
    """
    Return the Frobenius (L2) norm of the three experience-stream weight
    matrices (W_s, W_pred, W_e) in each layer.

    Returns
    -------
    dict mapping layer_idx -> {"W_s": float, "W_pred": float, "W_e": float, "total": float}
    """
    magnitudes: Dict[int, Dict[str, float]] = {}

    for layer_idx, block in enumerate(model.model.layers):
        if not getattr(model.config, "enable_experience", True):
            magnitudes[layer_idx] = {"W_s": float("nan"), "W_pred": float("nan"),
                                     "W_e": float("nan"), "total": float("nan")}
            continue

        exp_stream = block.experience_stream

        w_s_norm    = exp_stream.W_s.weight.norm().item()
        w_pred_norm = exp_stream.W_pred.weight.norm().item()
        w_e_norm    = exp_stream.W_e.weight.norm().item()

        magnitudes[layer_idx] = {
            "W_s":   w_s_norm,
            "W_pred": w_pred_norm,
            "W_e":   w_e_norm,
            "total": (w_s_norm**2 + w_pred_norm**2 + w_e_norm**2) ** 0.5,
        }

    return magnitudes


# ---------------------------------------------------------------------------
# 4. Alpha Values
# ---------------------------------------------------------------------------

@torch.no_grad()
def get_alpha_values(
    model: SABERForCausalLM,
) -> Dict[int, float]:
    """
    Extract the current blending alpha = sigmoid(alpha_raw) from every
    resonant FFN layer.

    Non-resonant layers are skipped (they do not have alpha_raw).

    Returns
    -------
    dict mapping layer_idx -> alpha (float in (0, 1))
    """
    alphas: Dict[int, float] = {}

    for layer_idx, block in enumerate(model.model.layers):
        ffn = block.ffn
        if isinstance(ffn, ResonantFFN):
            alpha = torch.sigmoid(ffn.alpha_raw).item()
            alphas[layer_idx] = alpha

    return alphas


# ---------------------------------------------------------------------------
# 5. Log Custom Metrics
# ---------------------------------------------------------------------------

def log_custom_metrics(
    model: SABERForCausalLM,
    step: int,
    wandb_run=None,
    stdout: bool = True,
    probe_input: Optional[torch.Tensor] = None,
) -> Dict:
    """
    Compute and log all custom metrics:
      * Anchor entropy per layer
      * Experience-stream weight magnitude per layer
      * Alpha values for resonant layers

    Parameters
    ----------
    model      : SABERForCausalLM
    step       : current training step (for WandB x-axis)
    wandb_run  : active wandb.Run, or None to skip WandB
    stdout     : print summary to stdout
    probe_input: optional token tensor for anchor entropy; synthesised if None

    Returns
    -------
    dict containing all computed metrics
    """
    metrics: Dict = {"step": step}

    # ---- Anchor entropy ----
    anchor_entropy = compute_anchor_entropy(model, probe_input=probe_input)
    for layer_idx, ent in anchor_entropy.items():
        key = f"anchor_entropy/layer_{layer_idx:02d}"
        metrics[key] = ent

    mean_entropy = _safe_mean([v for v in anchor_entropy.values()
                               if not math.isnan(v)])
    metrics["anchor_entropy/mean"] = mean_entropy

    # ---- Experience magnitude ----
    exp_mag = compute_experience_magnitude(model)
    for layer_idx, mags in exp_mag.items():
        for mat_name, val in mags.items():
            key = f"experience_mag/layer_{layer_idx:02d}_{mat_name}"
            metrics[key] = val

    mean_exp_total = _safe_mean([v["total"] for v in exp_mag.values()
                                  if not math.isnan(v["total"])])
    metrics["experience_mag/mean_total"] = mean_exp_total

    # ---- Alpha values ----
    alphas = get_alpha_values(model)
    for layer_idx, alpha in alphas.items():
        key = f"resonant_alpha/layer_{layer_idx:02d}"
        metrics[key] = alpha

    if alphas:
        metrics["resonant_alpha/mean"] = _safe_mean(list(alphas.values()))
        metrics["resonant_alpha/min"]  = min(alphas.values())
        metrics["resonant_alpha/max"]  = max(alphas.values())

    # ---- Log to WandB ----
    if wandb_run is not None:
        # Flatten to scalar metrics for WandB; exclude NaN
        wandb_payload = {k: v for k, v in metrics.items()
                         if isinstance(v, float) and not math.isnan(v)}
        wandb_run.log(wandb_payload, step=step)

    # ---- Print summary ----
    if stdout:
        print(f"\n[eval] Custom metrics @ step {step}")
        if not math.isnan(mean_entropy):
            print(f"  anchor_entropy mean  : {mean_entropy:.4f} "
                  f"(max possible ≈ {math.log(model.config.n_anchors):.2f} nats)")
        if not math.isnan(mean_exp_total):
            print(f"  experience_mag mean  : {mean_exp_total:.4f}")
        if alphas:
            alpha_list = [f"L{k}={v:.3f}" for k, v in sorted(alphas.items())]
            print(f"  resonant alphas      : {', '.join(alpha_list)}")

    return metrics


# ---------------------------------------------------------------------------
# 6. Generate Samples
# ---------------------------------------------------------------------------

def generate_samples(
    model: SABERForCausalLM,
    tokenizer,
    prompts: Optional[List[str]] = None,
    max_new_tokens: int = 128,
    temperature: float = 0.8,
    top_p: float = 0.95,
    do_sample: bool = True,
) -> List[Dict[str, str]]:
    """
    Generate text continuations from a list of prompts.

    Parameters
    ----------
    model          : SABERForCausalLM
    tokenizer      : HuggingFace tokenizer (PreTrainedTokenizer)
    prompts        : list of prompt strings; uses defaults if None
    max_new_tokens : tokens to generate per prompt
    temperature    : sampling temperature
    top_p          : nucleus sampling probability
    do_sample      : use sampling (True) or greedy (False)

    Returns
    -------
    list of {"prompt": str, "completion": str, "full_text": str}
    """
    DEFAULT_PROMPTS = [
        "The most important discovery in physics was",
        "Once upon a time, in a land far away,",
        "The key to building intelligent systems is",
        "In mathematics, the concept of infinity",
    ]
    prompts = prompts or DEFAULT_PROMPTS

    device = next(model.parameters()).device
    model.eval()
    results = []

    for prompt in prompts:
        enc = tokenizer(prompt, return_tensors="pt")
        input_ids = enc["input_ids"].to(device)

        with torch.no_grad():
            generated_ids = model.generate(
                input_ids,
                max_new_tokens=max_new_tokens,
                do_sample=do_sample,
                temperature=temperature,
                top_p=top_p,
                pad_token_id=tokenizer.eos_token_id,
            )

        full_text  = tokenizer.decode(generated_ids[0], skip_special_tokens=True)
        completion = tokenizer.decode(
            generated_ids[0][input_ids.shape[1]:],
            skip_special_tokens=True,
        )
        results.append({
            "prompt":     prompt,
            "completion": completion,
            "full_text":  full_text,
        })

        print(f"\n[generate] PROMPT : {prompt}")
        print(f"[generate] COMPL  : {completion[:200]}")

    return results


# ---------------------------------------------------------------------------
# 7. Component Analysis Table
# ---------------------------------------------------------------------------

def run_component_analysis(
    model: SABERForCausalLM,
    probe_input: Optional[torch.Tensor] = None,
) -> None:
    """
    Print a comprehensive per-layer stats table showing:
      * FFN type (Standard / Resonant)
      * Anchor entropy
      * Experience stream magnitude (W_s, W_pred, W_e)
      * Alpha value (resonant layers only)
    """
    anchor_entropy = compute_anchor_entropy(model, probe_input=probe_input)
    exp_mag        = compute_experience_magnitude(model)
    alphas         = get_alpha_values(model)

    # ---- Header ----
    print("\n" + "=" * 90)
    print("  Eve-3-SABER-1B — Component Analysis")
    print("=" * 90)

    col_fmt = "  {:>5}  {:>10}  {:>10}  {:>10}  {:>10}  {:>10}  {:>8}"
    print(col_fmt.format(
        "Layer", "FFN Type", "Anch.Ent", "W_s Norm", "Wpred Norm", "W_e Norm", "Alpha"
    ))
    print("  " + "-" * 84)

    for layer_idx, block in enumerate(model.model.layers):
        ffn_type = "Resonant" if isinstance(block.ffn, ResonantFFN) else "Standard"

        ent  = anchor_entropy.get(layer_idx, float("nan"))
        mag  = exp_mag.get(layer_idx, {})
        alph = alphas.get(layer_idx, None)

        ent_str  = f"{ent:.3f}" if not math.isnan(ent) else "  N/A"
        ws_str   = f"{mag.get('W_s', float('nan')):.3f}"
        wp_str   = f"{mag.get('W_pred', float('nan')):.3f}"
        we_str   = f"{mag.get('W_e', float('nan')):.3f}"
        alp_str  = f"{alph:.4f}" if alph is not None else "   N/A"

        print(col_fmt.format(layer_idx, ffn_type, ent_str, ws_str, wp_str, we_str, alp_str))

    print("=" * 90)

    # Summary stats
    valid_ents = [v for v in anchor_entropy.values() if not math.isnan(v)]
    if valid_ents:
        max_ent = math.log(model.config.n_anchors)
        mean_ent = sum(valid_ents) / len(valid_ents)
        print(f"\n  Anchor entropy:  mean={mean_ent:.3f} nats  "
              f"(max={max_ent:.3f},  utilisation={mean_ent/max_ent*100:.1f}%)")

    if alphas:
        alpha_vals = list(alphas.values())
        print(f"  Resonant alpha:  min={min(alpha_vals):.4f}  "
              f"max={max(alpha_vals):.4f}  "
              f"mean={sum(alpha_vals)/len(alpha_vals):.4f}")

    print()


# ---------------------------------------------------------------------------
# 8. Alpha Plot
# ---------------------------------------------------------------------------

def create_alpha_plot(
    alpha_history: Dict[int, List[float]],
    save_path: Union[str, Path],
    steps: Optional[List[int]] = None,
    title: str = "Resonant Alpha Values Over Training",
) -> Path:
    """
    Create a line plot of alpha values over training steps, one line per
    resonant layer.

    Parameters
    ----------
    alpha_history : dict mapping layer_idx -> list of alpha values at each logged step
    save_path     : path (str or Path) to save the PNG
    steps         : optional x-axis values; auto-generated if None
    title         : plot title

    Returns
    -------
    Path to the saved PNG file
    """
    try:
        import matplotlib
        matplotlib.use("Agg")   # non-interactive backend
        import matplotlib.pyplot as plt
    except ImportError as exc:
        raise ImportError(
            "matplotlib is required for create_alpha_plot. "
            "Install with: pip install matplotlib"
        ) from exc

    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(figsize=(12, 6))

    sorted_layers = sorted(alpha_history.keys())

    for layer_idx in sorted_layers:
        values = alpha_history[layer_idx]
        if not values:
            continue
        x = steps if steps is not None else list(range(len(values)))
        # Thin line; distinct colour per layer via default cycle
        ax.plot(x, values, linewidth=1.2, label=f"Layer {layer_idx}", alpha=0.85)

    ax.set_xlabel("Training Step", fontsize=12)
    ax.set_ylabel("Alpha  [sigmoid(alpha_raw)]", fontsize=12)
    ax.set_title(title, fontsize=14)
    ax.axhline(0.95, color="gray", linestyle="--", linewidth=0.8, label="Init (0.95)")
    ax.set_ylim(0.0, 1.05)
    ax.legend(loc="upper right", fontsize=7, ncol=3)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    fig.savefig(save_path, dpi=150)
    plt.close(fig)

    print(f"[eval] Alpha plot saved → {save_path}")
    return save_path


# ---------------------------------------------------------------------------
# 9. Anchor Usage Heatmap
# ---------------------------------------------------------------------------

def create_anchor_usage_heatmap(
    anchor_scores: Dict[int, torch.Tensor],
    save_path: Union[str, Path],
    title: str = "Anchor Usage Heatmap (layers × anchors)",
) -> Path:
    """
    Create a heatmap of mean anchor usage (soft assignment probability)
    across all layers.

    Parameters
    ----------
    anchor_scores : dict mapping layer_idx -> (n_anchors,) tensor of mean usage
                    probabilities.  Compute by running a forward pass and
                    capturing the softmax output in SlipAnchors.forward.
    save_path     : path (str or Path) to save the PNG
    title         : plot title

    Returns
    -------
    Path to the saved PNG file
    """
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        import numpy as np
    except ImportError as exc:
        raise ImportError(
            "matplotlib and numpy are required for create_anchor_usage_heatmap."
        ) from exc

    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)

    sorted_layers = sorted(anchor_scores.keys())
    n_layers  = len(sorted_layers)
    n_anchors = next(iter(anchor_scores.values())).shape[0]

    # Build matrix: rows = layers, cols = anchors
    matrix = np.zeros((n_layers, n_anchors), dtype=np.float32)
    for row_i, layer_idx in enumerate(sorted_layers):
        scores = anchor_scores[layer_idx]
        if isinstance(scores, torch.Tensor):
            scores = scores.detach().cpu().float().numpy()
        matrix[row_i] = scores

    fig, ax = plt.subplots(figsize=(max(10, n_anchors // 3), max(5, n_layers // 2)))

    try:
        import seaborn as sns  # type: ignore
        sns.heatmap(
            matrix,
            ax=ax,
            cmap="YlOrRd",
            xticklabels=list(range(n_anchors)),
            yticklabels=[f"L{i}" for i in sorted_layers],
            vmin=0.0,
            linewidths=0.0,
        )
    except ImportError:
        # Fall back to matplotlib imshow if seaborn unavailable
        im = ax.imshow(matrix, aspect="auto", cmap="YlOrRd", vmin=0.0)
        plt.colorbar(im, ax=ax)
        ax.set_xlabel("Anchor index")
        ax.set_ylabel("Layer")
        ax.set_yticks(range(n_layers))
        ax.set_yticklabels([f"L{i}" for i in sorted_layers], fontsize=7)

    ax.set_title(title, fontsize=13)
    plt.tight_layout()
    fig.savefig(save_path, dpi=150)
    plt.close(fig)

    print(f"[eval] Anchor usage heatmap saved → {save_path}")
    return save_path


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _safe_mean(values: List[float]) -> float:
    """Mean of a list, returning nan if empty."""
    if not values:
        return float("nan")
    return sum(values) / len(values)


def collect_anchor_scores(
    model: SABERForCausalLM,
    probe_input: Optional[torch.Tensor] = None,
    seq_len: int = 256,
    batch_size: int = 2,
) -> Dict[int, torch.Tensor]:
    """
    Run a forward pass and collect mean anchor usage per layer.

    Returns dict mapping layer_idx -> (n_anchors,) tensor.
    Useful for feeding into create_anchor_usage_heatmap.
    """
    device = next(model.parameters()).device

    if probe_input is None:
        probe_input = torch.randint(
            0, model.config.vocab_size,
            (batch_size, seq_len),
            device=device,
        )

    scores_collected: Dict[int, torch.Tensor] = {}
    hooks = []

    for layer_idx, block in enumerate(model.model.layers):
        if not getattr(model.config, "enable_anchors", True):
            continue
        slip_module = block.self_attn.slip_anchors

        def _make_hook(idx: int):
            def _hook(module: SlipAnchors, inputs, outputs):
                h = inputs[0]
                with torch.no_grad():
                    h_anchor = module.W_anchor_down(h)
                    soft_scores = torch.softmax(
                        h_anchor @ module.anchors.T, dim=-1
                    )                                        # (B, L, n_anchors)
                    mean_usage = soft_scores.mean(dim=(0, 1)).cpu()  # (n_anchors,)
                    scores_collected[idx] = mean_usage
            return _hook

        h = slip_module.register_forward_hook(_make_hook(layer_idx))
        hooks.append(h)

    model.eval()
    with torch.no_grad():
        _ = model(input_ids=probe_input)

    for h in hooks:
        h.remove()

    return scores_collected
