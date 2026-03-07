"""
ablation_runner.py — Orchestrates the 5-run ablation study for Eve-3-SABER-1B.

Ablation plan (from GPT-5.2 Thinking):
  Run 1  baseline   Standard transformer — all novel components disabled
  Run 2  anchors    +Slip-Anchors only
  Run 3  experience +Experience stream + curiosity loss only
  Run 4  resonant   +Resonant FFN only
  Run 5  full       All components enabled (default config)

Usage:
    # Run a single ablation
    python ablation_runner.py --ablation baseline
    python ablation_runner.py --ablation full --tokens 2_000_000_000

    # Run all five sequentially and print comparison table
    python ablation_runner.py --ablation all

    # Dry-run (build models, skip training)
    python ablation_runner.py --ablation all --dry-run

    # Resume a specific run
    python ablation_runner.py --ablation anchors --resume-from ./ablation_results/anchors/checkpoint-last

Environment variables:
    WANDB_PROJECT   WandB project name (default: eve-3-saber-1b-ablations)
    WANDB_ENTITY    WandB entity / team
    WANDB_API_KEY   WandB API key (or login beforehand)
"""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Dict, List, Optional

import torch

# ---------------------------------------------------------------------------
# Local imports (must be on PYTHONPATH or same directory)
# ---------------------------------------------------------------------------
sys.path.insert(0, str(Path(__file__).parent))

from configuration_saber import SABERConfig
from modeling_saber import SABERForCausalLM


# ---------------------------------------------------------------------------
# Hardware detection (mirrors train_eve3_saber_1b.ipynb)
# ---------------------------------------------------------------------------

def detect_hardware() -> Dict:
    """Auto-detect available training hardware and return config dict."""
    info: Dict = {
        "device": "cpu",
        "n_gpus": 0,
        "dtype": torch.float32,
        "dtype_name": "float32",
        "use_flash_attn": False,
        "use_amp": False,
    }

    if torch.cuda.is_available():
        info["n_gpus"] = torch.cuda.device_count()
        info["device"] = "cuda"
        info["dtype"] = torch.bfloat16
        info["dtype_name"] = "bfloat16"
        info["use_amp"] = True

        # Check for FlashAttention 2 support (Ampere or newer, sm80+)
        cap = torch.cuda.get_device_capability()
        if cap[0] >= 8:
            info["use_flash_attn"] = True

        gpu_names = [torch.cuda.get_device_name(i) for i in range(info["n_gpus"])]
        info["gpu_names"] = gpu_names
        print(f"[hardware] Found {info['n_gpus']} GPU(s): {', '.join(gpu_names)}")
        print(f"[hardware] Using dtype={info['dtype_name']}, "
              f"FlashAttn={info['use_flash_attn']}")

    elif torch.backends.mps.is_available():
        info["device"] = "mps"
        info["dtype"] = torch.float32  # MPS doesn't support bfloat16 fully yet
        info["dtype_name"] = "float32"
        print("[hardware] Using Apple MPS backend")

    else:
        print("[hardware] WARNING: No GPU found — running on CPU (very slow)")

    return info


# ---------------------------------------------------------------------------
# Ablation configuration
# ---------------------------------------------------------------------------

ABLATION_NAMES = ["baseline", "anchors", "experience", "resonant", "full"]

@dataclass
class AblationSpec:
    """Defines which novel components are active for a given run."""
    name: str
    enable_anchors: bool
    enable_experience: bool
    enable_resonant: bool
    description: str

    def to_config_kwargs(self, base_config: SABERConfig) -> Dict:
        """
        Return SABERConfig constructor kwargs that implement this ablation.

        Strategy: use boolean flags injected into the config object after
        construction (config.enable_anchors etc.) rather than trying to set
        n_anchors=0, which would break tensor shapes.
        """
        kwargs: Dict = {}

        # Resonant: empty list disables all resonant layers
        if not self.enable_resonant:
            kwargs["resonant_layers"] = []

        return kwargs


ABLATION_SPECS: Dict[str, AblationSpec] = {
    "baseline": AblationSpec(
        name="baseline",
        enable_anchors=False,
        enable_experience=False,
        enable_resonant=False,
        description="Standard transformer — all novel components disabled",
    ),
    "anchors": AblationSpec(
        name="anchors",
        enable_anchors=True,
        enable_experience=False,
        enable_resonant=False,
        description="+Slip-Anchors only",
    ),
    "experience": AblationSpec(
        name="experience",
        enable_anchors=False,
        enable_experience=True,
        enable_resonant=False,
        description="+Experience stream + curiosity loss only",
    ),
    "resonant": AblationSpec(
        name="resonant",
        enable_anchors=False,
        enable_experience=False,
        enable_resonant=True,
        description="+Resonant FFN only (even layers)",
    ),
    "full": AblationSpec(
        name="full",
        enable_anchors=True,
        enable_experience=True,
        enable_resonant=True,
        description="All components enabled — full Eve-3-SABER-1B",
    ),
}


# ---------------------------------------------------------------------------
# Model builder
# ---------------------------------------------------------------------------

def build_ablation_model(spec: AblationSpec, d_ff: int = 2855) -> SABERForCausalLM:
    """
    Construct a SABERForCausalLM with the component flags from *spec*.

    Component short-circuiting is implemented via config flags
    (enable_anchors, enable_experience, enable_resonant) that the modeling
    code checks before running the corresponding module.  This avoids shape
    mismatches from setting n_anchors=0 or d_exp=0.
    """
    config_kwargs = spec.to_config_kwargs(SABERConfig())

    config = SABERConfig(
        d_ff=d_ff,
        **config_kwargs,
    )

    # Attach ablation flags to config (PretrainedConfig accepts arbitrary attrs)
    config.enable_anchors    = spec.enable_anchors
    config.enable_experience = spec.enable_experience
    config.enable_resonant   = spec.enable_resonant  # redundant — resonant_layers handles it

    model = SABERForCausalLM(config)
    return model


def count_params(model: torch.nn.Module) -> int:
    return sum(p.numel() for p in model.parameters())


# ---------------------------------------------------------------------------
# Minimal dataloader (synthetic tokens for ablation purposes)
# ---------------------------------------------------------------------------

def make_synthetic_dataloader(
    total_tokens: int,
    seq_len: int = 2048,
    batch_size: int = 4,
    vocab_size: int = 50257,
    device: str = "cpu",
):
    """
    Yield batches of random token ids until *total_tokens* are consumed.

    For real training, replace with a proper streaming dataloader over
    FineWeb-Edu / DCLM (as in train_eve3_saber_1b.ipynb).
    """
    tokens_yielded = 0
    while tokens_yielded < total_tokens:
        ids = torch.randint(
            0, vocab_size,
            (batch_size, seq_len),
            device=device,
        )
        yield ids
        tokens_yielded += batch_size * seq_len


# ---------------------------------------------------------------------------
# Training loop (minimal; for ablations we care about loss trajectory)
# ---------------------------------------------------------------------------

@dataclass
class AblationResult:
    name: str
    description: str
    n_params: int
    total_tokens_seen: int
    final_loss: float
    final_perplexity: float
    best_loss: float
    loss_curve: List[float] = field(default_factory=list)
    wall_time_s: float = 0.0
    error: Optional[str] = None


def run_ablation(
    spec: AblationSpec,
    args: argparse.Namespace,
    hw: Dict,
    wandb_run=None,
) -> AblationResult:
    """Train one ablation for *args.tokens* tokens and return metrics."""

    print(f"\n{'='*60}")
    print(f"  ABLATION: {spec.name.upper()}")
    print(f"  {spec.description}")
    print(f"{'='*60}")

    t0 = time.time()
    result = AblationResult(
        name=spec.name,
        description=spec.description,
        n_params=0,
        total_tokens_seen=0,
        final_loss=float("nan"),
        final_perplexity=float("nan"),
        best_loss=float("inf"),
    )

    try:
        # ---- Build model ----
        model = build_ablation_model(spec, d_ff=args.d_ff)
        result.n_params = count_params(model)
        print(f"[{spec.name}] Model params: {result.n_params:,}")

        device = hw["device"]
        model = model.to(device)

        if args.dry_run:
            print(f"[{spec.name}] --dry-run: skipping training")
            result.final_loss = 0.0
            result.final_perplexity = 1.0
            result.wall_time_s = time.time() - t0
            return result

        # ---- Optimizer ----
        optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=args.lr,
            betas=(0.9, 0.95),
            weight_decay=0.1,
        )

        # ---- LR scheduler (cosine with linear warmup) ----
        total_steps = args.tokens // (args.batch_size * args.seq_len)
        warmup_steps = min(args.warmup_steps, total_steps // 10)

        def lr_lambda(step: int) -> float:
            if step < warmup_steps:
                return step / max(1, warmup_steps)
            progress = (step - warmup_steps) / max(1, total_steps - warmup_steps)
            return max(args.min_lr_ratio, 0.5 * (1.0 + math.cos(math.pi * progress)))

        scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

        # ---- Resume from checkpoint ----
        start_step = 0
        if args.resume_from:
            ckpt_path = Path(args.resume_from)
            if ckpt_path.exists():
                ckpt = torch.load(ckpt_path / "trainer_state.pt", map_location=device)
                model.load_state_dict(torch.load(ckpt_path / "model.pt", map_location=device))
                optimizer.load_state_dict(ckpt["optimizer"])
                scheduler.load_state_dict(ckpt["scheduler"])
                start_step = ckpt["step"]
                print(f"[{spec.name}] Resumed from step {start_step}")

        # ---- Mixed precision ----
        scaler = None
        if hw["use_amp"] and device == "cuda":
            scaler = torch.cuda.amp.GradScaler()

        # ---- Training loop ----
        model.train()
        log_interval  = max(1, total_steps // 200)
        save_interval = max(1, total_steps // 10)
        grad_accum    = args.grad_accum

        save_dir = Path(args.results_dir) / spec.name
        save_dir.mkdir(parents=True, exist_ok=True)

        step = start_step
        tokens_seen = 0
        loss_sum = 0.0
        loss_count = 0

        optimizer.zero_grad()

        for batch_ids in make_synthetic_dataloader(
            total_tokens=args.tokens,
            seq_len=args.seq_len,
            batch_size=args.batch_size,
            vocab_size=model.config.vocab_size,
            device=device,
        ):
            # ---- Forward ----
            if hw["use_amp"] and device == "cuda":
                with torch.autocast(device_type="cuda", dtype=hw["dtype"]):
                    out = model(input_ids=batch_ids, labels=batch_ids)
            else:
                out = model(input_ids=batch_ids, labels=batch_ids)

            loss = out.loss / grad_accum

            # ---- Backward ----
            if scaler is not None:
                scaler.scale(loss).backward()
            else:
                loss.backward()

            # ---- Gradient step ----
            if (step + 1) % grad_accum == 0:
                if scaler is not None:
                    scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                if scaler is not None:
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    optimizer.step()
                scheduler.step()
                optimizer.zero_grad()

            raw_loss = loss.item() * grad_accum
            loss_sum   += raw_loss
            loss_count += 1
            tokens_seen += args.batch_size * args.seq_len
            result.loss_curve.append(raw_loss)

            if raw_loss < result.best_loss:
                result.best_loss = raw_loss

            # ---- Logging ----
            if step % log_interval == 0:
                avg_loss = loss_sum / loss_count
                ppl      = math.exp(min(avg_loss, 20))
                lr_now   = scheduler.get_last_lr()[0]
                print(
                    f"[{spec.name}] step={step:>6d}  loss={avg_loss:.4f}  "
                    f"ppl={ppl:.2f}  lr={lr_now:.2e}  "
                    f"tokens={tokens_seen/1e6:.1f}M"
                )
                loss_sum  = 0.0
                loss_count = 0

                if wandb_run is not None:
                    wandb_run.log(
                        {
                            f"{spec.name}/loss":         avg_loss,
                            f"{spec.name}/perplexity":   ppl,
                            f"{spec.name}/lr":           lr_now,
                            f"{spec.name}/tokens_seen":  tokens_seen,
                        },
                        step=step,
                    )

            # ---- Checkpoint ----
            if step % save_interval == 0 and step > start_step:
                ckpt_dir = save_dir / f"checkpoint-{step}"
                ckpt_dir.mkdir(parents=True, exist_ok=True)
                torch.save(model.state_dict(), ckpt_dir / "model.pt")
                torch.save(
                    {
                        "step":      step,
                        "optimizer": optimizer.state_dict(),
                        "scheduler": scheduler.state_dict(),
                    },
                    ckpt_dir / "trainer_state.pt",
                )
                print(f"[{spec.name}] Saved checkpoint → {ckpt_dir}")

            step += 1

        # ---- Final metrics ----
        result.total_tokens_seen = tokens_seen
        if result.loss_curve:
            # Average of last 5% of steps for final loss
            tail = max(1, len(result.loss_curve) // 20)
            result.final_loss = sum(result.loss_curve[-tail:]) / tail
            result.final_perplexity = math.exp(min(result.final_loss, 20))

        # ---- Save final results JSON ----
        results_path = save_dir / "results.json"
        with open(results_path, "w") as fh:
            json.dump(
                {
                    "name":               result.name,
                    "description":        result.description,
                    "n_params":           result.n_params,
                    "total_tokens_seen":  result.total_tokens_seen,
                    "final_loss":         result.final_loss,
                    "final_perplexity":   result.final_perplexity,
                    "best_loss":          result.best_loss,
                    "wall_time_s":        time.time() - t0,
                    # Save every 10th point to keep file manageable
                    "loss_curve_sampled": result.loss_curve[::10],
                },
                fh,
                indent=2,
            )
        print(f"[{spec.name}] Results saved → {results_path}")

        if wandb_run is not None:
            wandb_run.summary[f"{spec.name}/final_loss"]       = result.final_loss
            wandb_run.summary[f"{spec.name}/final_perplexity"] = result.final_perplexity
            wandb_run.summary[f"{spec.name}/best_loss"]        = result.best_loss

    except Exception as exc:  # noqa: BLE001
        import traceback
        result.error = traceback.format_exc()
        print(f"[{spec.name}] ERROR: {exc}")
        print(result.error)

    result.wall_time_s = time.time() - t0
    return result


# ---------------------------------------------------------------------------
# Comparison table
# ---------------------------------------------------------------------------

def print_comparison_table(results: List[AblationResult]) -> None:
    """Print a formatted comparison table to stdout."""
    header = (
        f"\n{'='*80}\n"
        f"  ABLATION STUDY — COMPARISON TABLE\n"
        f"{'='*80}\n"
    )
    print(header)

    col_w = [18, 44, 14, 14, 14, 12]
    fmt   = "  {:<{}} {:<{}} {:>{}} {:>{}} {:>{}} {:>{}}"

    print(fmt.format(
        "Run",         col_w[0],
        "Description", col_w[1],
        "Final Loss",  col_w[2],
        "Perplexity",  col_w[3],
        "Best Loss",   col_w[4],
        "Time (s)",    col_w[5],
    ))
    print("  " + "-" * (sum(col_w) + len(col_w) * 2))

    for r in results:
        err_tag = " [ERROR]" if r.error else ""
        print(fmt.format(
            r.name + err_tag,                    col_w[0],
            r.description[:col_w[1]],            col_w[1],
            f"{r.final_loss:.4f}",               col_w[2],
            f"{r.final_perplexity:.2f}",         col_w[3],
            f"{r.best_loss:.4f}",                col_w[4],
            f"{r.wall_time_s:.0f}",              col_w[5],
        ))

    # Delta vs baseline
    baseline_result = next((r for r in results if r.name == "baseline"), None)
    if baseline_result and not math.isnan(baseline_result.final_loss):
        print(f"\n  Delta vs. baseline (↓ = better):")
        print("  " + "-" * 40)
        for r in results:
            if r.name == "baseline" or math.isnan(r.final_loss):
                continue
            delta = r.final_loss - baseline_result.final_loss
            sign  = "↓" if delta < 0 else "↑"
            print(f"  {r.name:<18}  {sign} {abs(delta):.4f}  ({delta:+.4f})")

    print(f"\n{'='*80}\n")

    # Save table to results dir
    table_path = Path(args.results_dir) / "comparison_table.txt"
    # (args is captured from outer scope via global; handled in __main__)


# ---------------------------------------------------------------------------
# WandB setup
# ---------------------------------------------------------------------------

def init_wandb(args: argparse.Namespace, ablation_name: str):
    """Initialise a WandB run for a single ablation.  Returns run or None."""
    if not args.wandb:
        return None
    try:
        import wandb  # type: ignore
        run = wandb.init(
            project=os.environ.get("WANDB_PROJECT", "eve-3-saber-1b-ablations"),
            entity=os.environ.get("WANDB_ENTITY", None),
            name=f"saber-ablation-{ablation_name}",
            tags=["ablation", ablation_name, f"tokens={args.tokens:.0e}"],
            config={
                "ablation":    ablation_name,
                "tokens":      args.tokens,
                "d_ff":        args.d_ff,
                "seq_len":     args.seq_len,
                "batch_size":  args.batch_size,
                "lr":          args.lr,
                "grad_accum":  args.grad_accum,
            },
            reinit=True,
        )
        return run
    except Exception as exc:  # noqa: BLE001
        print(f"[wandb] WARNING: Could not initialise WandB run: {exc}")
        return None


# ---------------------------------------------------------------------------
# Argument parsing
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Eve-3-SABER-1B Ablation Runner",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--ablation",
        choices=ABLATION_NAMES + ["all"],
        default="all",
        help="Which ablation to run. Use 'all' to run all five sequentially.",
    )
    parser.add_argument(
        "--tokens",
        type=float,
        default=1e9,
        help="Token budget per ablation run (e.g. 1e9 = 1B tokens).",
    )
    parser.add_argument(
        "--d-ff",
        type=int,
        default=2855,
        dest="d_ff",
        help="SwiGLU inner dimension. 2855 → exactly 1,000,001,548 params.",
    )
    parser.add_argument(
        "--seq-len",
        type=int,
        default=2048,
        dest="seq_len",
        help="Sequence length per sample.",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=4,
        dest="batch_size",
        help="Micro-batch size (sequences per step).",
    )
    parser.add_argument(
        "--grad-accum",
        type=int,
        default=8,
        dest="grad_accum",
        help="Gradient accumulation steps.",
    )
    parser.add_argument(
        "--lr",
        type=float,
        default=3e-4,
        help="Peak learning rate.",
    )
    parser.add_argument(
        "--min-lr-ratio",
        type=float,
        default=0.1,
        dest="min_lr_ratio",
        help="min_lr / max_lr for cosine schedule.",
    )
    parser.add_argument(
        "--warmup-steps",
        type=int,
        default=500,
        dest="warmup_steps",
        help="Number of linear warmup steps.",
    )
    parser.add_argument(
        "--results-dir",
        type=str,
        default=str(Path(__file__).parent / "ablation_results"),
        dest="results_dir",
        help="Directory where per-ablation results are saved.",
    )
    parser.add_argument(
        "--resume-from",
        type=str,
        default=None,
        dest="resume_from",
        help="Path to a checkpoint directory to resume from (single ablation only).",
    )
    parser.add_argument(
        "--wandb",
        action="store_true",
        default=False,
        help="Enable WandB logging.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        default=False,
        dest="dry_run",
        help="Build models and validate config, but skip training.",
    )
    return parser.parse_args()


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    args = parse_args()

    # Make args.tokens an int for arithmetic
    args.tokens = int(args.tokens)

    hw = detect_hardware()

    # Determine which ablations to run
    if args.ablation == "all":
        to_run = ABLATION_NAMES
    else:
        to_run = [args.ablation]

    results: List[AblationResult] = []

    for ablation_name in to_run:
        spec       = ABLATION_SPECS[ablation_name]
        wandb_run  = init_wandb(args, ablation_name)

        result = run_ablation(spec, args, hw, wandb_run=wandb_run)
        results.append(result)

        if wandb_run is not None:
            wandb_run.finish()

    # ---- Print comparison table when running all ablations ----
    if len(results) > 1:
        print_comparison_table(results)

        # Persist JSON summary of all runs
        summary_path = Path(args.results_dir) / "ablation_summary.json"
        summary_path.parent.mkdir(parents=True, exist_ok=True)
        with open(summary_path, "w") as fh:
            json.dump(
                [
                    {
                        "name":              r.name,
                        "description":       r.description,
                        "n_params":          r.n_params,
                        "tokens_seen":       r.total_tokens_seen,
                        "final_loss":        r.final_loss,
                        "final_perplexity":  r.final_perplexity,
                        "best_loss":         r.best_loss,
                        "wall_time_s":       r.wall_time_s,
                        "error":             r.error,
                    }
                    for r in results
                ],
                fh,
                indent=2,
            )
        print(f"Summary saved → {summary_path}")

    # Exit with non-zero status if any run errored
    if any(r.error for r in results):
        sys.exit(1)
