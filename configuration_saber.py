"""
configuration_saber.py — HuggingFace-compatible configuration for Eve-3-SABER-1B.

Usage:
    from configuration_saber import SABERConfig

    config = SABERConfig()                 # default 1B spec
    config.save_pretrained("./eve-3-saber-1b")
    config = SABERConfig.from_pretrained("./eve-3-saber-1b")
"""

from __future__ import annotations

from typing import List, Optional

from transformers import PretrainedConfig


class SABERConfig(PretrainedConfig):
    r"""
    Configuration class for Eve-3-SABER-1B.

    SABER (Semantic Anchor-Biased Experience-Resonant) is a dense decoder-only
    transformer with three novel components:

    1. **Slip-Anchors** — a per-layer learnable codebook that biases K and V
       *after* RoPE, preserving FlashAttention compatibility.
    2. **Experience Stream** — a low-dimensional per-token state that flows
       *layer-to-layer* (not token-to-token), with a curiosity auxiliary loss.
    3. **Resonant FFN** — even-numbered layers augment SwiGLU with a learned
       sinusoidal modulation, blended via a trainable alpha.

    Args:
        vocab_size (int):
            Vocabulary size. Defaults to ``50257`` (GPT-2 tokenizer).
        d_model (int):
            Hidden/residual dimension. Defaults to ``2048``.
        n_heads (int):
            Number of attention heads. Defaults to ``16``.
        head_dim (int):
            Per-head dimension; must satisfy ``d_model == n_heads * head_dim``.
            Defaults to ``128``.
        n_layers (int):
            Number of transformer blocks. Defaults to ``24``.
        d_ff (int):
            SwiGLU inner dimension. The spec value ``5461`` yields ~1.38B params;
            use ``2855`` (tuned via ``param_counter.py --tune-dff``) to hit
            exactly 1.0B. Defaults to ``5461`` (spec) so the number is always
            explicit and reviewable.
        max_position_embeddings (int):
            Maximum sequence length for RoPE. Defaults to ``2048``.
        rope_theta (float):
            Base for RoPE frequency computation. Defaults to ``10000.0``.
        rms_norm_eps (float):
            Epsilon for RMSNorm numerical stability. Defaults to ``1e-6``.
        initializer_range (float):
            Std-dev for weight initialization (Normal). Defaults to ``0.02``.
        tie_word_embeddings (bool):
            Whether to tie the LM head weights to the input embedding table.
            Defaults to ``True``.

        --- Slip-Anchor hyperparameters ---
        n_anchors (int):
            Codebook size. Defaults to ``64``.
        d_anchor (int):
            Anchor bottleneck dimension. Defaults to ``128``.

        --- Experience-stream hyperparameters ---
        d_exp (int):
            Experience stream dimension. Defaults to ``256``.
        curiosity_coeff (float):
            Weight of curiosity auxiliary loss. Defaults to ``0.01``.

        --- Resonant-FFN hyperparameters ---
        resonant_layers (Optional[List[int]]):
            Which layer indices use the resonant FFN.  ``None`` means "all even
            layers (0, 2, 4, …)".  Pass an explicit list to override (e.g. last
            8 layers only for predictability mode).
        resonant_alpha_init (float):
            Initial value of ``alpha_raw`` before sigmoid; ``sigmoid(3.0)≈0.95``
            starts training near pure SwiGLU. Defaults to ``3.0``.

        --- Predictability mode (GPT-5.2 Thinking) ---
        predictability_mode (bool):
            When ``True`` the following overrides are applied at model
            construction time:
            * Anchor gate bias → ``-3`` (anchors nearly silent).
            * ``U_e`` scale → ``0.05`` (tiny experience updates).
            * ``resonant_layers`` → last 8 layers only.
            Defaults to ``False``.

        --- Gradient checkpointing ---
        use_cache (bool):
            Whether past KV states are returned (not used during training).
            Defaults to ``True``.
        gradient_checkpointing (bool):
            Enable activation checkpointing.  Set via
            ``model.gradient_checkpointing_enable()`` rather than here in most
            cases. Defaults to ``False``.
    """

    # Required by HuggingFace AutoModel registry
    model_type: str = "saber"

    # Map canonical HF attribute names to SABER field names so that
    # generic HF utilities (e.g. model.config.hidden_size) work transparently.
    attribute_map = {
        "hidden_size":              "d_model",
        "num_hidden_layers":        "n_layers",
        "num_attention_heads":      "n_heads",
        "intermediate_size":        "d_ff",
        "max_position_embeddings":  "max_position_embeddings",
    }

    def __init__(
        self,
        # Core architecture
        vocab_size: int = 50257,
        d_model: int = 2048,
        n_heads: int = 16,
        head_dim: int = 128,
        n_layers: int = 24,
        d_ff: int = 2855,
        max_position_embeddings: int = 2048,
        rope_theta: float = 10_000.0,
        rms_norm_eps: float = 1e-6,
        initializer_range: float = 0.02,
        tie_word_embeddings: bool = True,
        # Slip-anchor
        n_anchors: int = 64,
        d_anchor: int = 128,
        # Experience stream
        d_exp: int = 256,
        curiosity_coeff: float = 0.01,
        # Resonant FFN
        resonant_layers: Optional[List[int]] = None,
        resonant_alpha_init: float = 3.0,
        # Predictability mode
        predictability_mode: bool = False,
        # Inference / training toggles
        use_cache: bool = True,
        gradient_checkpointing: bool = False,
        # Ablation flags (component enable/disable)
        enable_anchors: bool = True,
        enable_experience: bool = True,
        **kwargs,
    ) -> None:
        # ------------------------------------------------------------------ #
        # Validate key relationships
        # ------------------------------------------------------------------ #
        if d_model != n_heads * head_dim:
            raise ValueError(
                f"d_model ({d_model}) must equal n_heads ({n_heads}) × "
                f"head_dim ({head_dim}) = {n_heads * head_dim}."
            )

        # ------------------------------------------------------------------ #
        # Core
        # ------------------------------------------------------------------ #
        self.vocab_size             = vocab_size
        self.d_model                = d_model
        self.n_heads                = n_heads
        self.head_dim               = head_dim
        self.n_layers               = n_layers
        self.d_ff                   = d_ff
        self.max_position_embeddings = max_position_embeddings
        self.rope_theta             = rope_theta
        self.rms_norm_eps           = rms_norm_eps
        self.initializer_range      = initializer_range

        # ------------------------------------------------------------------ #
        # Slip-anchor
        # ------------------------------------------------------------------ #
        self.n_anchors   = n_anchors
        self.d_anchor    = d_anchor

        # ------------------------------------------------------------------ #
        # Experience stream
        # ------------------------------------------------------------------ #
        self.d_exp          = d_exp
        self.curiosity_coeff = curiosity_coeff

        # ------------------------------------------------------------------ #
        # Resonant FFN — default to all even layers
        # ------------------------------------------------------------------ #
        if resonant_layers is None:
            resonant_layers = [i for i in range(n_layers) if i % 2 == 0]
        self.resonant_layers    = resonant_layers
        self.resonant_alpha_init = resonant_alpha_init

        # ------------------------------------------------------------------ #
        # Predictability mode overrides
        # ------------------------------------------------------------------ #
        self.predictability_mode = predictability_mode
        if predictability_mode:
            # Last 8 layers only
            self.resonant_layers = list(range(n_layers - 8, n_layers))

        # ------------------------------------------------------------------ #
        # Inference / training
        # ------------------------------------------------------------------ #
        self.use_cache              = use_cache
        self.gradient_checkpointing = gradient_checkpointing

        # Ablation flags — allow disabling novel components
        self.enable_anchors    = enable_anchors
        self.enable_experience = enable_experience

        # ------------------------------------------------------------------ #
        # Pass through to PretrainedConfig (handles tie_word_embeddings, etc.)
        # ------------------------------------------------------------------ #
        super().__init__(
            tie_word_embeddings=tie_word_embeddings,
            **kwargs,
        )

    # ---------------------------------------------------------------------- #
    # Derived helpers (read-only properties, not serialized)
    # ---------------------------------------------------------------------- #

    @property
    def num_key_value_heads(self) -> int:
        """Alias for n_heads (SABER uses MHA, not GQA)."""
        return self.n_heads

    @property
    def n_resonant_layers(self) -> int:
        """Number of layers that use the resonant FFN."""
        return len(self.resonant_layers)

    def __repr__(self) -> str:  # noqa: D401
        resonant_str = (
            f"all-even (n={self.n_resonant_layers})"
            if self.resonant_layers == [i for i in range(self.n_layers) if i % 2 == 0]
            else str(self.resonant_layers)
        )
        return (
            f"SABERConfig(\n"
            f"  d_model={self.d_model}, n_heads={self.n_heads}, "
            f"head_dim={self.head_dim}, n_layers={self.n_layers},\n"
            f"  d_ff={self.d_ff}, vocab_size={self.vocab_size}, "
            f"max_seq={self.max_position_embeddings},\n"
            f"  n_anchors={self.n_anchors}, d_anchor={self.d_anchor}, "
            f"d_exp={self.d_exp},\n"
            f"  curiosity_coeff={self.curiosity_coeff}, "
            f"resonant_layers={resonant_str},\n"
            f"  resonant_alpha_init={self.resonant_alpha_init}, "
            f"predictability_mode={self.predictability_mode},\n"
            f"  tie_word_embeddings={self.tie_word_embeddings}, "
            f"use_cache={self.use_cache}\n"
            f")"
        )
