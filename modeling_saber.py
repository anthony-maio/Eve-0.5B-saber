"""
modeling_saber.py — Full PyTorch implementation of Eve-3-SABER-1B.

Architecture highlights
-----------------------
* Dense decoder-only transformer with pre-RMSNorm.
* RoPE (rotary position embeddings) applied to Q and K after head reshape.
* **Slip-Anchors**: learnable codebook biases K/V *after* RoPE, fully
  compatible with FlashAttention / F.scaled_dot_product_attention.
* **Experience Stream**: a per-token, layer-traversing state with a curiosity
  auxiliary loss (prediction-error on a stop-gradient summary).
* **Resonant FFN**: even-indexed layers augment SwiGLU with a learned
  sinusoidal modulation blended by a trainable scalar alpha.
* Weight-tied LM head.
* Gradient-checkpointing support.

Intended usage (HuggingFace Trainer / SFTTrainer compatible):
    from configuration_saber import SABERConfig
    from modeling_saber import SABERForCausalLM

    config = SABERConfig()
    model  = SABERForCausalLM(config)
"""

from __future__ import annotations

import math
from typing import List, Optional, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.checkpoint

from transformers import PreTrainedModel
from transformers.generation import GenerationMixin
from transformers.modeling_outputs import CausalLMOutputWithPast, BaseModelOutputWithPast
from transformers.utils import logging

from configuration_saber import SABERConfig

logger = logging.get_logger(__name__)

# ---------------------------------------------------------------------------
# 1. RMSNorm
# ---------------------------------------------------------------------------

class SABERRMSNorm(nn.Module):
    """Root-mean-square layer normalization (no bias, learnable scale)."""

    def __init__(self, hidden_size: int, eps: float = 1e-6) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.eps    = eps

    def _norm(self, x: torch.Tensor) -> torch.Tensor:
        # x: (..., hidden_size)
        return x * torch.rsqrt(x.pow(2).mean(dim=-1, keepdim=True) + self.eps)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Cast to float for numerical stability, then back to input dtype
        return (self._norm(x.float()) * self.weight.float()).to(x.dtype)


# ---------------------------------------------------------------------------
# 2. Rotary Position Embeddings (RoPE)
# ---------------------------------------------------------------------------

class SABERRotaryEmbedding(nn.Module):
    """
    Standard RoPE using complex-number rotation (Llama / GPT-NeoX style).

    Frequencies are cached up to ``max_seq_len`` and extended on the fly if
    a longer sequence is encountered.
    """

    def __init__(
        self,
        head_dim: int,
        max_seq_len: int = 2048,
        theta: float = 10_000.0,
        device: Optional[torch.device] = None,
    ) -> None:
        super().__init__()
        self.head_dim    = head_dim
        self.max_seq_len = max_seq_len
        self.theta       = theta

        # Precompute inverse frequencies (half of head_dim)
        inv_freq = 1.0 / (
            theta ** (torch.arange(0, head_dim, 2, dtype=torch.float32, device=device)
                      / head_dim)
        )
        self.register_buffer("inv_freq", inv_freq, persistent=False)
        self._build_cache(max_seq_len, device=device)

    def _build_cache(self, seq_len: int, device: Optional[torch.device] = None) -> None:
        """Build (or extend) the cos/sin cache."""
        t       = torch.arange(seq_len, dtype=torch.float32,
                               device=self.inv_freq.device if device is None else device)
        freqs   = torch.outer(t, self.inv_freq)          # (seq_len, head_dim/2)
        emb     = torch.cat([freqs, freqs], dim=-1)       # (seq_len, head_dim)
        self.register_buffer("cos_cached", emb.cos()[None, None, :, :], persistent=False)
        self.register_buffer("sin_cached", emb.sin()[None, None, :, :], persistent=False)
        self.max_seq_len = seq_len

    @staticmethod
    def _rotate_half(x: torch.Tensor) -> torch.Tensor:
        """Rotate the second half of the last dimension by -90°."""
        half = x.shape[-1] // 2
        x1, x2 = x[..., :half], x[..., half:]
        return torch.cat([-x2, x1], dim=-1)

    def forward(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        seq_len: int,
        position_ids: Optional[torch.LongTensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Apply RoPE to q and k.

        q, k: (batch, n_heads, seq_len, head_dim)
        position_ids: (batch, seq_len) or None
        """
        if seq_len > self.max_seq_len:
            self._build_cache(seq_len, device=q.device)

        if position_ids is not None:
            # Gather cos/sin for the specific positions in this batch.
            # cos_cached: (1, 1, max_seq, head_dim) → flatten to (max_seq, head_dim)
            # then index with position_ids (B, L) → (B, L, head_dim)
            # and unsqueeze head axis → (B, 1, L, head_dim)
            cos_2d = self.cos_cached.squeeze(0).squeeze(0).to(q.dtype)  # (max_seq, head_dim)
            sin_2d = self.sin_cached.squeeze(0).squeeze(0).to(q.dtype)
            cos = cos_2d[position_ids].unsqueeze(1)   # (B, 1, L, head_dim)
            sin = sin_2d[position_ids].unsqueeze(1)
        else:
            cos = self.cos_cached[:, :, :seq_len, :].to(q.dtype)   # (1, 1, L, head_dim)
            sin = self.sin_cached[:, :, :seq_len, :].to(q.dtype)

        q_rot = q * cos + self._rotate_half(q) * sin
        k_rot = k * cos + self._rotate_half(k) * sin
        return q_rot, k_rot


# ---------------------------------------------------------------------------
# 3. Slip-Anchors
# ---------------------------------------------------------------------------

class SlipAnchors(nn.Module):
    """
    Slip-anchor module — biases K and V using a learnable codebook.

    Applied *after* RoPE, so FlashAttention compatibility is preserved.

    Parameters
    ----------
    d_model   : residual hidden dimension (2048)
    n_anchors : codebook size (64)
    d_anchor  : anchor bottleneck dim (128)
    head_dim  : per-head dimension (128)
    n_heads   : number of attention heads (16)
    """

    def __init__(
        self,
        d_model: int,
        n_anchors: int,
        d_anchor: int,
        head_dim: int,
        n_heads: int,
    ) -> None:
        super().__init__()
        self.n_anchors = n_anchors
        self.d_anchor  = d_anchor
        self.n_heads   = n_heads
        self.head_dim  = head_dim

        # Learnable codebook: (n_anchors, d_anchor)
        self.anchors       = nn.Parameter(torch.empty(n_anchors, d_anchor))
        # h → anchor space
        self.W_anchor_down = nn.Linear(d_model, d_anchor, bias=False)
        # anchor context → K bias (per head)
        self.U_k = nn.Linear(d_anchor, head_dim, bias=False)
        # anchor context → V bias (per head)
        self.U_v = nn.Linear(d_anchor, head_dim, bias=False)

        self._init_weights()

    def _init_weights(self) -> None:
        nn.init.normal_(self.anchors, std=0.02)
        nn.init.normal_(self.W_anchor_down.weight, std=0.02)
        nn.init.normal_(self.U_k.weight, std=0.02)
        nn.init.normal_(self.U_v.weight, std=0.02)

    def forward(
        self,
        h: torch.Tensor,                  # (B, L, d_model) — pre-attention hidden state
        K: torch.Tensor,                  # (B, n_heads, L, head_dim) — post-RoPE
        V: torch.Tensor,                  # (B, n_heads, L, head_dim)
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Return K_modified, V_modified."""
        B, L, _ = h.shape

        # 1. Project h to anchor space: (B, L, d_anchor)
        h_anchor = self.W_anchor_down(h)

        # 2. Soft scores over codebook: (B, L, n_anchors)
        scores = torch.softmax(h_anchor @ self.anchors.T, dim=-1)

        # 3. Weighted anchor context: (B, L, d_anchor)
        anchor_context = scores @ self.anchors

        # 4. Project to K and V bias spaces: (B, L, head_dim)
        k_bias = self.U_k(anchor_context)   # (B, L, head_dim)
        v_bias = self.U_v(anchor_context)   # (B, L, head_dim)

        # 5. Broadcast across heads: unsqueeze head dim → (B, 1, L, head_dim)
        K_modified = K + k_bias.unsqueeze(1)
        V_modified = V + v_bias.unsqueeze(1)

        return K_modified, V_modified


# ---------------------------------------------------------------------------
# 4. Attention
# ---------------------------------------------------------------------------

class SABERAttention(nn.Module):
    """
    Multi-head attention with:
    * No projection biases.
    * RoPE applied to Q and K after head reshape.
    * Slip-anchor modulation of K and V after RoPE.
    * F.scaled_dot_product_attention (FlashAttention 2 compatible).
    """

    def __init__(self, config: SABERConfig, layer_idx: int) -> None:
        super().__init__()
        self.config    = config
        self.layer_idx = layer_idx
        self.d_model   = config.d_model
        self.n_heads   = config.n_heads
        self.head_dim  = config.head_dim

        # QKV and O projections — no bias throughout
        self.q_proj = nn.Linear(self.d_model, self.d_model, bias=False)
        self.k_proj = nn.Linear(self.d_model, self.d_model, bias=False)
        self.v_proj = nn.Linear(self.d_model, self.d_model, bias=False)
        self.o_proj = nn.Linear(self.d_model, self.d_model, bias=False)

        # Rotary embeddings (shared via the parent model, but instantiated here
        # for standalone correctness)
        self.rotary_emb = SABERRotaryEmbedding(
            head_dim=self.head_dim,
            max_seq_len=config.max_position_embeddings,
            theta=config.rope_theta,
        )

        # Slip-anchors
        self.slip_anchors = SlipAnchors(
            d_model=self.d_model,
            n_anchors=config.n_anchors,
            d_anchor=config.d_anchor,
            head_dim=self.head_dim,
            n_heads=self.n_heads,
        )

    def forward(
        self,
        hidden_states: torch.Tensor,          # (B, L, d_model)
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_value: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
        use_cache: bool = False,
        output_attentions: bool = False,
    ) -> Tuple[torch.Tensor, ...]:

        B, L, _ = hidden_states.shape

        # ---- QKV projections ----
        Q = self.q_proj(hidden_states)   # (B, L, d_model)
        K = self.k_proj(hidden_states)
        V = self.v_proj(hidden_states)

        # ---- Reshape to (B, n_heads, L, head_dim) ----
        def _reshape(t: torch.Tensor) -> torch.Tensor:
            return t.view(B, L, self.n_heads, self.head_dim).transpose(1, 2)

        Q, K, V = _reshape(Q), _reshape(K), _reshape(V)

        # ---- Apply RoPE to Q and K ----
        kv_seq_len = L
        if past_key_value is not None:
            kv_seq_len += past_key_value[0].shape[-2]

        Q, K = self.rotary_emb(Q, K, seq_len=kv_seq_len, position_ids=position_ids)

        # ---- KV cache ----
        if past_key_value is not None:
            K = torch.cat([past_key_value[0], K], dim=2)
            V = torch.cat([past_key_value[1], V], dim=2)

        present_kv = (K, V) if use_cache else None

        # ---- Slip-anchor modulation of K and V ----
        # Pass raw h (pre-attn hidden state) to avoid circularity
        if getattr(self.config, 'enable_anchors', True):
            K, V = self.slip_anchors(hidden_states, K, V)

        # ---- Scaled dot-product attention (FlashAttention 2 compatible) ----
        # Build causal mask if needed (SDPA handles is_causal natively)
        is_causal = attention_mask is None and L > 1
        attn_out = F.scaled_dot_product_attention(
            Q, K, V,
            attn_mask=attention_mask,
            dropout_p=0.0,
            is_causal=is_causal,
        )   # (B, n_heads, L, head_dim)

        # ---- Merge heads and project ----
        attn_out = attn_out.transpose(1, 2).contiguous().view(B, L, self.d_model)
        attn_out = self.o_proj(attn_out)

        outputs: Tuple = (attn_out,)
        if use_cache:
            outputs += (present_kv,)
        if output_attentions:
            # Attention weights are not explicitly computed when using SDPA
            outputs += (None,)

        return outputs


# ---------------------------------------------------------------------------
# 5. Experience Stream
# ---------------------------------------------------------------------------

class ExperienceStream(nn.Module):
    """
    Per-layer experience update with a curiosity (prediction-error) auxiliary loss.

    State flows layer-to-layer within a single forward pass; it is reset to
    zeros at the start of each new sequence.

    Parameters
    ----------
    d_model : residual hidden dimension
    d_exp   : experience state dimension (256)
    """

    def __init__(self, d_model: int, d_exp: int) -> None:
        super().__init__()
        # Summarise post-attention hidden state → experience space
        self.W_s    = nn.Linear(d_model, d_exp, bias=False)
        # Predict current summary from previous state (curiosity signal)
        self.W_pred = nn.Linear(d_exp,   d_exp, bias=False)
        # Gated update to experience state
        self.W_e    = nn.Linear(d_exp,   d_exp, bias=False)
        # Learned decay gate: sigmoid(3.0) ~ 0.95 retains most state initially
        self.decay_raw = nn.Parameter(torch.full((d_exp,), 3.0))
        # Layer-norm on experience state to prevent magnitude drift
        self.exp_norm = nn.LayerNorm(d_exp)

    def forward(
        self,
        h: torch.Tensor,                  # (B, L, d_model) post-attention
        experience_state: torch.Tensor,   # (B, L, d_exp)   previous state
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Returns
        -------
        new_experience_state : (B, L, d_exp)
        curiosity_loss       : scalar tensor
        """
        # 1. Summarise current hidden state
        s = self.W_s(h)                                # (B, L, d_exp)

        # 2. Stop-gradient on s for the curiosity term (CRITICAL for stability)
        s_sg = s.detach()

        # 3. Predict current summary from previous experience state
        s_pred = self.W_pred(experience_state)         # (B, L, d_exp)

        # 4. Curiosity = mean squared prediction error
        curiosity_loss = (s_sg - s_pred).pow(2).mean()

        # 5. Update experience state with SiLU-gated delta
        decay = torch.sigmoid(self.decay_raw)          # (d_exp,) in [0, 1]
        delta = F.silu(self.W_e(s))                    # (B, L, d_exp)
        new_state = decay * experience_state + delta
        new_state = self.exp_norm(new_state)

        return new_state, curiosity_loss


# ---------------------------------------------------------------------------
# 6. Feed-forward networks
# ---------------------------------------------------------------------------

class StandardFFN(nn.Module):
    """Standard SwiGLU FFN (used on odd-indexed layers)."""

    def __init__(self, d_model: int, d_ff: int) -> None:
        super().__init__()
        self.W1 = nn.Linear(d_model, d_ff, bias=False)   # gate projection
        self.W3 = nn.Linear(d_model, d_ff, bias=False)   # up projection
        self.W2 = nn.Linear(d_ff,   d_model, bias=False) # down projection

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # SwiGLU: silu(gate) ⊙ up, then project down
        return self.W2(F.silu(self.W1(x)) * self.W3(x))


class ResonantFFN(nn.Module):
    """
    Resonant FFN (used on even-indexed layers).

    Augments standard SwiGLU with a learned sinusoidal modulation.
    The blend is controlled by a per-layer scalar alpha (init ≈ 0.95).

        ffn_out  = W2(silu(W1(x)) * W3(x))          # standard SwiGLU
        mod      = sin(W_freq @ x)                   # sinusoidal modulation
        alpha    = sigmoid(alpha_raw)                # ≈ 0.95 at init
        output   = alpha * ffn_out + (1-alpha) * ffn_out * (1 + mod)
                 = ffn_out * (alpha + (1-alpha) * (1 + mod))
    """

    def __init__(self, d_model: int, d_ff: int, alpha_init: float = 3.0) -> None:
        super().__init__()
        # Shared SwiGLU matrices
        self.W1 = nn.Linear(d_model, d_ff,    bias=False)
        self.W3 = nn.Linear(d_model, d_ff,    bias=False)
        self.W2 = nn.Linear(d_ff,    d_model, bias=False)
        # Sinusoidal modulation projection
        self.W_freq = nn.Linear(d_model, d_model, bias=False)
        # Per-layer blending scalar; init so sigmoid(alpha_raw) ≈ 0.95
        self.alpha_raw = nn.Parameter(torch.tensor(alpha_init))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Standard SwiGLU output
        ffn_out = self.W2(F.silu(self.W1(x)) * self.W3(x))   # (B, L, d_model)

        # Sinusoidal modulation
        mod = torch.sin(self.W_freq(x))                        # (B, L, d_model)

        # Learned blend
        alpha = torch.sigmoid(self.alpha_raw)                  # scalar ∈ (0,1)
        output = alpha * ffn_out + (1.0 - alpha) * (ffn_out * (1.0 + mod))
        return output


# ---------------------------------------------------------------------------
# 7. Transformer Block
# ---------------------------------------------------------------------------

class SABERBlock(nn.Module):
    """
    Single SABER transformer block.

    Structure (pre-norm):
        h = h + Attention(RMSNorm(h))
        h = h + FFN(RMSNorm(h))
        experience_state, curiosity = ExperienceStream(h, experience_state)
    """

    def __init__(self, config: SABERConfig, layer_idx: int) -> None:
        super().__init__()
        self.config    = config
        self.layer_idx = layer_idx

        self.input_layernorm      = SABERRMSNorm(config.d_model, eps=config.rms_norm_eps)
        self.post_attention_layernorm = SABERRMSNorm(config.d_model, eps=config.rms_norm_eps)

        self.self_attn = SABERAttention(config, layer_idx=layer_idx)

        # Select FFN type based on layer index
        if layer_idx in config.resonant_layers:
            self.ffn: nn.Module = ResonantFFN(
                d_model=config.d_model,
                d_ff=config.d_ff,
                alpha_init=config.resonant_alpha_init,
            )
        else:
            self.ffn = StandardFFN(d_model=config.d_model, d_ff=config.d_ff)

        self.experience_stream = ExperienceStream(
            d_model=config.d_model,
            d_exp=config.d_exp,
        )

    def forward(
        self,
        hidden_states: torch.Tensor,                   # (B, L, d_model)
        experience_state: torch.Tensor,                # (B, L, d_exp)
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_value: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
        use_cache: bool = False,
        output_attentions: bool = False,
    ) -> Tuple:
        residual = hidden_states

        # ---- Pre-norm attention ----
        normed = self.input_layernorm(hidden_states)
        attn_outputs = self.self_attn(
            normed,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_value=past_key_value,
            use_cache=use_cache,
            output_attentions=output_attentions,
        )
        attn_out = attn_outputs[0]
        hidden_states = residual + attn_out                # residual connection

        # ---- Pre-norm FFN ----
        residual = hidden_states
        hidden_states = residual + self.ffn(self.post_attention_layernorm(hidden_states))

        # ---- Experience stream update ----
        if getattr(self.config, 'enable_experience', True):
            experience_state, curiosity_loss = self.experience_stream(
                hidden_states, experience_state
            )
        else:
            curiosity_loss = torch.tensor(0.0, device=hidden_states.device)

        # Pack remaining outputs
        extra = attn_outputs[1:]   # present_kv and/or attention_weights
        return (hidden_states, experience_state, curiosity_loss) + extra


# ---------------------------------------------------------------------------
# 8. Base Model
# ---------------------------------------------------------------------------

class SABERModel(PreTrainedModel):
    """
    SABER base model: token embeddings → blocks → final RMSNorm.

    Does not include the LM head — use ``SABERForCausalLM`` for training.
    """

    config_class = SABERConfig
    base_model_prefix = "model"
    supports_gradient_checkpointing = True
    _no_split_modules = ["SABERBlock"]
    _supports_flash_attn_2 = True

    def __init__(self, config: SABERConfig) -> None:
        super().__init__(config)
        self.embed_tokens = nn.Embedding(config.vocab_size, config.d_model)
        self.layers       = nn.ModuleList(
            [SABERBlock(config, layer_idx=i) for i in range(config.n_layers)]
        )
        self.norm = SABERRMSNorm(config.d_model, eps=config.rms_norm_eps)

        self.gradient_checkpointing = False
        self.post_init()   # weight init + gradient-checkpointing setup

    # ------------------------------------------------------------------ #
    # Weight initialization (called by post_init via _init_weights)
    # ------------------------------------------------------------------ #

    def _init_weights(self, module: nn.Module) -> None:
        std = self.config.initializer_range
        if isinstance(module, nn.Linear):
            nn.init.normal_(module.weight, mean=0.0, std=std)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, mean=0.0, std=std)
        elif isinstance(module, SABERRMSNorm):
            nn.init.ones_(module.weight)
        elif isinstance(module, SlipAnchors):
            # Handled inside SlipAnchors._init_weights; no-op here
            pass
        # ResonantFFN.alpha_raw: initialised inside the class (default=3.0)

    # ------------------------------------------------------------------ #
    # Accessors
    # ------------------------------------------------------------------ #

    def get_input_embeddings(self) -> nn.Embedding:
        return self.embed_tokens

    def set_input_embeddings(self, value: nn.Embedding) -> None:
        self.embed_tokens = value

    # ------------------------------------------------------------------ #
    # Forward
    # ------------------------------------------------------------------ #

    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[List[Tuple[torch.Tensor, torch.Tensor]]] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ) -> Union[BaseModelOutputWithPast, Tuple]:

        use_cache           = use_cache if use_cache is not None else self.config.use_cache
        output_attentions   = output_attentions or False
        output_hidden_states = output_hidden_states or False
        return_dict         = return_dict if return_dict is not None else self.config.use_return_dict

        # ---- Embeddings ----
        if inputs_embeds is None:
            if input_ids is None:
                raise ValueError("Provide either input_ids or inputs_embeds.")
            inputs_embeds = self.embed_tokens(input_ids)

        B, L, _ = inputs_embeds.shape

        # ---- Position ids ----
        if position_ids is None:
            past_len    = past_key_values[0][0].shape[-2] if past_key_values else 0
            position_ids = torch.arange(
                past_len, past_len + L,
                dtype=torch.long,
                device=inputs_embeds.device,
            ).unsqueeze(0).expand(B, -1)

        # ---- Attention mask conversion for SDPA ----
        # We rely on SDPA's built-in is_causal flag; user-supplied masks are
        # passed as-is (e.g., padding masks in float format).
        # If a 2-D (B, L) boolean mask is supplied, convert to additive float.
        causal_mask: Optional[torch.Tensor] = None
        if attention_mask is not None and attention_mask.dim() == 2:
            # 0 → masked (−∞), 1 → attended (0)
            # Expand to (B, 1, 1, L) for SDPA broadcasting
            causal_mask = (
                (1.0 - attention_mask[:, None, None, :].float())
                * torch.finfo(inputs_embeds.dtype).min
            )

        # ---- Initialise experience state ----
        # Shape: (B, L, d_exp) — zeros at the start of each sequence.
        # Note: when using KV cache the sequence length L changes per step;
        # experience state is kept external to the model for incremental
        # decoding (callers may pass zeros each step for generation).
        experience_state = torch.zeros(
            B, L, self.config.d_exp,
            dtype=inputs_embeds.dtype,
            device=inputs_embeds.device,
        )

        # ---- Layer loop ----
        hidden_states      = inputs_embeds
        all_hidden_states  = () if output_hidden_states else None
        all_self_attns     = () if output_attentions   else None
        next_cache         = []
        total_curiosity    = torch.tensor(0.0, device=inputs_embeds.device,
                                          dtype=inputs_embeds.dtype)

        for i, layer in enumerate(self.layers):
            if output_hidden_states:
                all_hidden_states += (hidden_states,)

            past_kv = past_key_values[i] if past_key_values is not None else None

            if self.gradient_checkpointing and self.training:
                # Wrap block forward through torch.utils.checkpoint.
                # Curiosity loss gradient flows normally; only activations
                # are recomputed.
                def _make_ckpt_fn(layer, experience_state):
                    def _fn(hidden_states, causal_mask, position_ids):
                        return layer(
                            hidden_states,
                            experience_state=experience_state,
                            attention_mask=causal_mask,
                            position_ids=position_ids,
                            past_key_value=None,
                            use_cache=False,
                            output_attentions=output_attentions,
                        )
                    return _fn

                layer_outputs = torch.utils.checkpoint.checkpoint(
                    _make_ckpt_fn(layer, experience_state),
                    hidden_states,
                    causal_mask,
                    position_ids,
                    use_reentrant=False,
                )
            else:
                layer_outputs = layer(
                    hidden_states,
                    experience_state=experience_state,
                    attention_mask=causal_mask,
                    position_ids=position_ids,
                    past_key_value=past_kv,
                    use_cache=use_cache,
                    output_attentions=output_attentions,
                )

            hidden_states    = layer_outputs[0]
            experience_state = layer_outputs[1]
            total_curiosity  = total_curiosity + layer_outputs[2]

            # Collect KV cache
            if use_cache:
                # present_kv is at index 3 (after hidden, exp_state, curiosity)
                next_cache.append(layer_outputs[3] if len(layer_outputs) > 3 else None)

            if output_attentions:
                # attn weights at last position when output_attentions=True
                all_self_attns += (layer_outputs[-1],)

        hidden_states = self.norm(hidden_states)

        if output_hidden_states:
            all_hidden_states += (hidden_states,)

        # Average curiosity loss across layers
        mean_curiosity = total_curiosity / self.config.n_layers

        next_cache_out = next_cache if use_cache else None

        if not return_dict:
            # Always emit a fixed-position tuple so SABERForCausalLM can
            # index reliably:
            #   [0] hidden_states
            #   [1] mean_curiosity
            #   [2] past_key_values (None when use_cache=False)
            #   [3] all_hidden_states (None when output_hidden_states=False)
            #   [4] all_self_attns   (None when output_attentions=False)
            return (
                hidden_states,
                mean_curiosity,
                next_cache_out,
                all_hidden_states,
                all_self_attns,
            )

        return BaseModelOutputWithPast(
            last_hidden_state=hidden_states,
            past_key_values=next_cache_out,
            hidden_states=all_hidden_states,
            attentions=all_self_attns,
        ), mean_curiosity


# ---------------------------------------------------------------------------
# 9. Causal LM wrapper
# ---------------------------------------------------------------------------

class SABERForCausalLM(PreTrainedModel, GenerationMixin):
    """
    Eve-3-SABER-1B for causal language modelling.

    Compatible with HuggingFace ``Trainer``, ``SFTTrainer``, PEFT, and
    standard ``generate()`` pipelines.

    Loss = L_CE + curiosity_coeff * L_curiosity
    """

    config_class = SABERConfig
    base_model_prefix = "model"
    supports_gradient_checkpointing = True
    _no_split_modules = ["SABERBlock"]
    _supports_flash_attn_2 = True
    # Map required for AutoModel/AutoModelForCausalLM
    # Dict mapping parameter to its tied source (HF 5.x format)
    _tied_weights_keys = {"lm_head.weight": "model.embed_tokens.weight"}

    def __init__(self, config: SABERConfig) -> None:
        super().__init__(config)
        self.model   = SABERModel(config)
        # LM head — tied to token embeddings (no extra params)
        self.lm_head = nn.Linear(config.d_model, config.vocab_size, bias=False)
        self.post_init()

    # ------------------------------------------------------------------ #
    # Weight tying (called by post_init)
    # ------------------------------------------------------------------ #

    def get_input_embeddings(self) -> nn.Embedding:
        return self.model.embed_tokens

    def set_input_embeddings(self, value: nn.Embedding) -> None:
        self.model.embed_tokens = value

    def get_output_embeddings(self) -> nn.Linear:
        return self.lm_head

    def set_output_embeddings(self, new_embeddings: nn.Linear) -> None:
        self.lm_head = new_embeddings

    def tie_weights(self, **kwargs) -> None:
        """Tie lm_head.weight ← embed_tokens.weight."""
        self.lm_head.weight = self.model.embed_tokens.weight

    # ------------------------------------------------------------------ #
    # Forward
    # ------------------------------------------------------------------ #

    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[List[Tuple[torch.Tensor, torch.Tensor]]] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ) -> Union[CausalLMOutputWithPast, Tuple]:

        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        # ---- Base model (always use tuple return for clean unpacking) ----
        # SABERModel always returns (hidden_states, curiosity, [pkv], [all_hs], [all_attn])
        # when return_dict=False.  We unpack manually and re-wrap for return_dict=True.
        base_out = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=False,   # always False so we get a plain tuple
        )
        # base_out: (hidden_states, curiosity_loss, [pkv], [all_hs], [all_attn])
        hidden_states  = base_out[0]                      # (B, L, d_model)
        curiosity_loss = base_out[1]                      # scalar
        pkv            = base_out[2] if len(base_out) > 2 else None
        all_hs         = base_out[3] if len(base_out) > 3 else None
        all_attn       = base_out[4] if len(base_out) > 4 else None

        # ---- LM logits ----
        logits = self.lm_head(hidden_states)   # (B, L, vocab_size)

        # ---- Loss computation ----
        loss: Optional[torch.Tensor] = None
        if labels is not None:
            # Causal LM: predict token t+1 from position t.
            # Shift logits left by one, labels right by one.
            shift_logits = logits[:, :-1, :].contiguous()   # (B, L-1, V)
            shift_labels = labels[:, 1:].contiguous()        # (B, L-1)

            loss_fct = nn.CrossEntropyLoss(ignore_index=-100)
            ce_loss  = loss_fct(
                shift_logits.view(-1, self.config.vocab_size),
                shift_labels.view(-1),
            )
            loss = ce_loss + self.config.curiosity_coeff * curiosity_loss

        if not return_dict:
            out = (logits,)
            if loss is not None:
                out = (loss,) + out
            if pkv is not None:
                out += (pkv,)
            return out

        output = CausalLMOutputWithPast(
            loss=loss,
            logits=logits,
            past_key_values=pkv,
            hidden_states=all_hs,
            attentions=all_attn,
        )
        # Expose component losses for training loop logging
        if labels is not None:
            output["ce_loss"] = ce_loss
            output["curiosity_loss"] = curiosity_loss
        return output

    # ------------------------------------------------------------------ #
    # Generation helpers
    # ------------------------------------------------------------------ #

    def prepare_inputs_for_generation(
        self,
        input_ids: torch.LongTensor,
        past_key_values: Optional[List] = None,
        attention_mask: Optional[torch.Tensor] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        **kwargs,
    ) -> dict:
        if past_key_values is not None:
            # Only pass the last token during incremental decoding
            input_ids = input_ids[:, -1:]

        # Build position_ids from the current seq length
        position_ids = kwargs.get("position_ids", None)
        if attention_mask is not None and position_ids is None:
            position_ids = attention_mask.long().cumsum(-1) - 1
            position_ids.masked_fill_(attention_mask == 0, 1)
            if past_key_values is not None:
                position_ids = position_ids[:, -1:]

        model_inputs: dict = {}
        if inputs_embeds is not None and past_key_values is None:
            model_inputs["inputs_embeds"] = inputs_embeds
        else:
            model_inputs["input_ids"] = input_ids

        model_inputs.update(
            {
                "position_ids":    position_ids,
                "past_key_values": past_key_values,
                "use_cache":       kwargs.get("use_cache", True),
                "attention_mask":  attention_mask,
            }
        )
        return model_inputs

    @staticmethod
    def _reorder_cache(
        past_key_values: List[Tuple[torch.Tensor, torch.Tensor]],
        beam_idx: torch.LongTensor,
    ) -> List[Tuple[torch.Tensor, torch.Tensor]]:
        """Re-order KV cache for beam search."""
        return [
            (
                past_kv[0].index_select(0, beam_idx.to(past_kv[0].device)),
                past_kv[1].index_select(0, beam_idx.to(past_kv[1].device)),
            )
            for past_kv in past_key_values
        ]


# ---------------------------------------------------------------------------
# Auto-class registration hint (used by HF hub auto-loading)
# ---------------------------------------------------------------------------

SABERConfig.register_for_auto_class("AutoConfig")
SABERForCausalLM.register_for_auto_class("AutoModelForCausalLM")
