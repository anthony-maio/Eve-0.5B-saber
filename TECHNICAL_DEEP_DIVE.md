# Eve-3-0.5B-SABER: Three Novel Mechanisms for Making Small Models Weird in Useful Ways

**Anthony Maio — Making Minds AI**
*March 2026*

> **SABER** — **S**emantic **A**nchor-**B**iased **E**xperience-**R**esonant. The third model in the Eve series, following Eve-2-MoE-272M. A 500M dense transformer with three novel mechanisms that test specific hypotheses about attention biasing, cross-layer memory, and FFN modulation.

---

## Opening: Why Build a 500M Research Model?

There are a lot of sub-1B parameter language models out there right now. Most of them are essentially LLaMA-small: dense decoder-only transformers with SwiGLU, RoPE, RMSNorm, no biases, and nothing architecturally surprising. They're good. They're well-understood. They're boring.

I don't mean that as an insult. Boring is powerful — boring means the optimizer knows what to do, FlashAttention works out of the box, and you can reason about training dynamics because a thousand people have already walked that path. But I kept wondering: what if you took that exact envelope — the same parameter budget, the same hardware constraints, the same training recipes — and made it *architecturally novel* instead of just scaling a known design down?

Not novel for novelty's sake. Novel in ways that test specific hypotheses about what transformers could be doing internally if we gave them slightly different machinery.

That's Eve-3-SABER. It started at 1B parameters, but budget constraints pushed it down to 500M — which turned out to be a blessing. At 500M, the model trains faster, ablations run cheaper, and the architectural hypotheses get stress-tested harder. If novel components help at 500M, they'll likely help more at scale, not less. And we're over-training at 100× Chinchilla-optimal (50B tokens for 500M params), which means the architecture has to actually *earn* its keep — there's nowhere to hide behind insufficient data.

Eve-3-SABER introduces three novel architectural components on top of a standard transformer skeleton:

1. **Slip-Anchors** — learned attention biases injected through K/V modification, preserving FlashAttention compatibility
2. **Experience Stream** — a per-token, cross-layer state vector with decay gating, LayerNorm stabilization, and a curiosity-driven auxiliary loss
3. **Resonant FFN** — sinusoidal modulation of feed-forward outputs with self-regulating learned blending

None of these require custom CUDA kernels. All of them are designed so that if they don't help, the model can learn to ignore them. And all of them connect to ideas I've been developing in the broader Eve project and slipstream work — semantic anchors, biomimetic curiosity, manifold resonance.

---

## 1. The Standard Skeleton

Before we get to the weird stuff, let's establish what's conventional. Eve-3-0.5B-SABER is built on a foundation that would look familiar to anyone who's read the LLaMA or Mistral papers:

| Parameter | Value |
|-----------|-------|
| `d_model` | 1536 |
| `n_layers` | 20 |
| `n_heads` | 12 |
| `head_dim` | 128 |
| `d_ff` | 2164 |
| `vocab_size` | 50257 |
| `max_seq_len` | 2048 |
| **Total params** | **499,976,458** |

The FFN uses SwiGLU with three matrices (W1, W3, W2). The `d_ff` was tuned via a param counter script to 2164, which — once you account for the novel components consuming ~34.5M params — lands the total at ~500M parameters. Positional information comes from RoPE with θ=10000. Normalization is RMSNorm in the pre-norm position. No biases anywhere in the main projections. The LM head shares weights with the token embedding layer.

Why these choices? Because they're *proven* at this scale. I didn't want to confound novel mechanisms with non-standard baselines. If I'm testing three new ideas, the last thing I need is to also wonder whether my normalization scheme is causing problems. The skeleton is intentionally vanilla so the novel components can be evaluated cleanly.

The standard components account for ~93% of the parameter budget. The three novel components together add only ~34.5M parameters (~6.9% of the total). They're lightweight modifications, not heavy new machinery.

### Scale Decision: 1B → 0.5B

The original design was 1B parameters (d_model=2048, n_layers=24, d_ff=2855, ~73M novel params). Budget constraints forced a downsizing. Rather than compromising on training tokens, I chose to halve the model and keep the 50B token target — resulting in roughly 100× Chinchilla-optimal training. This is actually a more rigorous test of the architecture: if the novel components provide useful inductive biases, over-training will make that signal clearer, not weaker.

The proportional structure was preserved: novel components remain ~7% of total parameters, the head dimension stays at 128 for hardware alignment, and all three mechanisms operate at every layer (or every other layer for Resonant FFN) just as in the 1B design.

---

## 2. Slip-Anchors: Biasing Attention Without Breaking Flash

### The Problem

Here's a question that bugged me for a long time: how do you inject *learned structural biases* into attention without giving up FlashAttention?

Relative position biases (ALiBi, etc.) solve one version of this — they tell the model "things far away matter less" or "things at position offsets of X have a certain relationship." But what if you want *semantic* biases? What if you want the model to learn that when the current token is "doing math," attention should shift in a particular way — not based on *position*, but based on *content type*?

The naive approach is to add a learned bias matrix inside the attention computation: `attn = softmax(QK^T / √d + bias)`. But FlashAttention doesn't support arbitrary attention biases efficiently. You'd need custom CUDA kernels, which means you lose the massive speedup from `F.scaled_dot_product_attention` and tie yourself to specific hardware.

### The Solution: Modify K and V, Not Attention

The insight behind slip-anchors is simple: if you can't add bias *inside* SDPA, bias the *inputs* to SDPA instead. Specifically, modify K and V *before* they enter the attention computation. From FlashAttention's perspective, it just sees slightly different K and V tensors — it doesn't know or care how they got that way.

This is the core architectural trick, and it's why the name includes "slip" — the biases *slip into* attention through the keys and values rather than through the attention matrix directly.

### The Mechanism in Detail

Each attention layer maintains a learnable **codebook** of 64 anchors, each living in a d_anchor=96 dimensional space. Think of these as prototypical "attention situations" — learned archetypes that the model can blend between.

Here's how it works step by step:

```python
# 1. Project hidden state into anchor space
h_anchor = h @ W_anchor_down          # (B, L, 1536) -> (B, L, 96)

# 2. Score against the codebook
scores = softmax(h_anchor @ anchors.T)  # (B, L, 96) @ (96, 64) -> (B, L, 64)

# 3. Compute weighted anchor context
anchor_ctx = scores @ anchors           # (B, L, 64) @ (64, 96) -> (B, L, 96)

# 4. Project to K and V bias
k_bias = anchor_ctx @ U_k              # (B, L, 96) -> (B, L, 128)
v_bias = anchor_ctx @ U_v              # (B, L, 96) -> (B, L, 128)

# 5. Apply RoPE to Q and K first, THEN add anchor bias to K
Q, K = apply_rope(Q, K)
K_modified = K + k_bias.unsqueeze(head_axis)
V_modified = V + v_bias.unsqueeze(head_axis)

# 6. Standard SDPA — FlashAttention sees nothing unusual
output = F.scaled_dot_product_attention(Q, K_modified, V_modified)
```

The anchors act as a soft lookup table. The model learns to score each token's hidden state against 64 archetypes, then uses the winning anchors to compute position-independent biases on K and V. This is content-dependent attention modulation that plays nicely with every existing attention optimization.

### The Circularity Trade-off

One of the most interesting design decisions was *what input* to score against the anchors.

The obvious candidate is Q — it carries the most attention-relevant information, and scoring from Q would let the anchors modulate attention in a more targeted way. But scoring from Q creates a circularity: the attention query would influence its own key biases, which could lead to degenerate fixed points during training where the anchor bias reinforces whatever Q already encodes.

The alternative is scoring from the raw hidden state `h` — the representation before it's been projected into Q/K/V spaces. This breaks the circularity cleanly. It's also computationally cheaper — you avoid projecting through W_Q first.

I went with `h`. The circularity argument is compelling, and if the Q-based approach turns out to be better, it's a clean ablation to test.

### RoPE Ordering

A critical design consideration is *when* to apply anchor biases relative to RoPE. If you add the bias to K *before* RoPE, the rotary encoding will corrupt the bias vector — it'll get position-dependent modulation that wasn't intended. The correct order is:

1. Apply RoPE to Q and K
2. *Then* add anchor bias to K

This way, the anchor bias operates in post-RoPE space. The bias is purely content-dependent, and positional encoding remains clean.

### Parameter Cost

The slip-anchor machinery is remarkably cheap at 0.5B scale:

- `W_anchor_down`: 1536 × 96 = 147K
- `anchors`: 64 × 96 = 6K
- `U_k`: 96 × 128 = 12K
- `U_v`: 96 × 128 = 12K
- **Per layer: ~178K**
- **Total across 20 layers: ~3.6M** (less than 1% of total parameters)

That's the beauty of the bottleneck design. The d_anchor=96 space is small enough that the entire mechanism is almost free in terms of parameters, and the computation is dominated by the `h @ W_anchor_down` matmul, which is trivially fast relative to the attention operation it modifies.

---

## 3. Experience Stream: Curiosity Without Recurrence

### The Motivation

Here's something that's always struck me as odd about transformers: each layer processes the residual stream, but layers have no *explicit* channel for communicating metadata about what they've done. Yes, the residual stream carries forward all the information. But a layer's contribution is mixed into the same high-dimensional space as everything else. There's no side-channel where layer 7 can tell layer 8 "I just found something surprising."

What if there was?

The experience stream is a low-dimensional per-token vector (d_exp=192) that flows layer-to-layer. It's a dedicated channel for cross-layer communication that exists *outside* the residual stream. Each layer reads the current experience state, updates it, and optionally uses it.

### How It Works

```
┌──────────────────────────────────────────────────────┐
│                     Layer i                           │
│                                                      │
│  Input: h (residual stream), exp_state (d_exp=192)   │
│                                                      │
│  1. h_post_attn = attention(h)                       │
│                                                      │
│  2. s = W_s @ h_post_attn      (d_model -> d_exp)   │
│     [summary of what this layer "saw"]               │
│                                                      │
│  3. s_sg = s.detach()           [stop gradient!]     │
│                                                      │
│  4. s_pred = W_pred @ exp_state  (d_exp -> d_exp)    │
│     [predict what we'd see based on prior layers]    │
│                                                      │
│  5. curiosity = ||s_sg - s_pred||²                   │
│     [prediction error = surprise signal]             │
│                                                      │
│  6. decay = sigmoid(decay_raw)   [element-wise gate] │
│     delta = silu(W_e @ s)                            │
│     exp_state = decay * exp_state + delta            │
│     exp_state = LayerNorm(exp_state)                 │
│                                                      │
│  Output: h (unchanged), exp_state (updated)          │
└──────────────────────────────────────────────────────┘
```

In code:

```python
# Per layer
s = h_post_attn @ W_s                        # (B, L, 1536) -> (B, L, 192)
s_sg = s.detach()                             # CRITICAL: stop gradient

s_pred = experience_state @ W_pred            # (B, L, 192) -> (B, L, 192)
curiosity = ((s_sg - s_pred) ** 2).mean()     # scalar prediction error

decay = torch.sigmoid(self.decay_raw)         # element-wise, init ~0.95
delta = F.silu(self.W_e(s))
experience_state = decay * experience_state + delta
experience_state = self.exp_norm(experience_state)  # LayerNorm stabilization
```

### Decay Gate and LayerNorm: Stabilization Fixes

The original 1B design used a simpler additive update: `exp_state = exp_state + silu(W_e @ s)`. In practice, this led to the experience state growing unboundedly through 20+ layers, which destabilized training.

Two fixes were applied:

1. **Decay gate**: Each layer has a learnable per-dimension decay vector, initialized at `sigmoid(3.0) ≈ 0.95`. This means the experience state naturally decays — information from early layers fades unless actively maintained. The gate is element-wise, so different dimensions can have different time constants.

2. **LayerNorm**: Applied after each state update to keep the experience state in a well-conditioned range. This prevents the accumulation of scale drift across 20 layers and keeps gradients flowing cleanly.

Together, these make the experience stream much more stable during training without changing its fundamental character.

### Why This Is NOT an RNN

I want to be very clear about this, because it's the first question everyone asks.

The experience stream flows **layer-to-layer**, not **token-to-token**. For any given sequence position, the experience state at layer `i` depends only on that same position's hidden states at layers 0 through i-1. There is zero cross-token dependency in the experience stream.

This means:

- **Initialization**: `exp_state` is zeros at the start of each sequence. No hidden state carries over between sequences.
- **Parallelism**: the entire forward pass is fully parallelizable over the sequence length, just like a standard transformer.
- **Recomputability**: during gradient checkpointing, you can recompute any layer's experience state from the activations — no need to cache it.
- **Inference**: the KV cache is all you need. No extra state to manage.

It's a vertical skip connection with learned dynamics, not recurrence.

### The Curiosity Mechanism

The curiosity signal is the most speculative part of Eve-3-SABER, and it connects directly to ideas from the broader Eve project about biomimetic AI.

Each layer predicts what the *next* layer will see, based on accumulated experience. When that prediction is wrong — when a layer encounters something the prior layers didn't prepare it for — that's a "curiosity" signal. The prediction error is collected as an auxiliary loss:

```
L_curiosity = 0.01 * mean(curiosity across all layers and tokens)
```

The coefficient is intentionally small. This isn't trying to compete with the cross-entropy loss. It's a gentle regularizer that says "don't let the experience stream go dead." Without it, the model could learn to ignore the experience mechanism entirely — `W_s` and `W_e` could collapse to near-zero, and the experience state would stay at its zero initialization forever.

### Why Stop-Grad on s Is Critical

This is one of the most important implementation details in the entire architecture, and it's easy to get wrong.

If you don't stop the gradient on `s` when computing curiosity, you create a perverse incentive: the model can minimize curiosity loss by making `s` *easier to predict* — which means making each layer's summary as uninformative as possible. The main representation collapses to satisfy the auxiliary loss.

With stop-grad, the curiosity loss can only train `W_pred` to be a better predictor. It can't reach back and simplify what the layers are actually computing. The experience stream stays honest.

### Parameter Cost

- `W_s`: 1536 × 192 = 295K
- `W_pred`: 192 × 192 = 37K
- `W_e`: 192 × 192 = 37K
- `decay_raw`: 192 (negligible)
- `exp_norm`: 192 (negligible)
- **Per layer: ~369K**
- **Total across 20 layers: ~7.4M** (~1.5% of total parameters)

---

## 4. Resonant FFN: Self-Regularizing Sinusoidal Modulation

### The Idea

This one came from thinking about manifold resonance — a concept from my earlier theoretical work on how learned representations might organize themselves along periodic structures in activation space.

The question: what if FFN layers could learn to modulate their outputs with periodic (sinusoidal) functions? Standard FFNs apply a monotonic nonlinearity (SiLU in SwiGLU) and a linear transformation. They can approximate any function, sure, but they have to learn periodicity from scratch through composition. What if we gave them a *native* periodic channel?

### The Architecture

Eve-3-SABER alternates between two types of FFN:

- **Even layers (0, 2, 4, ..., 18)**: Resonant FFN (10 layers)
- **Odd layers (1, 3, 5, ..., 19)**: Standard SwiGLU FFN (10 layers)

The alternating pattern is deliberate. Half the layers are conventional, providing a stable backbone. The other half have the option — but not the obligation — to use sinusoidal modulation.

Here's the resonant FFN:

```python
# Standard SwiGLU output (always computed)
ffn_out = W2(silu(W1(x)) * W3(x))         # (B, L, d_model)

# Sinusoidal modulation
freq_proj = x @ W_freq                     # (B, L, 1536) -> (B, L, 1536)
mod = torch.sin(freq_proj)                 # periodic response

# Self-regulating blend
alpha = torch.sigmoid(alpha_raw)           # learnable per-layer scalar
output = alpha * ffn_out + (1 - alpha) * (ffn_out * (1 + mod))
```

### The Alpha Mechanism

This is the part I'm most proud of from a design perspective.

`alpha_raw` is initialized at 3.0. Since sigmoid(3.0) ≈ 0.95, the blend formula at initialization is:

```
output ≈ 0.95 * ffn_out + 0.05 * (ffn_out * (1 + mod))
       = 0.95 * ffn_out + 0.05 * ffn_out + 0.05 * ffn_out * mod
       = ffn_out + 0.05 * ffn_out * mod
```

At training start, the model is **95% standard SwiGLU** with a tiny 5% resonant contribution. This means:

1. **Training begins near baseline** — the model doesn't have to learn to cope with random sinusoidal noise on day one
2. **Gradients flow cleanly** through the standard path initially
3. **Each layer independently decides** whether to increase resonance

Over training, the optimizer can push `alpha_raw` in either direction:
- **Toward +∞**: alpha → 1.0, the layer becomes pure SwiGLU (resonance isn't helping)
- **Toward -∞**: alpha → 0.0, the layer goes fully resonant (resonance is dominant)
- **Somewhere in between**: the layer has found its preferred blend

This is what I mean by "self-regularizing." If sinusoidal modulation doesn't help for a particular layer, the path of least resistance is to keep alpha high and effectively disable it. The mechanism doesn't fight the optimizer. It gets out of the way.

### Parameter Cost

- `W_freq`: 1536 × 1536 = 2,359K
- `alpha_raw`: 1 (negligible)
- **Per resonant layer: ~2.4M additional**
- **Across 10 even layers: ~23.6M** (~4.7% of total parameters)

The W_freq matrix is the single most expensive novel component in the entire architecture. It's a full d_model × d_model matrix, which is substantial. But it only appears on half the layers, and the alpha mechanism ensures that if it's not earning its keep, the model can functionally disable it.

---

## 5. Design Trade-offs and Key Decisions

### Non-Negotiable Constraints

Some decisions weren't really decisions at all — they were constraints I set early and refused to compromise on:

- **FlashAttention compatibility is non-negotiable.** Any attention modification that requires custom kernels is a dealbreaker. This is why slip-anchors modify K/V rather than the attention matrix.
- **Stop-gradient on `s` in the curiosity mechanism.** Without it, representational collapse is almost guaranteed.
- **The self-regulating alpha in Resonant FFN.** Without a way for the model to disable resonance per-layer, you're forcing an inductive bias that might not help.
- **SwiGLU with ⅔ scaling as the baseline FFN.** The right choice for this scale in 2026.
- **The ablation plan is essential.** With three novel components, you need to measure each one's contribution independently.

### Depth vs. Width

I considered two configurations: 16 layers with a wider model, or 20 layers with a narrower one. I went with 20. The reasoning: depth gives the experience stream more layers to accumulate information, and the slip-anchors more opportunities to specialize per layer. Width helps raw capacity, but the novel components benefit more from depth.

### Parameter Budget Allocation

The 0.5B budget was tuned so that novel components consume ~6.9% of total parameters — close to the 7.3% ratio of the original 1B design. This means results should transfer directionally between scales. The d_ff was tuned to 2164 to absorb the ~34.5M cost of the novel components while hitting the 500M target.

### Novel Component Summary

| Component | Per Layer | Layers | Total | % of Model |
|-----------|----------|--------|-------|------------|
| Slip-Anchors | 178K | 20 | 3.6M | 0.7% |
| Experience Stream | 369K | 20 | 7.4M | 1.5% |
| Resonant FFN | 2,359K | 10 | 23.6M | 4.7% |
| **Total novel** | | | **34.5M** | **6.9%** |

---

## 6. Training Strategy

### Data: 3-Stage Curriculum

Eve-3-SABER is trained on 50B tokens using a 3-stage curriculum designed to build broad foundations first, then specialize:

**Stage 1: Foundation (25B tokens)** — broad language + code

| Source | Proportion | Purpose |
|--------|-----------|---------|
| SmolLM python-edu | 35% | High-quality Python/ML code |
| FineWeb-Edu | 25% | Curated educational web text |
| Open-Web-Math | 15% | Mathematical reasoning |
| DCLM | 10% | Diverse curated web text |
| Cosmopedia | 10% | Synthetic educational content |
| FinePDFs | 5% | Academic/technical PDFs |

**Stage 2: Specialization (15B tokens)** — heavy code + technical

| Source | Proportion |
|--------|-----------|
| SmolLM python-edu | 50% |
| FineMath | 15% |
| FinePDFs | 15% |
| Open-Web-Math | 10% |
| Cosmopedia | 5% |
| FineWeb-Edu | 5% |

**Stage 3: Anneal (10B tokens)** — highest quality, LR → 0

| Source | Proportion |
|--------|-----------|
| SmolLM python-edu | 35% |
| FineMath | 20% |
| Cosmopedia | 15% |
| Wikipedia | 10% |
| FinePDFs | 10% |
| FineWeb-Edu | 5% |
| Open-Web-Math | 5% |

The curriculum is designed for the model's target use case: **code and tool use**, not general-purpose chat. Python-edu is the dominant source throughout, ensuring the model develops strong code understanding. The stage transitions shift toward increasingly specialized and high-quality sources.

### Learning Rate Schedule

- **Stages 1-2**: Cosine schedule, peak 3e-4, minimum 3e-5, warmup over 2000 steps
- **Stage 3**: Linear decay from 3e-5 to 0 (annealing)

### Compute

- **Primary training**: NVIDIA B200 (178 GB VRAM) on RunPod. Single GPU, bf16 mixed precision.
- **Ablation study**: Ran on H200 SXM (140 GB). Five configurations: baseline, +anchors, +experience, +resonant, full.
- **Batch size**: ~524K tokens per optimizer step (micro_batch=16, grad_accum=16, seq_len=2048).
- **Optimizer**: AdamW with betas=(0.9, 0.95), weight_decay=0.1, gradient clipping at 1.0.
- **Checkpointing**: Every 500 steps, with stage-boundary checkpoints preserved.
- **Data pipeline**: Pre-tokenized binary files read via numpy mmap, eliminating network I/O bottlenecks.

### Over-Training Rationale

50B tokens for a 500M parameter model is roughly 100× Chinchilla-optimal. This is deliberate:

1. **Architecture stress test**: With abundant data, any performance difference between the full model and ablated baselines is more likely architectural, not data-limited.
2. **Code-focused**: Code and tool-use tasks benefit from extended training more than general language, since the distribution is narrower and the model needs to memorize patterns and APIs.
3. **Practical**: The model is intended for deployment, not as a scaling law data point. Over-training produces better models for a given inference cost.

---

## 7. Post-Training Pipeline

After pretraining, Eve-3-SABER goes through a two-phase supervised fine-tuning pipeline:

**SFT Phase 1**: General instruction following using alpaca-cleaned + code instructions. This teaches the model to follow instructions and format outputs.

**SFT Phase 2**: Function calling and tool use using specialized datasets (glaive function calling, xlam, hermes). This is the core capability target — a small model that can reliably parse function schemas, generate tool calls, and process tool outputs.

An optional DPO/ORPO phase may follow for tool-use quality refinement.

---

## 8. What I Hope To Learn

This is an experiment. I want to be clear about what I'm trying to learn, not what I'm trying to prove.

### Does the experience stream actually help?

The ablation will tell. If the +experience run doesn't improve over baseline, the idea was wrong (or my implementation of it was wrong). I'll publish the results either way. The experience stream is the most biomimetic component, and I'm genuinely uncertain whether the curiosity signal provides useful training pressure or just adds noise.

### Do anchor usage patterns reveal interpretable biases?

After training, I can inspect which anchors fire for which inputs. If anchor 17 consistently activates for mathematical content and anchor 42 lights up for conversational text, that's evidence that the codebook is learning meaningful semantic archetypes. If the usage patterns are random or degenerate (one anchor dominates), the mechanism isn't working as intended.

### Do alpha values converge to interesting patterns?

Across the 10 resonant FFN layers, do the learned alpha values show a gradient? Do early layers suppress resonance while later layers embrace it? Or do all layers converge to alpha ≈ 1.0, meaning the model learned to ignore resonance entirely? This is one of the cleanest readouts of whether the resonant FFN hypothesis has any merit.

### Can this approach scale?

If the novel components help at 500M, do they help more or less at 1B? 3B? The parameter overhead is small enough that scaling is feasible. But inductive biases that help at small scale sometimes become irrelevant at large scale — the model just learns around them. I'd love to know where on that curve slip-anchors and experience streams sit.

---

## Closing

Eve-3-0.5B-SABER is an experiment, not a claim of superiority. I'm not arguing that it will beat other 500M models on benchmarks (though I'll measure and report). I'm arguing that the architectural design space for small transformers is much larger than what's currently being explored, and that there are testable hypotheses worth testing.

The three novel components — slip-anchors, experience stream, resonant FFN — are each designed to be independently evaluable, independently disableable, and compatible with standard training and inference infrastructure. If all three fail, the model degrades gracefully to a standard transformer. If any of them succeed, we learn something about what transformers can do with slightly different machinery.

Everything is open-source: model weights, training code, training logs, ablation results. The goal is to expand what's possible, not to gatekeep what's proven.

**Anthony Maio**
anthony@making-minds.ai
[making-minds.ai](https://making-minds.ai)
[github.com/anthony-maio](https://github.com/anthony-maio)
