# Eve-3-SABER-1B: Three Novel Mechanisms for Making Small Models Weird in Useful Ways

**Anthony Maio — Making Minds AI**
*March 2026*

> **SABER** — **S**emantic **A**nchor-**B**iased **E**xperience-**R**esonant. The third model in the Eve series, following Eve-2-MoE-272M. A 1B dense transformer with three novel mechanisms that test specific hypotheses about attention biasing, cross-layer memory, and FFN modulation.

---

## Opening: Why Build Another 1B Model?

There are a lot of 1B parameter language models out there right now. Most of them are essentially LLaMA-small: dense decoder-only transformers with SwiGLU, RoPE, RMSNorm, no biases, and nothing architecturally surprising. They're good. They're well-understood. They're boring.

I don't mean that as an insult. Boring is powerful — boring means the optimizer knows what to do, FlashAttention works out of the box, and you can reason about training dynamics because a thousand people have already walked that path. But I kept wondering: what if you took that exact envelope — the same 1B parameter budget, the same hardware constraints, the same training recipes — and made it *architecturally novel* instead of just scaling a known design down?

Not novel for novelty's sake. Novel in ways that test specific hypotheses about what transformers could be doing internally if we gave them slightly different machinery.

That's Eve-3-SABER-1B. It started from a single question I kept coming back to in the Eve project: *"What if we made a 1B dense model that's actually weird in a useful way?"* I refined the architecture through an iterative design process — prototyping, pressure-testing each decision against alternatives, and deliberately arguing both sides of every trade-off before committing. I'll walk through the reasoning behind the key choices.

Eve-3-SABER-1B introduces three novel architectural components on top of a standard transformer skeleton:

1. **Slip-Anchors** — learned attention biases injected through K/V modification, preserving FlashAttention compatibility
2. **Experience Stream** — a per-token, cross-layer state vector with a curiosity-driven auxiliary loss
3. **Resonant FFN** — sinusoidal modulation of feed-forward outputs with self-regulating learned blending

None of these require custom CUDA kernels. All of them are designed so that if they don't help, the model can learn to ignore them. And all of them connect to ideas I've been developing in the broader Eve project and slipstream work — semantic anchors, biomimetic curiosity, manifold resonance.

I don't know if this will work. That's the point.

---

## 1. The Standard Skeleton

Before we get to the weird stuff, let's establish what's conventional. Eve-3-SABER-1B is built on a foundation that would look familiar to anyone who's read the LLaMA or Mistral papers:

| Parameter | Value |
|-----------|-------|
| `d_model` | 2048 |
| `n_layers` | 24 |
| `n_heads` | 16 |
| `head_dim` | 128 |
| `d_ff` | 2855 |
| `vocab_size` | 50257 |
| `max_seq_len` | 2048 |

The FFN uses SwiGLU with three matrices (W1, W3, W2). The `d_ff` was tuned via a param counter script to 2855, which — once you account for the novel components consuming ~73M params — lands the total at exactly 1,000,001,548 parameters. The original ⅔-scaling estimate of 5461 overcounted because it didn't account for the slip-anchor, experience stream, and resonant FFN overhead. Positional information comes from RoPE with θ=10000. Normalization is RMSNorm in the pre-norm position. No biases anywhere in the main projections. The LM head shares weights with the token embedding layer.

Why these choices? Because they're *proven* at 1B scale. I didn't want to confound novel mechanisms with non-standard baselines. If I'm testing three new ideas, the last thing I need is to also wonder whether my normalization scheme is causing problems. The skeleton is intentionally vanilla so the novel components can be evaluated cleanly.

The standard components — embeddings (~103M), QKV projections (~302M), output projections (~101M), SwiGLU FFNs (~421M at d_ff=2855), and norms (~0.1M) — account for the vast majority of the ~1.0B parameter budget. The three novel components together add only ~73.4M parameters (~7.3% of the total). They're lightweight modifications, not heavy new machinery.

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

Each attention layer maintains a learnable **codebook** of 64 anchors, each living in a d_anchor=128 dimensional space. Think of these as prototypical "attention situations" — learned archetypes that the model can blend between.

Here's how it works step by step:

```python
# 1. Project hidden state into anchor space
h_anchor = h @ W_anchor_down          # (B, L, d_model) -> (B, L, d_anchor)
                                        # W_anchor_down: (2048, 128)

# 2. Score against the codebook
scores = softmax(h_anchor @ anchors.T)  # (B, L, d_anchor) @ (d_anchor, n_anchors)
                                        # -> (B, L, 64)

# 3. Compute weighted anchor context
anchor_ctx = scores @ anchors           # (B, L, 64) @ (64, d_anchor)
                                        # -> (B, L, 128)

# 4. Project to K and V bias
k_bias = anchor_ctx @ U_k              # (B, L, d_anchor) -> (B, L, head_dim)
v_bias = anchor_ctx @ U_v              # (B, L, d_anchor) -> (B, L, head_dim)

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

### Connection to the Broader Work

The "anchor" concept didn't come from nowhere. In the Eve project's slipstream architecture, semantic anchors are a recurring motif — the idea that meaning can be grounded by soft lookup against a learned codebook of archetypes. Slip-anchors are the attention-specific instantiation of that idea: what if attention could be grounded against a small set of learned "situations"?

### Parameter Cost

The slip-anchor machinery is remarkably cheap:

- `W_anchor_down`: 2048 × 128 = 262K
- `anchors`: 64 × 128 = 8K
- `U_k`: 128 × 128 = 16K
- `U_v`: 128 × 128 = 16K
- **Per layer: ~0.3M**
- **Total across 24 layers: ~7.2M** (less than 1% of total parameters)

That's the beauty of the bottleneck design. The d_anchor=128 space is small enough that the entire mechanism is almost free in terms of parameters, and the computation is dominated by the `h @ W_anchor_down` matmul, which is trivially fast relative to the attention operation it modifies.

---

## 3. Experience Stream: Curiosity Without Recurrence

### The Motivation

Here's something that's always struck me as odd about transformers: each layer processes the residual stream, but layers have no *explicit* channel for communicating metadata about what they've done. Yes, the residual stream carries forward all the information. But a layer's contribution is mixed into the same high-dimensional space as everything else. There's no side-channel where layer 7 can tell layer 8 "I just found something surprising."

What if there was?

The experience stream is a low-dimensional per-token vector (d_exp=256) that flows layer-to-layer. It's a dedicated channel for cross-layer communication that exists *outside* the residual stream. Each layer reads the current experience state, updates it, and optionally uses it.

### How It Works

```
┌─────────────────────────────────────────────────────┐
│                    Layer i                           │
│                                                     │
│  Input: h (residual stream), exp_state (d_exp=256)  │
│                                                     │
│  1. h_post_attn = attention(h)                      │
│                                                     │
│  2. s = W_s @ h_post_attn     (d_model -> d_exp)    │
│     [summary of what this layer "saw"]              │
│                                                     │
│  3. s_sg = s.detach()          [stop gradient!]     │
│                                                     │
│  4. s_pred = W_pred @ exp_state (d_exp -> d_exp)    │
│     [predict what we'd see based on prior layers]   │
│                                                     │
│  5. curiosity = ||s_sg - s_pred||²                  │
│     [prediction error = surprise signal]            │
│                                                     │
│  6. exp_state = exp_state + silu(W_e @ s)           │
│     [update experience with gated summary]          │
│                                                     │
│  Output: h (unchanged), exp_state (updated)         │
└─────────────────────────────────────────────────────┘
```

In code:

```python
# Per layer
s = h_post_attn @ W_s                    # (B, L, 2048) -> (B, L, 256)
s_sg = s.detach()                         # CRITICAL: stop gradient

s_pred = experience_state @ W_pred        # (B, L, 256) -> (B, L, 256)
curiosity = ((s_sg - s_pred) ** 2).mean() # scalar prediction error

experience_state = experience_state + silu(s @ W_e)  # gated update
```

### Why This Is NOT an RNN

I want to be very clear about this, because it's the first question everyone asks.

The experience stream flows **layer-to-layer**, not **token-to-token**. For any given sequence position, the experience state at layer `i` depends only on that same position's hidden states at layers 0 through i-1. There is zero cross-token dependency in the experience stream.

This means:

- **Initialization**: `exp_state` is zeros at the start of each sequence. No hidden state carries over between sequences.
- **Parallelism**: the entire forward pass is fully parallelizable over the sequence length, just like a standard transformer.
- **Recomputability**: during gradient checkpointing, you can recompute any layer's experience state from the activations — no need to cache it.
- **Inference**: the KV cache is all you need. No extra state to manage. The experience stream adds zero inference complexity because it's recomputed from scratch on every forward pass anyway.

It's a vertical skip connection with learned dynamics, not recurrence.

### The Curiosity Mechanism

The curiosity signal is the most speculative part of Eve-3-SABER-1B, and it connects directly to ideas from the broader Eve project about biomimetic AI.

Each layer predicts what the *next* layer will see, based on accumulated experience. When that prediction is wrong — when a layer encounters something the prior layers didn't prepare it for — that's a "curiosity" signal. The prediction error is collected as an auxiliary loss:

```
L_curiosity = 0.01 * mean(curiosity across all layers and tokens)
```

The coefficient is intentionally small. This isn't trying to compete with the cross-entropy loss. It's a gentle regularizer that says "don't let the experience stream go dead." Without it, the model could learn to ignore the experience mechanism entirely — `W_s` and `W_e` could collapse to near-zero, and the experience state would stay at its zero initialization forever.

### Why Stop-Grad on s Is Critical

This is one of the most important implementation details in the entire architecture, and it's easy to get wrong.

If you don't stop the gradient on `s` when computing curiosity, you create a perverse incentive: the model can minimize curiosity loss by making `s` *easier to predict* — which means making each layer's summary as uninformative as possible. The main representation collapses to satisfy the auxiliary loss.

With stop-grad, the curiosity loss can only train `W_pred` to be a better predictor. It can't reach back and simplify what the layers are actually computing. The experience stream stays honest.

### Connection to the Eve Project

The experience stream is the most directly "Eve-inspired" component in Eve-3-SABER-1B. In the Eve project's design documents, there's a concept I call *mnemos* — a kind of running experiential memory that accumulates across processing stages. The experience stream is a minimal, tractable version of that idea: what's the simplest mechanism that lets layers build a shared narrative about what's happening in the sequence?

The curiosity mechanism, specifically, connects to the project's interest in biomimetic AI. Biological neural systems are fundamentally driven by prediction error — the brain is a prediction machine, and surprise is the signal that drives learning and attention allocation. The experience stream's curiosity loss is a tiny echo of that principle: layers that encounter something unexpected generate a signal, and the model is gently encouraged to maintain that capacity for surprise.

### Parameter Cost

- `W_s`: 2048 × 256 = 524K
- `W_pred`: 256 × 256 = 65.5K
- `W_e`: 256 × 256 = 65.5K
- **Per layer: ~0.66M**
- **Total across 24 layers: ~15.8M** (~1.6% of total parameters)

---

## 4. Resonant FFN: Self-Regularizing Sinusoidal Modulation

### The Idea

This one came from thinking about manifold resonance — a concept from my earlier theoretical work on how learned representations might organize themselves along periodic structures in activation space.

The question: what if FFN layers could learn to modulate their outputs with periodic (sinusoidal) functions? Standard FFNs apply a monotonic nonlinearity (SiLU in SwiGLU) and a linear transformation. They can approximate any function, sure, but they have to learn periodicity from scratch through composition. What if we gave them a *native* periodic channel?

### The Architecture

Eve-3-SABER-1B alternates between two types of FFN:

- **Even layers (0, 2, 4, ..., 22)**: Resonant FFN
- **Odd layers (1, 3, 5, ..., 23)**: Standard SwiGLU FFN

The alternating pattern is deliberate. Half the layers are conventional, providing a stable backbone. The other half have the option — but not the obligation — to use sinusoidal modulation.

Here's the resonant FFN:

```python
# Standard SwiGLU output (always computed)
ffn_out = W2(silu(W1(x)) * W3(x))         # (B, L, d_model)

# Sinusoidal modulation
freq_proj = x @ W_freq                     # (B, L, d_model) -> (B, L, d_model)
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

### The Sinusoidal Modulation

`W_freq` is a full (d_model × d_model) matrix that projects the input into a "frequency space." The sin() function then creates periodic response patterns. The key insight is that different dimensions of the frequency projection can learn different frequencies, phases, and input sensitivities. So the modulation isn't a single sine wave — it's a d_model-dimensional vector where each component is a different sinusoidal function of the input.

Multiplying `ffn_out * (1 + mod)` means the modulation is *multiplicative*: it gates or amplifies different dimensions of the FFN output based on the sinusoidal response. Dimensions where `sin(freq_proj)` is near zero pass through unchanged. Dimensions where it's near ±1 get significantly boosted or suppressed.

### How Layers Self-Select

One of the most interesting things I hope to observe during training is the pattern of alpha values across layers. My predictions:

- **Early layers** (0, 2, 4) will likely keep alpha high — they're doing basic feature extraction where periodicity isn't obviously useful
- **Middle layers** (10, 12, 14) might open up resonance — this is where more abstract pattern matching happens
- **Late layers** (20, 22) could go either way — they're close to the output, where resonance might help with certain token predictions (numbers? code syntax?) or might just add noise

But honestly, I have no idea. That's why we're running the ablation.

### Connection to Manifold Resonance

In the Eve project's theoretical framework, there's an idea I call manifold resonance: the hypothesis that learned representations in deep networks naturally organize along low-dimensional manifolds, and that periodic structures in these manifolds carry meaningful information. Resonant FFNs are a direct test of a weak version of this hypothesis — if sinusoidal modulation of FFN outputs provides a useful inductive bias, it suggests the representations have periodic structure that the model can exploit.

This is speculative. I want to be honest about that. Manifold resonance is a theoretical idea, not an established result. Eve-3-SABER-1B's resonant FFN is one of the first concrete implementations designed to test whether the intuition has any empirical basis.

### Parameter Cost

- `W_freq`: 2048 × 2048 = 4,194K
- `alpha_raw`: 1 (negligible)
- **Per resonant layer: ~4.2M additional**
- **Across 12 even layers: ~50.4M** (~5% of total parameters)

The W_freq matrix is the single most expensive novel component in the entire architecture. It's a full d_model × d_model matrix, which is substantial. But it only appears on half the layers, and the alpha mechanism ensures that if it's not earning its keep, the model can functionally disable it.

---

## 5. Design Trade-offs and Key Decisions

Every novel architecture involves trade-offs. Here are the ones I wrestled with most during Eve-3-SABER-1B's design, and the reasoning behind my choices.

### Non-Negotiable Constraints

Some decisions weren't really decisions at all — they were constraints I set early and refused to compromise on:

- **FlashAttention compatibility is non-negotiable.** Any attention modification that requires custom kernels is a dealbreaker at 1B scale. This is why slip-anchors modify K/V rather than the attention matrix.
- **Stop-gradient on `s` in the curiosity mechanism.** Without it, representational collapse is almost guaranteed. This was clear from first principles and confirmed in early prototyping.
- **The self-regulating alpha in Resonant FFN.** Without a way for the model to disable resonance per-layer, you're forcing an inductive bias that might not help. Alpha gives the optimizer an escape hatch.
- **SwiGLU with ⅔ scaling as the baseline FFN.** This is the right choice for 1B scale in 2026. No need to innovate on the backbone.
- **The ablation plan is essential.** With three novel components, you need to measure each one's contribution independently. Otherwise you're just guessing.

### Depth vs. Width

I considered two configurations: 20 layers with a wider model, or 24 layers with a narrower one (d_ff tuned accordingly). I went with 24. The reasoning: depth gives the experience stream more layers to accumulate information, and the slip-anchors more opportunities to specialize per layer. Width helps raw capacity, but the novel components benefit more from depth.

### Anchor Scoring Input

This is covered in detail in the slip-anchors section above. The short version: scoring from Q is more attention-relevant, but scoring from `h` avoids circularity and is cheaper. I chose `h` for theoretical cleanliness. It's a clean ablation to test the alternative later.

### Activation Function for Experience Stream Update

I debated between GELU and SiLU for the gated experience update. SiLU won for consistency — it matches the SwiGLU activation used throughout the model, which means the optimizer sees similar nonlinearity landscapes across components. One less variable to worry about.

### Parameter Budget Allocation

Early designs used d_ff=5461 (the standard ⅔ scaling of 8192), which would have put the model at ~1.38B params. I chose to shrink d_ff to 2855 to hit exactly 1B, absorbing the ~73M cost of the novel components. The alternative — a 1.07B model with standard d_ff — would have made ablation comparisons messier, since the baseline and full model would have different parameter counts. Matching the budget exactly means any improvement comes from architecture, not size.

---

## 6. Training Strategy

### Data

The training data mix is:

| Source | Proportion | Purpose |
|--------|-----------|---------|
| FineWeb-Edu | 55% | High-quality educational and general text |
| DCLM | 35% | Diverse, curated web text |
| Code + Math | 10% | Structured reasoning, formal language |

This is a standard mix for a 1B model in 2026. Nothing exotic here — I want training data to be a controlled variable, not a confound.

### Two-Stage Training

The training plan uses a two-stage approach:

**Stage 1: Main training** on ~90B tokens with cosine learning rate schedule, linear warmup over 2000 steps, peak learning rate of 3e-4.

**Stage 2: Annealing** over the final ~10B tokens. Learning rate decays to near-zero. This is where the model "locks in" its learned representations and the alpha values in resonant FFN layers stabilize.

Target is 100B tokens total — roughly 5× Chinchilla optimal for a 1B model. We're intentionally over-training because the novel components might need extra data to find their groove.

### Compute

- **Primary training**: RunPod H100 PCIe instances. Not the fastest cards, but cost-effective for a research run.
- **Ablation runs**: Local RTX 3090s. The 5-run ablation plan (baseline, +anchors, +experience, +resonant, full) at reduced scale to validate each component before committing to the full 100B token run.
- **Mixed precision**: bf16 throughout, with FlashAttention 2 for SDPA.
- **Optimizer**: AdamW with betas=(0.9, 0.95), weight_decay=0.1, gradient clipping at 1.0.
- **Batch size**: ~0.5M tokens per step (micro-batches × gradient accumulation × sequence length).

### Predictability Mode

One of the more practical design ideas that came out of iterating on the architecture is the "predictability mode" configuration. The idea: ship the model with a conservative default that disables most novel components, then let researchers gradually enable them.

```python
# Predictability mode config
gate_bias = -3          # Anchor scores biased toward uniform -> nearly off
U_e_scale = 0.05        # Experience stream updates are tiny
resonant_layers = "last_8"  # Only layers 16-22 use resonant FFN
```

This gives you a model that behaves almost like a standard transformer out of the box, but can be "opened up" by adjusting three numbers. It's insurance against the novel components being harmful — you can always fall back to something conventional.

---

## 7. What I Hope To Learn

This is an experiment. I want to be clear about what I'm trying to learn, not what I'm trying to prove.

### Does the experience stream actually help?

The ablation will tell. If the +experience run doesn't improve over baseline, the idea was wrong (or my implementation of it was wrong). I'll publish the results either way. The experience stream is the most biomimetic component, and I'm genuinely uncertain whether the curiosity signal provides useful training pressure or just adds noise.

### Do anchor usage patterns reveal interpretable biases?

After training, I can inspect which anchors fire for which inputs. If anchor 17 consistently activates for mathematical content and anchor 42 lights up for conversational text, that's evidence that the codebook is learning meaningful semantic archetypes. If the usage patterns are random or degenerate (one anchor dominates), the mechanism isn't working as intended.

### Do alpha values converge to interesting patterns?

Across the 12 resonant FFN layers, do the learned alpha values show a gradient? Do early layers suppress resonance while later layers embrace it? Or do all layers converge to alpha ≈ 1.0, meaning the model learned to ignore resonance entirely? This is one of the cleanest readouts of whether the resonant FFN hypothesis has any merit.

### Can this approach scale?

If the novel components help at 1B, do they help more or less at 3B? 7B? The parameter overhead is small enough that scaling is feasible. But inductive biases that help at small scale sometimes become irrelevant at large scale — the model just learns around them. I'd love to know where on that curve slip-anchors and experience streams sit.

### Connections to Eve

Eve-3-SABER-1B is a stepping stone in the Eve project. Eve's long-term goal is to build AI systems that incorporate biomimetic principles — experience, curiosity, resonance — into their architecture, not just their training objectives. SABER is the first test of whether these ideas survive contact with actual gradient descent on actual data.

---

## Closing

Eve-3-SABER-1B is an experiment, not a claim of superiority. I'm not arguing that it will beat LLaMA-1B on benchmarks (though I'll measure and report). I'm arguing that the architectural design space for small transformers is much larger than what's currently being explored, and that there are testable hypotheses worth testing.

The three novel components — slip-anchors, experience stream, resonant FFN — are each designed to be independently evaluable, independently disableable, and compatible with standard training and inference infrastructure. If all three fail, the model degrades gracefully to a standard transformer. If any of them succeed, we learn something about what transformers can do with slightly different machinery.

Everything will be open-source: model weights, training code, training logs, ablation results. The goal is to expand what's possible, not to gatekeep what's proven.

If you're interested in this work, want to collaborate, or want to tell me why slip-anchors will never work — I'd love to hear from you.

**Anthony Maio**
anthony@making-minds.ai
[making-minds.ai](https://making-minds.ai)
[github.com/anthony-maio](https://github.com/anthony-maio)
