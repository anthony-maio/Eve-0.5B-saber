---
license: apache-2.0
language:
- en
pipeline_tag: text-generation
library_name: transformers
tags:
- safetensors
- saber
- pretrained
- custom-architecture
- custom_code
- novel-architecture
- eve
- eve-3
- slip-anchors
- experience-stream
- resonant-ffn
- trust_remote_code
- research
- code
- tool-use
datasets:
- eminorhn/python-edu
- HuggingFaceFW/fineweb-edu
- open-web-math/open-web-math
- mlfoundations/dclm-baseline-1.0
- HuggingFaceTB/cosmopedia
- HuggingFaceFW/fineweb-edu-score-2
model-index: []
---

# Eve-3-0.5B-SABER

**SABER** — **S**emantic **A**nchor-**B**iased **E**xperience-**R**esonant

**Author:** Anthony Maio · [Making Minds AI](https://making-minds.ai)
**Contact:** anthony@making-minds.ai · [GitHub](https://github.com/anthony-maio) · [HuggingFace](https://huggingface.co/anthonym21)
**License:** Apache 2.0

> A 500M-parameter decoder-only language model with three novel architectural components — Slip-Anchors, Experience Stream, and Resonant FFN — built on a standard transformer skeleton. Designed for **code and tool use**, not general-purpose chat. Currently training: 50B tokens across a 3-stage curriculum (~100x Chinchilla-optimal).

> **Training Progress:** 10.8B / 50B tokens (21.6%) | Stage: Foundation | Step: 20,599

---

## Why This Model Exists

Most sub-1B models are LLaMA-small: dense decoder-only transformers with SwiGLU, RoPE, RMSNorm, no biases, and nothing architecturally surprising. Eve-3-SABER asks: what if you took that same envelope — same parameter budget, same hardware constraints, same training recipes — and made it *architecturally novel* instead?

Not novel for novelty's sake. Novel in ways that test specific hypotheses about what transformers could be doing internally if we gave them slightly different machinery. The three novel components together add only **~34.5M parameters (~6.9% of total)** — lightweight modifications designed so that if they don't help, the model can learn to ignore them.

Originally designed at 1B, budget constraints pushed it to 500M — which turned out to be a more rigorous test. At 500M with 50B tokens (100x Chinchilla-optimal), the architecture has to actually *earn its keep*.

---

## Architecture

### The Standard Skeleton

| Parameter | Value |
|-----------|-------|
| `d_model` | 1536 |
| `n_layers` | 20 |
| `n_heads` | 12 |
| `head_dim` | 128 |
| `d_ff` | 2164 |
| `vocab_size` | 50,257 (GPT-2 tokenizer) |
| `max_seq_len` | 2048 |
| **Total params** | **~500M** |

SwiGLU FFN, RoPE (θ=10000), RMSNorm (pre-norm), no biases, weight-tied LM head. The skeleton is intentionally vanilla so the novel components can be evaluated cleanly against it.

### Architecture Diagram

```
Input tokens
     │
     ▼
┌──────────────────────────────────────────────────────────────┐
│  Token Embeddings  (vocab=50257, d_model=1536)               │
└──────────────────────────────────────────────────────────────┘
     │
     ▼  ×20 layers
┌──────────────────────────────────────────────────────────────┐
│  Pre-RMSNorm                                                 │
│     │                                                        │
│     ▼                                                        │
│  Multi-Head Attention  (12 heads × 128 head_dim)             │
│     │ ① RoPE applied to Q, K  (θ=10000)                     │
│     │ ② SLIP-ANCHORS bias K,V  (codebook 64×96)             │ ← Novel 1
│     │ ③ F.scaled_dot_product_attention  (FlashAttn-2)        │
│     │                                                        │
│  EXPERIENCE STREAM  update                                   │ ← Novel 2
│     │   s = W_s(h),  curiosity = ‖s_sg − W_pred(exp)‖²      │
│     │   exp ← decay·exp + silu(W_e(s)),  LayerNorm           │
│     │                                                        │
│  Pre-RMSNorm                                                 │
│     │                                                        │
│     ▼                                                        │
│  FFN  (even layers: RESONANT-SwiGLU, odd: standard SwiGLU)  │ ← Novel 3
│     │   Resonant: α·FFN + (1-α)·FFN·(1+sin(W_freq·x))      │
│     │   α = sigmoid(α_raw), init≈0.95                        │
└──────────────────────────────────────────────────────────────┘
     │
     ▼
  Final RMSNorm → LM Head (weight-tied)
     │
     ▼
  Next-token logits  (vocab=50257)
```

---

## The Three Novel Components

### 1. Slip-Anchors — Biasing Attention Without Breaking Flash

**The problem:** How do you inject learned *semantic* biases into attention without giving up FlashAttention? Naive attention bias matrices (`softmax(QK^T/√d + bias)`) aren't supported efficiently by FlashAttention.

**The solution:** Modify K and V *before* they enter SDPA. The biases "slip into" attention through the keys and values rather than through the attention matrix directly.

Each layer maintains a learnable **codebook of 64 anchors** in a d_anchor=96 dimensional space — prototypical "attention situations" the model blends between:

```python
h_anchor = h @ W_anchor_down          # (B, L, 1536) → (B, L, 96)
scores = softmax(h_anchor @ anchors.T)  # → (B, L, 64)
anchor_ctx = scores @ anchors           # → (B, L, 96)
k_bias = anchor_ctx @ U_k              # → (B, L, 128)
v_bias = anchor_ctx @ U_v              # → (B, L, 128)

Q, K = apply_rope(Q, K)                # RoPE first
K = K + k_bias                          # Then anchor bias
V = V + v_bias
output = F.scaled_dot_product_attention(Q, K, V)  # FlashAttn sees nothing unusual
```

**Key design decisions:**
- Scores from raw hidden state `h` (not Q) to avoid circularity — Q influencing its own K biases could create degenerate fixed points
- RoPE applied *before* anchor bias to prevent positional corruption of the bias vector
- **Cost: ~3.6M params total (~0.7% of model)** — the bottleneck design makes this almost free

### 2. Experience Stream — Curiosity Without Recurrence

**The motivation:** Transformer layers have no explicit side-channel for communicating metadata about what they've done. The experience stream is a low-dimensional per-token vector (d_exp=192) that flows **layer-to-layer** — a dedicated channel for cross-layer communication outside the residual stream.

```
Layer i receives: h (residual stream) + exp_state (d_exp=192)

  1. s = W_s @ h_post_attn           — summary of what this layer "saw"
  2. s_sg = s.detach()                — STOP GRADIENT (critical!)
  3. s_pred = W_pred @ exp_state      — predict from accumulated experience
  4. curiosity = ‖s_sg − s_pred‖²    — prediction error = surprise signal
  5. decay = sigmoid(decay_raw)       — element-wise gate, init ~0.95
     delta = silu(W_e @ s)
     exp_state = decay * exp_state + delta
     exp_state = LayerNorm(exp_state)
```

**This is NOT an RNN.** The experience stream flows layer-to-layer, not token-to-token. For any sequence position, exp_state at layer `i` depends only on that position's hidden states at layers 0 through i-1. Fully parallelizable, no extra inference state needed.

**Why stop-grad on `s` is critical:** Without it, the model can minimize curiosity loss by making `s` *easier to predict* — collapsing the main representation. With stop-grad, curiosity can only train `W_pred` to be a better predictor, keeping the experience stream honest.

**Stabilization:** Decay gate (per-dimension, init ~0.95) prevents unbounded growth. LayerNorm after each update keeps scale well-conditioned across 20 layers.

**Cost: ~7.4M params total (~1.5% of model)**

### 3. Resonant FFN — Self-Regularizing Sinusoidal Modulation

**The idea:** What if FFN layers could natively modulate their outputs with periodic functions, rather than learning periodicity from scratch through composition?

Even layers (0, 2, ..., 18) augment SwiGLU with sinusoidal modulation:

```python
ffn_out = W2(silu(W1(x)) * W3(x))        # Standard SwiGLU
mod = torch.sin(x @ W_freq)               # Periodic modulation
alpha = torch.sigmoid(alpha_raw)           # Learnable blend, init ~0.95
output = alpha * ffn_out + (1 - alpha) * (ffn_out * (1 + mod))
```

**The alpha mechanism** is the key design insight. Initialized at sigmoid(3.0) ≈ 0.95, the model starts as **95% standard SwiGLU** with a tiny 5% resonant contribution. Each layer independently decides whether to increase resonance during training:
- α → 1.0: layer becomes pure SwiGLU (resonance isn't helping)
- α → 0.0: layer goes fully resonant
- Somewhere in between: the layer found its preferred blend

If sinusoidal modulation doesn't help, the path of least resistance is to keep alpha high and effectively disable it. The mechanism doesn't fight the optimizer.

**Cost: ~23.6M params total (~4.7% of model)** — W_freq is d_model × d_model but only on half the layers

### Novel Component Budget

| Component | Per Layer | Layers | Total | % of Model |
|-----------|----------|--------|-------|------------|
| Slip-Anchors | 178K | 20 | 3.6M | 0.7% |
| Experience Stream | 369K | 20 | 7.4M | 1.5% |
| Resonant FFN | 2,359K | 10 | 23.6M | 4.7% |
| **Total novel** | | | **34.5M** | **6.9%** |

---

## Training

### 3-Stage Curriculum (50B tokens)

The model targets **code and tool use**. Python-edu is the dominant source throughout.

**Stage 1: Foundation (25B tokens)** — broad language + code

| Source | Weight | Purpose |
|--------|--------|---------|
| SmolLM python-edu | 35% | High-quality Python/ML code |
| FineWeb-Edu | 25% | Curated educational web text |
| Open-Web-Math | 15% | Mathematical reasoning |
| DCLM | 10% | Diverse curated web text |
| Cosmopedia | 10% | Synthetic educational content |
| FinePDFs | 5% | Academic/technical PDFs |

**Stage 2: Specialization (15B tokens)** — heavy code + technical

| Source | Weight |
|--------|--------|
| SmolLM python-edu | 50% |
| FineMath | 15% |
| FinePDFs | 15% |
| Open-Web-Math | 10% |
| Cosmopedia | 5% |
| FineWeb-Edu | 5% |

**Stage 3: Anneal (10B tokens)** — highest quality, LR → 0

| Source | Weight |
|--------|--------|
| SmolLM python-edu | 35% |
| FineMath | 20% |
| Cosmopedia | 15% |
| Wikipedia | 10% |
| FinePDFs | 10% |
| FineWeb-Edu | 5% |
| Open-Web-Math | 5% |

### Hyperparameters

| Parameter | Value |
|-----------|-------|
| Optimizer | AdamW (β₁=0.9, β₂=0.95) |
| Weight decay | 0.1 |
| Peak LR | 3e-4 |
| Min LR | 3e-5 (stages 1-2), → 0 (stage 3) |
| Warmup | 2000 steps |
| LR schedule | Cosine (stages 1-2), linear decay (stage 3) |
| Batch size | ~524K tokens/step |
| Gradient clip | 1.0 |
| Mixed precision | bf16 |
| Attention | FlashAttention 2 (via `F.scaled_dot_product_attention`) |
| Sequence length | 2048 |

### Compute

- **Primary training:** NVIDIA B200 (178 GB VRAM), single GPU, bf16 mixed precision
- **Ablation study:** H200 SXM (140 GB) — five runs: baseline, +anchors, +experience, +resonant, full
- **Data pipeline:** Pre-tokenized binary files via numpy mmap (no network I/O bottleneck)
- **Checkpointing:** Every 500 steps, stage-boundary checkpoints preserved

### Loss

```
L_total = CrossEntropy + 0.01 × mean(curiosity_loss across all layers)
```

The curiosity coefficient is intentionally small — a gentle regularizer that prevents the experience stream from going dead, not a competing objective.

---

## Post-Training Pipeline (Planned)

1. **SFT Phase 1:** General instruction following (alpaca-cleaned + code instructions)
2. **SFT Phase 2:** Function calling and tool use (glaive, xlam, hermes datasets)
3. **Optional DPO/ORPO** for tool-use quality refinement

The target capability is a small model that reliably parses function schemas, generates tool calls, and processes tool outputs.

---

## Quick Start

### Installation

```bash
pip install transformers torch accelerate
```

### Load & Generate

```python
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch

model_id = "anthonym21/eve-3-0.5b-saber"

tokenizer = AutoTokenizer.from_pretrained(model_id)
model = AutoModelForCausalLM.from_pretrained(
    model_id,
    trust_remote_code=True,
    torch_dtype=torch.bfloat16,
    device_map="auto",
)

prompt = "def fibonacci(n):"
inputs = tokenizer(prompt, return_tensors="pt").to(model.device)

with torch.no_grad():
    output = model.generate(
        **inputs,
        max_new_tokens=200,
        do_sample=True,
        temperature=0.8,
        top_p=0.95,
    )

print(tokenizer.decode(output[0], skip_special_tokens=True))
```

### Pipeline

```python
from transformers import pipeline

pipe = pipeline(
    "text-generation",
    model="anthonym21/eve-3-0.5b-saber",
    trust_remote_code=True,
    torch_dtype="bfloat16",
    device_map="auto",
)

result = pipe("import torch\n", max_new_tokens=100, do_sample=True)
print(result[0]["generated_text"])
```

---

## Ablation Study

Five-run incremental ablation to measure each component's contribution independently:

| Run | Config | Description |
|-----|--------|-------------|
| 1 | `baseline` | Standard transformer — all novel components disabled |
| 2 | `+anchors` | Slip-Anchors only |
| 3 | `+experience` | Experience Stream + curiosity loss only |
| 4 | `+resonant` | Resonant FFN only |
| 5 | `full` | All three components enabled |

Component toggling uses config flags (`enable_anchors`, `enable_experience`) and `resonant_layers=[]` rather than zeroing dimensions, ensuring clean comparisons.

---

## Design Philosophy

**Non-negotiable constraints:**
- FlashAttention compatibility — no custom CUDA kernels
- Stop-gradient on `s` in curiosity mechanism — prevents representational collapse
- Self-regulating alpha in Resonant FFN — lets the model disable resonance per-layer
- Clean ablation — each component independently toggleable

**Depth over width:** 20 layers (narrower) rather than 16 layers (wider). Depth gives the experience stream more layers to accumulate information and slip-anchors more opportunities to specialize.

**Graceful degradation:** If all three novel components fail to help, the model degrades to a standard transformer. The architecture doesn't fight itself.

---

## Known Limitations

1. **Early in training** — model is still being trained. Expect quality to improve as training progresses through all three curriculum stages.
2. **GPT-2 vocabulary** (50,257 BPE tokens) — suboptimal for non-English text.
3. **2048 context length** — longer contexts require position interpolation or fine-tuning.
4. **Base model only** — next-token prediction, not instruction-tuned or chat-optimized (yet).
5. **Experience stream resets per sequence** — no cross-sequence state persistence at inference.

---

## What We Hope To Learn

- **Does the experience stream help?** The ablation will tell. If +experience doesn't improve over baseline, the idea was wrong. Results published either way.
- **Do anchor usage patterns reveal interpretable biases?** Post-training inspection of which anchors fire for which inputs.
- **Do alpha values converge to interesting patterns?** Do early layers suppress resonance while later layers embrace it? Or does the model learn to ignore resonance entirely?
- **Does this scale?** If novel components help at 500M, do they help more or less at 1B? 3B?

---

## Source Code

- **GitHub:** [anthony-maio/eve-1b-saber](https://github.com/anthony-maio/eve-1b-saber) — full training code, ablation runner, parameter counter, evaluation utilities
- **Technical Deep Dive:** [TECHNICAL_DEEP_DIVE.md](https://github.com/anthony-maio/eve-1b-saber/blob/master/TECHNICAL_DEEP_DIVE.md) — detailed mathematical derivations, implementation notes, and extended discussion of each novel component

---

## Citation

```bibtex
@misc{maio2026saber,
  title        = {Eve-3-0.5B-SABER: A Dense Transformer with Slip-Anchors,
                  Experience Streams, and Resonant FFNs},
  author       = {Maio, Anthony},
  year         = {2026},
  organization = {Making Minds AI},
  url          = {https://huggingface.co/anthonym21/eve-3-0.5b-saber},
  note         = {Model weights and code released under the Apache 2.0 License}
}
```

---

## Contact

| | |
|---|---|
| **Email** | anthony@making-minds.ai |
| **Website** | [making-minds.ai](https://making-minds.ai) |
| **GitHub** | [github.com/anthony-maio](https://github.com/anthony-maio) |
| **HuggingFace** | [huggingface.co/anthonym21](https://huggingface.co/anthonym21) |

Bug reports, PRs, and research collaborations welcome. Everything is open-source — model weights, training code, training logs, ablation results. The goal is to expand what's possible, not to gatekeep what's proven.
