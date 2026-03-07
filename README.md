---
license: mit
language:
- en
pipeline_tag: text-generation
library_name: transformers
tags:
- pytorch
- safetensors
- text-generation
- causal-lm
- novel-architecture
- saber
- eve
- eve-3
- slip-anchors
- experience-stream
- resonant-ffn
- trust_remote_code
- custom-architecture
datasets:
- HuggingFaceFW/fineweb-edu
- mlfoundations/dclm-baseline-1.0
model-index: []
---

# Eve-3-SABER-1B

**SABER** — **S**emantic **A**nchor-**B**iased **E**xperience-**R**esonant

**Author:** Anthony Maio · [Making Minds AI](https://making-minds.ai)  
**Contact:** anthony@making-minds.ai · [GitHub](https://github.com/anthony-maio)  
**License:** MIT

> A 1-billion-parameter decoder-only language model that introduces three independently
> ablatable architectural novelties on top of a standard transformer baseline — all
> engineered to be FlashAttention-compatible and drop-in usable with the HuggingFace
> Transformers library.

---

## What Makes Eve-3-SABER-1B Different?

Standard 1B transformer models (GPT-2 XL scale) are well-understood but architecturally
frozen. Eve-3-SABER-1B ships three novel components, each solving a distinct limitation:

| Component | What It Fixes | Where |
|-----------|---------------|-------|
| **Slip-Anchors** | Attention lacks explicit *prototype* memory — every query must discover patterns from scratch. | All 24 layers (attention K/V bias) |
| **Experience Stream** | Information flows token-to-token (KV cache) but not *layer-to-layer within a pass*. | All 24 layers (cross-layer state) |
| **Resonant FFN** | FFN non-linearity is fixed at init; the network cannot modulate frequency content. | 12 even layers (FFN augmentation) |

All three components are designed so that a 5-run ablation (baseline → +anchors →
+experience → +resonant → full) can precisely measure each component's contribution.

---

## Architecture Diagram (ASCII)

```
Input tokens
     │
     ▼
┌─────────────────────────────────────────────────────────────┐
│  Token Embeddings  (vocab=50257, d_model=2048)              │
└─────────────────────────────────────────────────────────────┘
     │
     ▼  ×24 layers
┌─────────────────────────────────────────────────────────────┐
│  Pre-RMSNorm                                                │
│     │                                                        │
│     ▼                                                        │
│  Multi-Head Attention  (16 heads × 128 head_dim)            │
│     │ ① RoPE applied to Q, K  (θ=10000)                     │
│     │ ② SLIP-ANCHORS bias K,V  (codebook 64×128)            │  ← Novel 1
│     │ ③ F.scaled_dot_product_attention  (FlashAttn-2 compat) │
│     │                                                        │
│  Pre-RMSNorm                                                │
│     │                                                        │
│     ▼                                                        │
│  FFN  (even layers: RESONANT-SwiGLU,  odd: standard SwiGLU) │  ← Novel 3
│     │   Resonant: α·FFN + (1-α)·FFN·(1+sin(W_freq·x))      │
│     │   α = sigmoid(alpha_raw), init≈0.95                   │
│     │                                                        │
│  EXPERIENCE STREAM  update                                   │  ← Novel 2
│     │   s = W_s(h),  curiosity = ‖s_sg − W_pred(exp)‖²     │
│     │   exp ← exp + silu(W_e(s))                            │
└─────────────────────────────────────────────────────────────┘
     │
     ▼
  Final RMSNorm → LM Head (weight-tied with embedding)
     │
     ▼
  Next-token logits  (vocab=50257)
```

---

## Parameter Breakdown

| Component | Per Layer | Layers | Total |
|-----------|-----------|--------|-------|
| Token Embeddings | — | 1 | ~103M |
| QKV Projections (no bias) | 3 × 2048 × 2048 | 24 | ~302M |
| O Projection | 2048 × 2048 | 24 | ~101M |
| SwiGLU FFN (W1, W3, W2) | 3 × 2048 × 2855 | 24 | ~422M |
| Slip-Anchor Params (W_anchor_down, anchors, U_k, U_v) | ~0.3M | 24 | ~7.2M |
| Experience Stream (W_s, W_pred, W_e) | ~0.66M | 24 | ~15.8M |
| Resonant FFN (W_freq + α_raw) | ~4.2M | 12 (even) | ~50.4M |
| RMSNorm (per-layer × 2 + final) | 2048 | 49 | ~0.1M |
| LM Head | — | — | 0 (tied) |
| **Total** | | | **~1,000,001,548** |

> **d_ff = 2855** (verified by `param_counter.py --tune-dff`).  The architecture spec
> listed 5461 but the param counter showed 2855 yields exactly 1,000,001,548 parameters.

---

## Quick Start

### Installation

```bash
pip install transformers torch accelerate
# For FlashAttention 2 (Ampere GPUs):
pip install flash-attn --no-build-isolation
```

### Load the Model

```python
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch

model_id = "anthony-maio/eve-3-saber-1b"   # HuggingFace Hub path (after upload)

tokenizer = AutoTokenizer.from_pretrained(model_id)
model = AutoModelForCausalLM.from_pretrained(
    model_id,
    trust_remote_code=True,          # required for custom architecture
    torch_dtype=torch.bfloat16,
    device_map="auto",
)

prompt = "The discovery that changed everything was"
inputs = tokenizer(prompt, return_tensors="pt").to(model.device)

with torch.no_grad():
    output_ids = model.generate(
        **inputs,
        max_new_tokens=200,
        do_sample=True,
        temperature=0.8,
        top_p=0.95,
    )

print(tokenizer.decode(output_ids[0], skip_special_tokens=True))
```

### Load from Local Directory

```python
import sys
sys.path.insert(0, "./eve-3-saber-1b")   # path containing configuration_saber.py

from configuration_saber import SABERConfig
from modeling_saber import SABERForCausalLM

config = SABERConfig.from_pretrained("./eve-3-saber-1b")
model  = SABERForCausalLM.from_pretrained("./eve-3-saber-1b", config=config)
```

### Text Generation Pipeline

```python
from transformers import pipeline

pipe = pipeline(
    "text-generation",
    model=model_id,
    trust_remote_code=True,
    torch_dtype="bfloat16",
    device_map="auto",
)

result = pipe("Once upon a time,", max_new_tokens=100, do_sample=True)
print(result[0]["generated_text"])
```

---

## Novel Component Descriptions

### 1. Slip-Anchors

A learnable codebook of 64 prototype vectors (d_anchor=128) that biases the Key and
Value tensors in every attention layer **after** RoPE application. Each token position
soft-assigns to anchors via a softmax over `h @ anchors.T`, then projects the weighted
sum back to K/V space. This gives attention a persistent, trainable memory of prototypical
patterns without breaking FlashAttention compatibility.

- Parameters per layer: W_anchor_down (2048→128), anchors (64×128), U_k (128→128), U_v (128→128)
- Applied: all 24 layers
- Controlled by: `config.enable_anchors` (ablation flag)

### 2. Experience Stream

A per-token, low-dimensional state vector (d_exp=256) that evolves **layer-to-layer**
within a single forward pass, reset to zero at the start of each sequence. Unlike the
KV cache (which flows token-to-token), the experience stream lets early layers communicate
a compressed "context summary" to later layers beyond what flows through residual connections.

A **curiosity auxiliary loss** (`L_curiosity = 0.01 × ‖s_sg − W_pred(exp)‖²`) with
stop-gradient on the target encourages the stream to carry predictive information without
collapsing.

- Parameters per layer: W_s (2048→256), W_pred (256→256), W_e (256→256)
- Applied: all 24 layers
- Controlled by: `config.enable_experience` (ablation flag)

### 3. Resonant FFN

Even-indexed layers (0, 2, 4, …, 22) augment standard SwiGLU with a sinusoidal
modulation. A learned scalar α (initialized so sigmoid(α_raw)=0.95 ≈ pure SwiGLU at
init) blends the standard and modulated outputs:

```
output = α · FFN(x)  +  (1-α) · FFN(x) · (1 + sin(W_freq · x))
```

The network self-selects how much resonance each layer uses as training progresses.
Layers that prefer pure SwiGLU will keep α close to 1.0; layers that find sinusoidal
modulation useful will push α lower.

- Parameters per resonant layer: W_freq (2048×2048), α_raw (scalar)
- Applied: 12 even layers (layers 0, 2, 4, …, 22)
- Controlled by: `config.resonant_layers` (empty list = disabled)

---

## Training Details

| Hyperparameter | Value |
|---|---|
| Optimizer | AdamW |
| Learning rate | 3e-4 (peak) |
| LR schedule | Cosine with linear warmup (2000 steps) |
| Betas | (0.9, 0.95) |
| Weight decay | 0.1 |
| Gradient clip | 1.0 |
| Batch size | ~0.5M tokens (micro_batch × grad_accum × seq_len) |
| Mixed precision | bf16 |
| Attention | FlashAttention 2 (via F.scaled_dot_product_attention) |
| Token budget (target) | 100B tokens (5× Chinchilla optimal) |

### Training Data Mix

| Source | Weight | Description |
|--------|--------|-------------|
| [FineWeb-Edu](https://huggingface.co/datasets/HuggingFaceFW/fineweb-edu) | 55% | High-quality educational web text |
| [DCLM-Baseline-1.0](https://huggingface.co/datasets/mlfoundations/dclm-baseline-1.0) | 35% | Curated web crawl |
| Code + Math | 10% | Diverse reasoning data |

Last 10% of tokens uses cosine annealing (learning rate decayed to 10% of peak).

---

## Ablation Study

Eve-3-SABER-1B was developed using a 5-run incremental ablation protocol:

| Run | Configuration | Description |
|-----|---------------|-------------|
| 1 | baseline | Standard transformer — all novel components off |
| 2 | +anchors | Slip-Anchors only |
| 3 | +experience | Experience stream + curiosity loss only |
| 4 | +resonant | Resonant FFN only |
| 5 | full | All components enabled |

Run ablations yourself with `ablation_runner.py`:

```bash
python ablation_runner.py --ablation all --tokens 1e9 --wandb
```

---

## Evaluation Utilities

`eval_utils.py` provides tools for monitoring novel components during and after training:

```python
from eval_utils import (
    compute_perplexity,
    compute_anchor_entropy,
    get_alpha_values,
    run_component_analysis,
    create_alpha_plot,
    create_anchor_usage_heatmap,
)

# Comprehensive per-layer analysis
run_component_analysis(model)

# Monitor alpha evolution
alpha_history = {layer_idx: [] for layer_idx in range(0, 24, 2)}
# ... (log get_alpha_values(model) at each eval step) ...
create_alpha_plot(alpha_history, "alpha_evolution.png")
```

---

## Known Limitations

1. **Vocabulary size**: Uses GPT-2's 50,257 BPE vocabulary. Suboptimal for non-English
   text. Future versions may use a larger, multilingual tokenizer.

2. **Sequence length**: Trained on 2048-token contexts. Long-form document understanding
   requires position interpolation or fine-tuning at longer lengths.

3. **Experience stream at inference**: The experience state resets to zero each forward
   pass; it does not persist across turns in a conversational setting without explicit
   state management.

4. **No RLHF / instruction tuning**: This is a base language model (next-token prediction
   only). It has not been fine-tuned to follow instructions or be chat-optimized.

5. **Ablation scope**: Ablation runs use 1B tokens by default (1% of full training budget)
   for compute efficiency. Results may not perfectly predict behavior at full scale.

6. **Compute**: Training the full model to 100B tokens requires ~8× A100-80GB GPUs for
   a practical run time. Expect higher loss on smaller hardware.

---

## Files

```
eve-3-saber-1b/
├── config.json                # Serialised SABERConfig (d_ff=2855)
├── configuration_saber.py  # Config class (trust_remote_code)
├── modeling_saber.py       # Full model implementation (trust_remote_code)
├── ablation_runner.py         # 5-run ablation orchestration script
├── eval_utils.py              # Evaluation and logging utilities
├── train_eve3_saber_1b.ipynb    # Full training notebook
├── param_counter.py           # Parameter counting and d_ff tuning
├── TECHNICAL_DEEP_DIVE.md     # In-depth architecture documentation
├── generation_config.json     # Default HF generation config
├── requirements.txt           # Python dependencies
├── tokenizer.json             # GPT-2 tokenizer
├── tokenizer_config.json
├── special_tokens_map.json
└── model.safetensors          # Trained weights (after training)
```

---

## Full Technical Deep Dive

For detailed mathematical derivations, implementation notes, and extended discussion of
each novel component, see [TECHNICAL_DEEP_DIVE.md](./TECHNICAL_DEEP_DIVE.md).

---

## Citation

If you use Eve-3-SABER-1B in your research, please cite:

```bibtex
@misc{maio2026saber,
  title        = {Eve-3-SABER-1B: A Dense Transformer with Slip-Anchors,
                  Experience Streams, and Resonant FFNs},
  author       = {Maio, Anthony},
  year         = {2026},
  organization = {Making Minds AI},
  url          = {https://huggingface.co/anthony-maio/eve-3-saber-1b},
  note         = {Model weights and code released under the MIT License}
}
```

---

## Contact

| | |
|---|---|
| **Email** | anthony@making-minds.ai |
| **Website** | [making-minds.ai](https://making-minds.ai) |
| **GitHub** | [github.com/anthony-maio](https://github.com/anthony-maio) |
| **HuggingFace** | [huggingface.co/anthony-maio](https://huggingface.co/anthony-maio) |

Bug reports, PRs, and research collaborations welcome.
