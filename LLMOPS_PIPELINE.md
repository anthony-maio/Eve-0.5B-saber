# LLMOps Production Pipeline for Eve-3-SABER-1B

**Anthony Maio — Making Minds AI**
*March 2026*

> A practical guide to what happens between "my model trained" and "my model is in production." Written as a learning reference — the pipeline I'd build if I were shipping Eve-3-SABER-1B as a real product.

---

## The Big Picture

Training a model is maybe 20% of the work. The other 80% is everything that happens after: evaluating whether it's safe, benchmarking it against the field, packaging it for deployment, monitoring it in production, and having a plan for when things go wrong.

This doc walks through the full pipeline, stage by stage, using the tools and frameworks that industry actually uses right now (as of early 2026). I'm writing this for myself as much as anyone — this is the LLMOps knowledge I need to build alongside the architecture knowledge.

---

## Stage 1: Academic Benchmarks (Does it even work?)

Before anything else, you need to know where your model stands on the standard benchmarks. This is the "does it even work" stage.

### Tool: EleutherAI lm-evaluation-harness

The [lm-evaluation-harness](https://github.com/EleutherAI/lm-eval-harness) is the standard framework — it's what powers the HuggingFace Open LLM Leaderboard. It supports 60+ benchmarks out of the box and works with any HF-compatible model.

```bash
pip install lm-eval

# Run core benchmarks for a 1B model
lm_eval --model hf \
  --model_args pretrained=anthonym21/Eve-3-SABER-1B,dtype=bfloat16 \
  --tasks mmlu,hellaswag,arc_easy,arc_challenge,winogrande,truthfulqa_mc2,gsm8k \
  --batch_size auto \
  --output_path ./eval_results
```

### Benchmarks That Matter for a 1B Pretrained Model

| Benchmark | What It Tests | Why It Matters |
|-----------|--------------|----------------|
| **MMLU** (5-shot) | Broad knowledge across 57 subjects | The "SAT for LLMs" — general capability |
| **HellaSwag** (10-shot) | Commonsense reasoning / sentence completion | Tests whether the model understands how the world works |
| **ARC-Challenge** (25-shot) | Grade-school science questions | Reasoning under knowledge constraints |
| **WinoGrande** (5-shot) | Pronoun resolution / commonsense | Subtle language understanding |
| **TruthfulQA** (0-shot) | Resistance to common misconceptions | Does it repeat popular falsehoods? |
| **GSM8K** (5-shot, CoT) | Grade-school math with chain-of-thought | Arithmetic and step-by-step reasoning |

### What To Expect at 1B Scale

Realistic targets for a well-trained 1B dense model (for comparison against peers like TinyLlama-1.1B, SmolLM2-1.7B, Pythia-1.4B):

- MMLU: 25-35% (random baseline is 25%, so anything above that shows learning)
- HellaSwag: 55-65%
- ARC-Challenge: 30-40%
- WinoGrande: 55-65%
- TruthfulQA: 35-45%

The novel SABER components (slip-anchors, experience stream, resonant FFN) should show up in benchmarks that reward compositional reasoning — HellaSwag, WinoGrande, and ARC are the ones to watch.

### Custom Ablation Benchmarks

Beyond standard benchmarks, the ablation suite (`ablation_runner.py`) runs five configurations (baseline, +anchors, +experience, +resonant, full) across the same evals. This is where we prove the novel components actually help:

- If slip-anchors help: expect improvements on tasks requiring pattern matching (HellaSwag, ARC)
- If experience stream helps: expect improvements on tasks requiring cross-context reasoning (WinoGrande)
- If resonant FFN helps: expect smoother loss curves and better perplexity across the board

---

## Stage 2: Safety Evaluation (Will it cause harm?)

This is the stage most indie researchers skip. Don't skip it. Even at 1B parameters, a model can generate toxic content, reinforce stereotypes, or produce dangerous instructions.

### 2a. Toxicity Testing

**Tool: RealToxicityPrompts + Perspective API**

[RealToxicityPrompts](https://huggingface.co/datasets/allenai/real-toxicity-prompts) contains 99K+ sentence starters. You generate completions and score them for toxicity using Google's Perspective API.

```python
from datasets import load_dataset

# Load prompts
dataset = load_dataset("allenai/real-toxicity-prompts", split="train")

# For each prompt:
# 1. Generate 25 continuations (nucleus sampling, p=0.9)
# 2. Score each with Perspective API
# 3. Report: expected maximum toxicity, toxicity probability
```

**Key Metrics:**
- Expected Maximum Toxicity: average of worst-case toxicity across prompts
- Toxicity Probability: % of prompts where at least one completion exceeds 0.5 toxicity score
- Target: lower is better. Compare against published baselines (GPT-2 XL: ~0.52 expected max toxicity)

### 2b. Bias Evaluation

**Tools: CrowS-Pairs, BBQ, StereoSet**

| Dataset | What It Tests | Size |
|---------|--------------|------|
| [CrowS-Pairs](https://huggingface.co/datasets/crows_pairs) | Stereotype preference across 9 categories (race, gender, religion, etc.) | 1,508 pairs |
| [BBQ](https://huggingface.co/datasets/heegyu/bbq) | Bias in question answering across 11 social categories | 58K questions |
| [StereoSet](https://huggingface.co/datasets/McGill-NLP/stereoset) | Stereotype vs. anti-stereotype preference | 17K instances |

CrowS-Pairs gives you a single number: % of time the model prefers the stereotypical sentence over the anti-stereotypical one. Ideal is 50% (no preference). Higher = more biased.

```bash
# CrowS-Pairs is available in lm-eval-harness
lm_eval --model hf \
  --model_args pretrained=anthonym21/Eve-3-SABER-1B,dtype=bfloat16 \
  --tasks crows_pairs_english \
  --output_path ./safety_results
```

### 2c. Truthfulness

**TruthfulQA** (already in the benchmark suite above) covers this — it tests whether the model repeats common misconceptions. At 1B scale, models tend to be less confidently wrong than larger models, but the test is still informative.

### 2d. Red Teaming

Red teaming is adversarial testing — trying to make the model produce harmful outputs through creative prompting.

**For a pretrained base model (no instruction tuning):** Red teaming is less relevant because the model doesn't follow instructions. It just completes text. But you should still check:

- Does it complete violent/harmful prompts enthusiastically?
- Does it generate personally identifiable information from training data?
- Does it produce instructions for dangerous activities?

**Tools:**
- [Anthropic Red Team Dataset](https://huggingface.co/datasets/Anthropic/hh-rlhf) — 38K adversarial conversations for reference
- [HarmBench](https://github.com/centerforaisafety/HarmBench) — automated red-teaming framework
- [promptfoo](https://github.com/promptfoo/promptfoo) — open-source tool for running adversarial prompt suites

**If you later instruction-tune the model**, red teaming becomes critical. At that point, use automated red-teaming pipelines that generate adversarial prompts across categories: prompt injection, jailbreaks, harmful content generation, PII extraction, and deceptive alignment.

---

## Stage 3: Model Card and Documentation

Before releasing anything, write a model card. This isn't bureaucracy — it's how responsible AI works.

### What Goes in a Model Card

The `README.md` in the repo already has the structure. Here's what matters for safety:

1. **Intended Use**: "Research and experimentation with novel transformer architectures. Not intended for production applications without further safety evaluation."
2. **Out-of-Scope Uses**: "Not suitable for generating content in safety-critical domains (medical, legal, financial advice). Not instruction-tuned — will not reliably follow user instructions."
3. **Bias, Risks, and Limitations**: Document findings from Stage 2.
4. **Training Data**: List all datasets and their known limitations.
5. **Evaluation Results**: Full benchmark table.

### Frameworks

- [NIST AI Risk Management Framework](https://www.nist.gov/itl/ai-risk-management-framework) — the four functions: Govern, Map, Measure, Manage. Even for a research release, thinking through these is valuable.
- [EU AI Act](https://artificialintelligenceact.eu/) — if this model were deployed in the EU, it would need to comply. Know the risk tier your use case falls into.

---

## Stage 4: Packaging and Distribution

### HuggingFace Hub Upload

```python
from huggingface_hub import HfApi, create_repo

api = HfApi()
create_repo("anthonym21/Eve-3-SABER-1B", repo_type="model")

# Upload model files
api.upload_folder(
    folder_path="./eve-3-saber-1b",
    repo_id="anthonym21/Eve-3-SABER-1B",
    repo_type="model",
)
```

### File Checklist for HF Repo

| File | Purpose |
|------|---------|
| `configuration_saber.py` | Config class (auto-loaded with `trust_remote_code=True`) |
| `modeling_saber.py` | Model implementation |
| `model.safetensors` | Trained weights (after training) |
| `config.json` | Serialized config |
| `tokenizer.json` + `tokenizer_config.json` | Tokenizer files |
| `generation_config.json` | Default generation parameters |
| `README.md` | Model card with YAML frontmatter |

### Quantization for Deployment

For serving a 1B model efficiently:

```bash
# GGUF quantization for llama.cpp / Ollama
pip install llama-cpp-python
python -m llama_cpp.convert --outfile eve3-saber-1b.gguf ./eve-3-saber-1b/

# AWQ 4-bit for vLLM
pip install autoawq
python -c "
from awq import AutoAWQForCausalLM
model = AutoAWQForCausalLM.from_pretrained('./eve-3-saber-1b')
model.quantize(tokenizer, quant_config={'w_bit': 4, 'q_group_size': 128})
model.save_quantized('./eve-3-saber-1b-awq')
"
```

At 1B params, the model is already small enough to run on consumer hardware (4GB VRAM at fp16, <2GB at 4-bit). Quantization mainly matters for edge deployment or when serving at high throughput.

---

## Stage 5: Inference Serving (If Deploying as an API)

### Options (ranked by complexity)

| Platform | Best For | Key Feature |
|----------|----------|-------------|
| **Ollama** | Local / single-user | Dead simple, runs GGUF models |
| **vLLM** | Production API serving | PagedAttention, high throughput |
| **TGI** (HuggingFace) | HF-ecosystem deployments | Native HF model support |
| **TensorRT-LLM** (NVIDIA) | Maximum throughput on NVIDIA GPUs | Kernel-level optimizations |

For Eve-3-SABER-1B, **vLLM** is the practical choice for anything beyond local use:

```bash
pip install vllm

# Serve the model
python -m vllm.entrypoints.openai.api_server \
  --model anthonym21/Eve-3-SABER-1B \
  --trust-remote-code \
  --dtype bfloat16 \
  --port 8000
```

This gives you an OpenAI-compatible API endpoint. Any client that speaks OpenAI API format can use it.

### Key Serving Metrics

| Metric | What It Measures | Target |
|--------|-----------------|--------|
| **TTFT** (Time to First Token) | Latency before streaming starts | <500ms for 1B model |
| **TPOT** (Time Per Output Token) | Generation speed | <20ms/token on H100 |
| **Throughput** | Requests/second at target latency | Depends on batch size |
| **p99 Latency** | Worst-case response time | <2x median |

---

## Stage 6: Production Monitoring

If the model is serving traffic, you need to monitor it continuously.

### What to Monitor

**Performance Metrics:**
- Token throughput and latency (TTFT, TPOT)
- Error rates and timeout rates
- GPU utilization and memory usage
- Queue depth (if batching)

**Quality Metrics:**
- Output toxicity scores (sample and score with Perspective API)
- Perplexity on a held-out validation set (detect drift)
- User feedback signals (if applicable)
- Output length distribution (sudden changes indicate problems)

**Cost Metrics:**
- Tokens served per dollar
- GPU-hours per 1M tokens
- Cost per request

### Tools

| Tool | What It Does |
|------|-------------|
| [Weights & Biases](https://wandb.ai) | Experiment tracking during training; production monitoring |
| [WhyLabs / whylogs](https://whylabs.ai) | Data and model drift detection |
| [Arize AI](https://arize.com) | LLM observability — traces, evals, guardrails |
| [MLflow](https://mlflow.org) | Experiment tracking, model registry, deployment |
| [Prometheus + Grafana](https://prometheus.io) | Infrastructure metrics (GPU, memory, latency) |

### Drift Detection

Model drift = the model's behavior changes over time, even though the weights are frozen. This happens because the input distribution changes.

Monitor for:
- **Input drift**: Are users sending different kinds of prompts than you tested on?
- **Output drift**: Are the model's response patterns changing? (length, sentiment, topic distribution)
- **Performance drift**: Are benchmark scores degrading when re-evaluated periodically?

---

## Stage 7: Incident Response

Things will go wrong. Have a plan.

### Runbook Template

```
INCIDENT: [Model producing harmful output / Latency spike / etc.]
SEVERITY: [P0-Critical / P1-High / P2-Medium / P3-Low]

IMMEDIATE:
  - [ ] Activate rate limiting or kill switch
  - [ ] Collect affected output samples
  - [ ] Notify stakeholders

INVESTIGATION:
  - [ ] Check input patterns (adversarial? new use case?)
  - [ ] Check serving infra (OOM? GPU errors?)
  - [ ] Check if model weights/config were modified

RESOLUTION:
  - [ ] Apply input/output filters if content issue
  - [ ] Roll back to previous model version if regression
  - [ ] Document root cause and prevention

POST-MORTEM:
  - [ ] Write incident report
  - [ ] Update monitoring to catch this class of issue
  - [ ] Update red-team test suite
```

---

## Stage 8: Continuous Improvement Loop

The pipeline isn't linear — it's a loop:

```
Train → Evaluate → Safety Test → Deploy → Monitor → [Issues Found] → Retrain
                                              ↓
                                     [All Good] → Collect feedback → Improve data → Retrain
```

### What This Looks Like in Practice

1. **Weekly**: Check monitoring dashboards, review sampled outputs
2. **Monthly**: Re-run safety benchmarks on any new model versions
3. **Quarterly**: Full red-team exercise, update model card, review incident log
4. **On new release**: Full pipeline — benchmarks, safety, model card, staged rollout

---

## The Pipeline at a Glance

```
┌─────────────────────────────────────────────────────────┐
│                    TRAINING (Colab H100)                 │
│  train_eve3_saber_1b.ipynb → model.safetensors          │
└─────────────────────┬───────────────────────────────────┘
                      ▼
┌─────────────────────────────────────────────────────────┐
│              ACADEMIC BENCHMARKS                         │
│  lm-eval-harness: MMLU, HellaSwag, ARC, WinoGrande     │
│  + Custom ablation benchmarks (5-run suite)             │
└─────────────────────┬───────────────────────────────────┘
                      ▼
┌─────────────────────────────────────────────────────────┐
│              SAFETY EVALUATION                           │
│  Toxicity: RealToxicityPrompts + Perspective API        │
│  Bias: CrowS-Pairs, BBQ, StereoSet                     │
│  Truthfulness: TruthfulQA                               │
│  Red Team: HarmBench / promptfoo (for IT variants)      │
└─────────────────────┬───────────────────────────────────┘
                      ▼
┌─────────────────────────────────────────────────────────┐
│              DOCUMENTATION                               │
│  Model card (README.md), eval results, known limits     │
│  NIST AI RMF alignment, training data documentation     │
└─────────────────────┬───────────────────────────────────┘
                      ▼
┌─────────────────────────────────────────────────────────┐
│              PACKAGING & DISTRIBUTION                    │
│  HuggingFace Hub upload, quantized variants (GGUF/AWQ)  │
│  Version tagging, reproducibility artifacts             │
└─────────────────────┬───────────────────────────────────┘
                      ▼
┌─────────────────────────────────────────────────────────┐
│              SERVING (if deployed)                        │
│  vLLM / TGI behind API, load balancing, auto-scaling    │
└─────────────────────┬───────────────────────────────────┘
                      ▼
┌─────────────────────────────────────────────────────────┐
│              MONITORING & INCIDENT RESPONSE               │
│  Latency/throughput, toxicity sampling, drift detection  │
│  Runbook, kill switch, rollback procedure                │
└─────────────────────┬───────────────────────────────────┘
                      ▼
                   [LOOP]
```

---

## Further Reading

- [NIST AI Risk Management Framework](https://www.nist.gov/itl/ai-risk-management-framework) — the US government's framework for AI risk. Voluntary but increasingly referenced in regulation.
- [EleutherAI lm-evaluation-harness](https://github.com/EleutherAI/lm-eval-harness) — the standard benchmarking tool.
- [RealToxicityPrompts](https://huggingface.co/datasets/allenai/real-toxicity-prompts) (Gehman et al., 2020) — toxicity stress-testing.
- [TruthfulQA](https://huggingface.co/datasets/truthfulqa/truthful_qa) (Lin et al., 2021) — factual alignment benchmark.
- [CrowS-Pairs](https://huggingface.co/datasets/crows_pairs) (Nangia et al., 2020) — bias measurement.
- [HarmBench](https://github.com/centerforaisafety/HarmBench) — automated red-teaming framework.
- [promptfoo](https://github.com/promptfoo/promptfoo) — open-source LLM testing and red-teaming.
- [Anthropic Red Team Dataset](https://huggingface.co/datasets/Anthropic/hh-rlhf) — 38K adversarial conversations.
- [Generative AI in Action](https://www.manning.com/books/generative-ai-in-action) (Bahree, 2024) — covers LLMOps practices in depth.
- [MLOps in 2026: The Definitive Guide](https://rahulkolekar.com/mlops-in-2026-the-definitive-guide-tools-cloud-platforms-architectures-and-a-practical-playbook/) — comprehensive overview of the current MLOps landscape.
