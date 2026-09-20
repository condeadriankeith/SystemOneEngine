# System One Decision Engine: Deep Research, Architectural Learnings & Reference Analysis

**Document Status:** Comprehensive System Architecture & Research Synthesis (v2.0)  
**Foundational Inspiration:** TypeSafe AI & Jev (Diogo Almeida, Sept 2026)  
**Primary Reference Repositories:** 
- [`snellingio/system-one`](https://github.com/snellingio/system-one.git) (System One Lite)
- [`typesafe-ai/system-one-adapter-python`](https://github.com/typesafe-ai/system-one-adapter-python.git) (Official Drop-in Adapter)
- [`TypeSafe AI Docs & Blog`](https://docs.typesafe.ai/) (`/concepts/system-one`, `/confidence`, `/patterns`)
**Local Runtime Asset:** `qwen2.5-coder:latest` (Ollama, 4.7 GB Q4_K_M)  
**Target Project Specification:** Fast, Calibrated "System One" Decision Engine ([PRD.md](file:///c:/Users/conde/OneDrive/Desktop/SystemOneEngine/PRD.md))  

---

## 1. Executive Summary & Paradigm Shift

### 1.1 The Machine-Native Intelligence Paradigm (TypeSafe / Diogo Almeida)
Modern generative Large Language Models (LLMs) were shaped by **RLHF (Reinforcement Learning from Human Feedback)** to excel at conversation and open-ended text generation. However, in enterprise and automated systems:
- **99% of future AI executions will be machine-to-machine (software-native)**, while only 1% will be human-to-chat.
- Optimizing for human conversational preference leads to **mode dropping** (favoring polite, sycophantic styles) and **severe miscalibration** (sounding confident even when hallucinating).
- Serial token decoding ($O(N)$ autoregressive generation) introduces catastrophic latency bottlenecks (3s to 300s) and syntactic fragility (broken JSON, missing keys, hallucinated enum values).

### 1.2 "System One": Fast, Non-Autoregressive Decision Engines
Inspired by Daniel Kahneman's *Thinking, Fast and Slow*, TypeSafe's Jev introduces **System One Models**:
- **Non-Autoregressive / Parallel Sampling:** Evaluates state and questions in a single forward pass ($O(1)$ decoding complexity).
- **Zero Hallucinations by Construction:** Outputs are constrained directly to predefined types (`Choice`, `Score`, `Noul/Boolean`). There is mathematically zero probability mass outside the allowable schema.
- **Epistemically Calibrated Probabilities:** Trained with **RLCD (Reinforcement Learning for Calibrated Decisions)** instead of RLHF. Predicted probabilities strictly track empirical frequency (e.g. predictions with 0.8 probability are correct 80% of the time).
- **Latency & Cost Inversion:** 70–500 ms cloud latency (40x–200x faster than frontier LLMs) or sub-35 ms on local laptop CPUs, with output tokens essentially free.

---

## 2. The Core Decision Primitives

System One standardizes all structured software decisions into three foundational primitives:

```
                               +------------------------------------+
                               |            Input State             |
                               | (Unstructured text, JSON, records) |
                               +-----------------+------------------+
                                                 |
                                                 v
                               +------------------------------------+
                               |     System One Decision Engine     |
                               +---+-------------+----------------+-+
                                   |             |                |
                +------------------+             |                +------------------+
                v                                v                                   v
       +-----------------+              +-----------------+                 +-----------------+
       |     Choice      |              |      Score      |                 |  Noul / Boolean |
       | Unordered set   |              |  Ordinal scale  |                 | Binary statement|
       |  selection      |              |  (2-10 levels)  |                 | under doubt     |
       +--------+--------+              +--------+--------+                 +--------+--------+
                |                                |                                   |
                v                                v                                   v
       {                                {                                   {
         "choice": "refund",              "score": 1.72,                      "noul": 0.94
         "probabilities": {...},          "legend": {...},                  }
         "confidence": 0.82               "probabilities": {...},
       }                                  "confidence": 0.61
                                        }
```

### 2.1 `Choice[T]` (Categorical Selection)
- **Role:** Selects one item from a discrete, unordered set of options $\{c_1, c_2, \dots, c_k\}$.
- **Input:** `instructions` string + `criteria` dictionary mapping keys to semantic descriptions.
- **Mathematical Form:** Softmax over candidate projection logits:
  $$P(c_i) = \frac{\exp(z_{c_i} / T)}{\sum_{j=1}^k \exp(z_{c_j} / T)}$$
- **Output:**
  - `choice`: Top winning key (ties broken by first-listed order).
  - `probabilities`: Normalized dictionary mapping every candidate key to its probability ($\sum P_i = 1.0$).
  - `confidence`: Distributional concentration metric $\in [0, 1]$.

### 2.2 `Score` (Ordinal Position on an Ordered Scale)
- **Role:** Places input on an ordered multi-level ladder (2 to 10 levels). Represents degrees of sentiment, urgency, seniority, or clarity.
- **Input:** `instructions` string + `criteria` list of level descriptions in ascending order (0 = lowest, $m-1$ = highest).
- **Mathematical Form:**
  The output scalar is the **probability-weighted expected value** of the level indices:
  $$\text{score} = \sum_{i=0}^{m-1} i \cdot P(i)$$
  Because it is an expectation, it smoothly interpolates between discrete levels (e.g. `1.72` between level 1 and level 2).
- **Output:**
  - `score`: Continuous scalar $\in [0, m-1]$.
  - `legend`: Mapping from index strings (`"0"`, `"1"`, ...) back to level descriptions.
  - `probabilities`: Level probabilities.
  - `confidence`: Ordinal concentration metric.

### 2.3 `Noul` / `Boolean` (Binary Assertion Under Uncertainty)
- **Role:** Evaluates whether a single factual condition holds true given the state.
- **Name Origin:** TypeSafe's term for a calibrated boolean probability.
- **Input:** `instructions` statement + optional `criteria` specifying what counts as `true` and `false`.
- **Mathematical Form:** Single probability $P(\text{yes}) \in [0.0, 1.0]$.
- **Output:**
  - `noul`: A single float representing $P(\text{yes})$. No separate confidence field is needed, as $|P - 0.5| \times 2$ already reflects decision certainty.

---

## 3. Mathematical Formulations of Confidence

A critical finding from inspecting the official `typesafe-ai/system-one-adapter-python` repository is that **Choice and Score confidence are mathematically distinct**:

### 3.1 Choice Confidence (Peak Lead over Uniform)
For an unordered categorical distribution with $n$ options, confidence measures how much the peak probability exceeds a uniform random guess:

$$\text{Confidence}_{\text{Choice}} = \frac{\max(P) - \frac{1}{n}}{1 - \frac{1}{n}}$$

- **Uniform distribution** ($\max(P) = 1/n$): $\text{Confidence} = 0.0$.
- **One-hot distribution** ($\max(P) = 1.0$): $\text{Confidence} = 1.0$.
- **Why this matters:** In a 2-choice question, $P = 0.55$ gives confidence $\frac{0.55 - 0.50}{0.50} = 0.10$ (barely above chance). In a 10-choice question, $P = 0.55$ gives confidence $\frac{0.55 - 0.10}{0.90} = 0.50$ (highly decided).

### 3.2 Score Confidence (Mean Absolute Deviation around the Mode)
Unlike Choice, Score levels possess an intrinsic **geometric ordering** ($0 < 1 < 2 < \dots < m-1$). An answer where probability is split between adjacent levels 1 and 2 is far more confident than an answer split between opposite extremes 0 and 3.

TypeSafe computes Score Confidence via the **Mean Absolute Deviation (MAD) from the modal index**:

1. Find the modal level index:
   $$k^* = \arg\max_i P(i)$$
2. Compute the expected absolute distance from the mode:
   $$\text{MAD}_{\text{mode}} = \sum_{i=0}^{m-1} P(i) \cdot |i - k^*|$$
3. Compute the MAD of a uniform distribution over $m$ levels centered at $c = \frac{m - 1}{2}$:
   $$\text{MAD}_{\text{uniform}} = \frac{1}{m} \sum_{i=0}^{m-1} |i - c|$$
4. Compute normalized concentration:
   $$\text{Confidence}_{\text{Score}} = \max\left(0.0, 1.0 - \frac{\text{MAD}_{\text{mode}}}{\text{MAD}_{\text{uniform}}}\right)$$

This metric rewards probability mass clustering tightly near the mode while penalizing bi-modal or dispersed uncertainty across the scale.

---

## 4. Advanced Architectural Patterns for Software Systems

TypeSafe's documentation outlines four battle-tested production patterns that govern how System One engines interact with software:

### 4.1 Speculative Fan-Out
In traditional LLMs, asking multiple questions costs significant latency and token fees. In System One, inference is so fast and cheap that systems can send **broad, speculative question bundles** in a single roundtrip:
- Example: Send sentiment, intent, churn risk, toxicity, language, and escalation need all in one payload.
- Downstream software branches on the relevant outputs and ignores unnecessary ones without latency penalty.

### 4.2 Confidence-Gated Routing (Two-Axis Control)
Confidence acts as an orthogonal control axis: **The answer determines WHAT to do; Confidence determines WHETHER to act automatically.**
```python
action = response.answers["action"]

if action.confidence < 0.50:
    # High uncertainty: route to human agent
    route_to_human(state)
elif action.choice == "view_balance":
    # Low stakes: 0.50 confidence is acceptable
    display_balance()
elif action.choice == "transfer_funds":
    # High stakes: require high confidence or request user confirmation
    if action.confidence > 0.85:
        execute_transfer()
    else:
        ask_user_to_confirm()
```

### 4.3 Composite Scoring
Complex business metrics (e.g. Customer Urgency or Lead Quality) should not be evaluated as a single monolithic prompt. Instead:
- Decompose into independent 1D Score questions: `severity` (0–2), `frustration` (0–2), `customer_tier` (0–3).
- Combine them in application code using normalized weighted sums:
  $$\text{Urgency} = w_1 \frac{\text{score}_1}{m_1 - 1} + w_2 \frac{\text{score}_2}{m_2 - 1} + w_3 \frac{\text{score}_3}{m_3 - 1}$$
- When business rules change, you adjust coefficients in Python code rather than re-engineering fragile prompt text!

### 4.4 Intent Routing
Classify incoming customer communications or agent events and route to:
1. Deterministic Python/SQL functions (for clean, high-confidence known intents).
2. Specialized System 2 LLMs (for complex multi-step creative writing or deep reasoning).
3. Human queues (for ambiguous, sensitive, or low-confidence edge cases).

---

## 5. Architectural Comparison: Pretrained Masking vs. Dedicated Encoder

We now have complete technical clarity on the two implementation paths:

| Feature | Approach A: Masked Logits on Pretrained LLM (`snellingio` / `system-one-lite`) | Approach B: Multi-Head Encoder Backbone (Our [PRD.md](file:///c:/Users/conde/OneDrive/Desktop/SystemOneEngine/PRD.md)) |
| :--- | :--- | :--- |
| **Model Family** | Pretrained Causal LM (Qwen 2.5-Coder 7B, Qwen 1.7B) | Bidirectional Encoder (ModernBERT-base 149M, DeBERTa-v3 44M) |
| **Training Requirement** | **Zero** (Instant off-the-shelf deployment) | **Required** (Fine-tuned on Kaggle T4 GPU using multi-task loss) |
| **Inference Mechanism** | Logits masked at fixed answer slot `{"answer": "` | Direct parallel classification & regression heads |
| **Choice Evaluation** | Next-token probability over vocabulary letters `A`, `B`, `C` | Normalized dot-product between state vector $\mathbf{h}_{\text{state}}$ and candidate embeddings $\mathbf{e}_{c_i}$ |
| **Latency (Vivobook 15 CPU)**| 120 ms – 350 ms | **15 ms – 25 ms** (ONNX INT8) |
| **RAM Consumption** | 1.5 GB – 5.0 GB | **< 300 MB – 500 MB** |
| **Multi-Question Scaling**| Re-evaluates prompt per question ($O(Q \cdot L)$ tokens) | Single shared backbone forward pass for all questions |
| **Position Bias** | Present (requires 4 cyclic rotations to neutralize) | **None** (Dot-product cross-projection is permutation equivariant) |
| **Calibration** | Uncalibrated raw softmax scaled by constant temperature | Post-hoc temperature scaling fitted via NLL minimization ($\text{ECE} \le 0.05$) |

---

## 6. Integration Strategy with Local `qwen 2.5-coder`

Your local `qwen 2.5-coder:latest` (Ollama, 4.7 GB) plays a pivotal role across three phases:

1. **Phase 1 (Immediate Zero-Shot Prototype):**
   We can implement a local adapter wrapping Ollama `qwen 2.5-coder` using the masked prompt technique. This gives us a working System One prototype on your machine immediately to test schemas and SDK clients.
2. **Phase 2 (Synthetic Data Generation & Soft-Target Teacher):**
   Using the cyclic permutation pipeline from `teacher.py`, we can run Qwen 2.5-Coder locally (or cloud endpoints) to label thousands of domain-specific samples with averaged soft probability distributions.
3. **Phase 3 (Validation & Distillation Oracle):**
   The distilled ModernBERT ONNX model can be benchmarked directly against Qwen 2.5-Coder, proving that our 149M/44M model matches or exceeds the 7B model's accuracy on structured tasks at **10x the speed and 1/10th the memory**.

---

## 7. Reference Dataset Assets Ready for Training

From `snellingio/system-one`, we have cloned and inspected:
- **`datasets/system-one-all77.soft.jsonl`**: 3,981 labeled records across agent traces, routing, policy validation, and satisfaction scoring.
- Each record contains `state`, `questions`, `expected` gold answers, and `soft_targets` generated with 4 cyclic permutations.
- This dataset serves as an ideal warm-up corpus for our Kaggle fine-tuning pipeline.
