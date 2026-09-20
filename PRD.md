# Product Requirements Document (PRD)

**Project:** Fast, Calibrated "System One" Decision Engine  
**Status:** Draft / Ready for Implementation  
**Document Version:** 1.0.0  
**Target Environments:** Cloud Training (Kaggle T4/P100 GPUs) | Local Runtime (Asus Vivobook 15 CPU)  

---

## 1. Executive Summary & Vision

### 1.1 Problem Statement
Modern generative Large Language Models (LLMs) operate autoregressively, generating tokens sequentially ($O(N)$ inference complexity). When used strictly for structured decision-making—such as agent routing, guardrail filtering, intent classification, and multi-factor scoring—autoregressive generation presents critical liabilities:
- **Excess Latency:** 200–2,000 ms time-to-first-token and decoding delays.
- **Syntactic Fragility:** Probabilistic JSON generation often violates schemas or outputs hallucinations.
- **Poor Calibration:** Generative token probabilities do not reflect true statistical confidence, leading to overconfident agent failures.
- **Compute Inefficiency:** Consuming high-end server GPUs simply to generate a boolean flag or categorical classification.

### 1.2 Solution Concept
This project will build a specialized, non-autoregressive System One Decision Engine modeled after TypeSafe's Jev. Instead of token generation, the model uses an encoder or truncated transformer backbone with specialized, parallel classification and regression heads to evaluate structured decisions in a single forward pass ($O(1)$ decoding complexity).

The system maps state inputs directly into validated, typed primitives (`Choice`, `Boolean`, `Score`) with post-hoc probability calibration, operating in under 35 ms on consumer-grade laptop CPUs.

---

## 2. Target Persona & User Scenarios

| Persona | Primary Goal | Key Requirement |
| :--- | :--- | :--- |
| **Agentic Workflow Architect** | Route user requests between tools and sub-agents. | Deterministic outputs, sub-50 ms roundtrips, valid fallback paths. |
| **Edge / Local AI Developer** | Run guardrails and intent parsers locally without an expensive GPU. | Sub-500 MB RAM footprint, fast CPU execution via ONNX. |
| **Risk & Compliance Engineer** | Verify model confidence before executing autonomous actions. | Well-calibrated probabilities ($\text{Expected Calibration Error} < 0.05$). |

---

## 3. Core Functional & Architectural Specifications

```
                     +---------------------------------------+
                     |             Input Payload             |
                     | Context: "Transfer $500 to savings"   |
                     | Schema:  Action { Transfer, Balance } |
                     +-------------------+-------------------+
                                         |
                                         v
                     +---------------------------------------+
                     |         Transformer Backbone          |
                     |      (ModernBERT / DeBERTa-v3)        |
                     |        Single Forward Pass            |
                     +-------------------+-------------------+
                                         |
                                [CLS] Token / Pooled State
                                         |
            +----------------------------+---------------------------+
            |                            |                           |
            v                            v                           v
   +-----------------+          +-----------------+         +-----------------+
   |   Choice Head   |          |  Boolean Head   |         |   Score Head    |
   | Dynamic Dot-Prod|          | Dense -> Sigmoid|         | Dense -> Linear |
   |  (Cross-Attn)   |          |                 |         |   Bounded [0-1] |
   +--------+--------+          +--------+--------+         +--------+--------+
            |                            |                           |
            v                            v                           v
   +-------------------------------------------------------------------------+
   |                  Calibration Layer (Temperature Scaling)                |
   +-------------------------------------------------------------------------+
            |                            |                           |
            v                            v                           v
   Target: "Transfer"           Confirmed: True             Confidence: 0.94
   Latency: 28ms                Latency: 28ms               Latency: 28ms
```

### 3.1 Supported Decision Primitives
The model will discard generative heads and output directly to typed contracts:

- **`Choice[T]` (Categorical Selection):**
  - Evaluates a dynamically passed list of candidate options $\{c_1, c_2, \dots, c_k\}$.
  - Computes candidate embeddings using either the backbone or a lightweight projection, followed by a normalized dot-product with the context vector $\mathbf{h}_{\text{state}}$:
    $$P(c_i \mid \mathbf{h}_{\text{state}}) = \frac{\exp\left(\frac{\mathbf{h}_{\text{state}}^\top \mathbf{W}_c \mathbf{e}_{c_i}}{\sqrt{d}}\right)}{\sum_{j=1}^k \exp\left(\frac{\mathbf{h}_{\text{state}}^\top \mathbf{W}_c \mathbf{e}_{c_j}}{\sqrt{d}}\right)}$$

- **`Boolean` (Binary Confirmation / Guardrail):**
  - Single linear projection followed by a calibrated Sigmoid activation.
  - Evaluates policy compliance, toxicity, or affirmative/negative validation.

- **`Score` (Continuous Scalar / Rubric):**
  - Linear projection with bounded activation (Sigmoid or Scaled ReLU) to map continuous states directly to $[0.0, 1.0]$ or $[0, 100]$.

### 3.2 Probability Calibration
Raw Softmax and Sigmoid outputs will be calibrated using held-out validation data to avoid overconfidence:
- **Temperature Scaling:**
  $$\hat{P}_i = \frac{\exp(z_i / T)}{\sum_j \exp(z_j / T)}$$
  Where scalar parameter $T > 0$ is optimized post-training via Negative Log-Likelihood minimization.
- **Metric Requirement:** Expected Calibration Error (ECE) must remain $\le 0.05$.

---

## 4. Hardware & Infrastructure Strategy

The development cycle is split strictly between local authoring/runtime execution and cloud-accelerated training.

### 4.1 Local Development: Asus Vivobook 15
- **Typical configuration:** Intel Core i3/i5/i7 (11th–13th Gen) or AMD Ryzen 5/7, 8–16 GB RAM, Integrated Graphics (Iris Xe / Radeon).
- **Responsibilities:**
  - Code development, repository management, unit testing.
  - Mock dataset validation and pipeline orchestration.
  - Final artifact benchmarking, local API serving, and ONNX Runtime CPU inference.
- **Resource Constraints:**
  - Peak memory allocated for inference: $\le 1.5\text{ GB}$.
  - Peak inference thread count: 4 CPU threads.

### 4.2 Training Platform: Kaggle Notebooks
- **Hardware:** NVIDIA Tesla T4 Dual (2x 16 GB VRAM) or NVIDIA P100 (16 GB VRAM) with 30 weekly free compute hours.
- **Responsibilities:**
  - Tokenization of large-scale synthetic and structured datasets.
  - Model fine-tuning with mixed-precision (fp16 or bf16).
  - Multi-task loss backpropagation and parameter optimization.
  - Temperature calibration fitting.
  - Exporting trained weights to ONNX / TorchScript formats.
- **Kaggle Execution Constraints:**
  - Max interactive session: 12 hours.
  - Checkpoint strategy: Save state dictionaries to Kaggle working storage every 500 steps; push final weights directly to Hugging Face Model Hub (private) or Kaggle Dataset outputs.

---

## 5. Dataset Architecture & Training Objective

### 5.1 Dataset Schema
All data points must be converted into a multi-task tabular format:

```json
{
  "task_type": "choice",
  "instruction": "Route the user query to the appropriate microservice.",
  "context": "My order #8291 never arrived and I need a refund immediately.",
  "candidates": ["billing_service", "logistics_tracking", "general_inquiry", "escalations"],
  "target": "escalations"
}
```

### 5.2 Multi-Task Loss Formulation
During Kaggle training, mini-batches will be partitioned or masked across tasks. The total objective function is:

$$\mathcal{L}_{\text{total}} = \lambda_1 \mathcal{L}_{\text{Choice}} + \lambda_2 \mathcal{L}_{\text{Boolean}} + \lambda_3 \mathcal{L}_{\text{Score}}$$

- $\mathcal{L}_{\text{Choice}}$: Cross-Entropy loss over dynamically pooled candidate scores.
- $\mathcal{L}_{\text{Boolean}}$: Binary Cross-Entropy (`BCEWithLogitsLoss`).
- $\mathcal{L}_{\text{Score}}$: Mean Squared Error (MSE) or Huber Loss against rubric ground truths.

---

## 6. Technical Stack & Dependencies

| Layer | Component | Description / Tool |
| :--- | :--- | :--- |
| **Base Architecture** | Encoder Backbone | `modernbert-base` (149M params) or `deberta-v3-small` (44M params) |
| **Framework** | Deep Learning Core | PyTorch 2.x, Hugging Face `transformers`, `accelerate` |
| **Training Execution** | Cloud Notebooks | Kaggle GPU Environment (Python 3.10+, CUDA 12.x) |
| **Export Format** | Optimized Inference | ONNX Runtime (CPU-optimized, AVX-512 enabled) |
| **Local Runtime** | API Server | FastAPI, Pydantic v2, Uvicorn |
| **Calibration** | Metrics & Fitting | Scikit-learn, NetCal (`netcal.metrics.ECE`) |

---

## 7. Non-Functional Requirements (NFRs)

### 7.1 Performance & Latency Budgets
- **Local Latency (Vivobook 15 CPU):**
  - ModernBERT Backbone (ONNX FP32): $\le 45\text{ ms}$ per request.
  - ModernBERT Backbone (ONNX INT8 Quantized): $\le 25\text{ ms}$ per request.
- **Throughput:** $\ge 30\text{ queries/second}$ on local multi-threaded CPU.
- **Cold Start Time:** $\le 1.5\text{ seconds}$ to initialize weights in memory.

### 7.2 Reliability & Determinism
- **Zero Hallucination:** 100% adherence to defined schema keys. The model cannot output choices outside the provided `candidates` array.
- **Determinism:** Greedy decision paths must yield identical outputs across repeated calls on identical context ($T = 0$ equivalent behavior).

---

## 8. Implementation Roadmap

### Phase 1: Local Scaffolding & Dynamic Head Design (Vivobook 15)
- [ ] Define the PyTorch model module with dynamic dot-product candidate projection.
- [ ] Build synthetic multi-task mock data generator (500 samples) for unit testing forward passes.
- [ ] Verify error-free CPU execution on local Python environment.

### Phase 2: Kaggle Pipeline & Model Training (Kaggle T4)
- [ ] Set up Kaggle Notebook connected to Hugging Face API secrets.
- [ ] Assemble training corpus (intent routing, classification benchmarks, safety flags).
- [ ] Execute fine-tuning loop for 5 epochs with AdamW ($lr = 3\times 10^{-5}$) using mixed precision.
- [ ] Run validation step and save temperature calibration vectors.

### Phase 3: ONNX Conversion & Local Optimization (Kaggle -> Vivobook 15)
- [ ] Trace the trained PyTorch module to ONNX format with dynamic batch and sequence axes.
- [ ] Perform static post-training INT8 quantization.
- [ ] Download `.onnx` model weights to local Vivobook 15.

### Phase 4: Local Microservice & Validation (Vivobook 15)
- [ ] Wrap the ONNX runtime in a FastAPI microservice using Pydantic schemas.
- [ ] Benchmark latency, CPU core usage, and memory footprints.
- [ ] Validate ECE and classification accuracy against validation split.

---

## 9. Risk Matrix & Mitigations

| Risk | Impact | Likelihood | Mitigation Strategy |
| :--- | :--- | :--- | :--- |
| **Kaggle 12-hour session timeout** | High | Low | Save checkpoints to persistent Kaggle dataset storage every 500 steps. |
| **Dynamic candidate count out-of-memory** | Medium | Medium | Cap candidate arrays to $k \le 64$ options during inference; pad dynamic matrices during training. |
| **CPU thermal throttling on Vivobook 15** | Medium | Medium | Export to ONNX INT8 to reduce arithmetic intensity and core heat generation. |
| **Overconfident uncalibrated predictions** | High | High | Enforce automated temperature optimization step in validation pipeline before weight release. |
