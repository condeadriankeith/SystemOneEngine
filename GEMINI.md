# SystemOneEngine - Workspace Rules & Agent Memory

## Core Directives & Memory

### 1. Mandatory Documentation Protocol (CRITICAL RULE)
For every phase, action, and task undertaken in this project:
- **Every Single Step Must Be Documented:** Any file added, edited, updated, improved, refactored, or removed must be recorded in real-time in [`docs/DEV_LOG.md`](file:///c:/Users/conde/OneDrive/Desktop/SystemOneEngine/docs/DEV_LOG.md).
- **Rationale & Impact:** Documentation must capture not just *what* was changed, but *why* it was changed, architectural trade-offs, and empirical results.
- **Verification Records:** Every test, CPU latency benchmark, ECE calibration result, and validation run must be logged with timestamped output summaries.

---

## 2. Architectural Guardrails & Standards
- **Readability & Order > Speed & Complex Interconnections**: Keep components decoupled, explicit, and self-documenting.
- **Type Safety**: Strictly typed schemas using Pydantic v2 and Python type annotations.
- **No Placeholders**: Never leave stubs, `// TODO`, or incomplete logic.
- **Calibration Guarantee**: Expected Calibration Error (ECE) must strictly adhere to $\le 0.05$.
- **Performance Budget**: Under 35 ms inference on local CPU (Vivobook 15) using ONNX INT8 quantization.
