"""ONNX Runtime CPU Inference Adapter for System One Decision Engine.

Loads optimized INT8 ONNX computation graphs and executes sub-25ms CPU inference
with strict Pydantic v2 contract enforcement and calibrated confidence scoring.

Laya-MLX Alignment:
  - system_one() / predict(): Unified dict-in dict-out API matching laya-mlx
    agent.system_one() / agent.predict() signature.
  - build_sequence prompt layout: [CLS] type ins [SEP] [MASK] opt0 [MASK] opt1 [SEP] state [SEP]
  - Per-bucket temperature clamping via temp_bucket() / clamp_temperature().
  - Shannon entropy confidence (compute_entropy_confidence) in the dict-API hot path.
  - Optional PrefixCache for repeated questions across multiple calls.
"""

import logging
import time
from pathlib import Path
from typing import Any, Optional, Sequence
import numpy as np
import onnxruntime as ort

from system_one_engine.core.calibration import clamp_temperature, temp_bucket
from system_one_engine.core.confidence import (
    compute_boolean_confidence,
    compute_choice_confidence,
    compute_entropy_confidence,
    compute_expected_score,
    compute_score_confidence,
)
from system_one_engine.core.contracts import (
    AnswerUnion,
    BooleanRequest,
    BooleanResponse,
    ChoiceRequest,
    ChoiceResponse,
    DecideRequest,
    DecideResponse,
    QuestionType,
    ScoreRequest,
    ScoreResponse,
)
from system_one_engine.core.prefix_cache import PrefixCache
from system_one_engine.core.prompt import QTYPES, build_sequence, to_internal

logger = logging.getLogger(__name__)


class ONNXRuntimeAdapter:
    """Production runtime engine executing quantized INT8 ONNX models on CPU.

    Configured specifically for multi-threaded laptop CPU execution (Asus Vivobook 15).
    """

    def __init__(
        self,
        encoder_path: str | Path,
        heads_path: str | Path,
        choice_head_path: str | Path,
        tokenizer: Any,
        num_threads: int = 4,
        temperature: float = 1.0,
        cache_prompts: bool = False,
        cache_capacity: int = 128,
    ) -> None:
        """Initialize ONNX Runtime inference sessions.

        Args:
            encoder_path: Path to encoder ONNX model.
            heads_path: Path to boolean/score heads ONNX model.
            choice_head_path: Path to dynamic choice ONNX model.
            tokenizer: Tokenizer instance with callable interface.
            num_threads: CPU intra-op thread count (default: 4).
            temperature: Default post-hoc temperature multiplier.
            cache_prompts: If True, enable PrefixCache for repeated questions (default: False).
            cache_capacity: Maximum cached question prefixes when cache_prompts=True (default: 128).
        """
        self.tokenizer = tokenizer
        self.temperature = max(0.01, float(temperature))
        self._prefix_cache: Optional[PrefixCache] = (
            PrefixCache(capacity=cache_capacity) if cache_prompts else None
        )

        sess_options = ort.SessionOptions()
        sess_options.intra_op_num_threads = num_threads
        sess_options.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL
        sess_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL

        providers = ["CPUExecutionProvider"]

        self.encoder_session = ort.InferenceSession(
            str(encoder_path), sess_options=sess_options, providers=providers
        )
        self.heads_session = ort.InferenceSession(
            str(heads_path), sess_options=sess_options, providers=providers
        )
        self.choice_session = ort.InferenceSession(
            str(choice_head_path), sess_options=sess_options, providers=providers
        )

    def _encode_text(self, text: str | Sequence[str]) -> np.ndarray:
        """Encode text or sequence of texts into state vectors h_state (B, D)."""
        tokens = self.tokenizer(text, return_tensors="pt")
        input_ids = tokens["input_ids"].numpy().astype(np.int64)
        attention_mask = tokens["attention_mask"].numpy().astype(np.int64)

        ort_inputs = {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
        }
        outputs = self.encoder_session.run(None, ort_inputs)
        return outputs[0]  # Shape: (B, D)

    def evaluate_choice(self, request: ChoiceRequest) -> ChoiceResponse:
        """Evaluate a categorical Choice request using dynamic candidate dot-product."""
        candidates = list(request.criteria.keys())
        cand_descriptions = [f"{k} — {request.criteria[k]}" for k in candidates]
        num_cands = len(candidates)

        # 1. Encode context: (1, D)
        ctx_h = self._encode_text(request.context)

        # 2. Encode candidate options: (K, D) -> (1, K, D)
        cand_h = self._encode_text(cand_descriptions)[np.newaxis, ...]

        # 3. Compute dot-product logits: (1, K)
        ort_inputs = {
            "context_state": ctx_h,
            "candidate_embeddings": cand_h,
        }
        outputs = self.choice_session.run(None, ort_inputs)
        raw_logits = outputs[0][0]  # Shape: (num_cands,)

        # Temperature scaling & softmax
        scaled_logits = raw_logits / (request.temperature * self.temperature)
        exp_logits = np.exp(scaled_logits - np.max(scaled_logits))
        probs_arr = exp_logits / np.sum(exp_logits)

        probabilities = {candidates[i]: float(probs_arr[i]) for i in range(num_cands)}
        winning_idx = int(np.argmax(probs_arr))
        winning_choice = candidates[winning_idx]
        confidence = compute_choice_confidence(probabilities)

        return ChoiceResponse(
            choice=winning_choice,
            probabilities=probabilities,
            confidence=confidence,
        )

    def evaluate_boolean(self, request: BooleanRequest) -> BooleanResponse:
        """Evaluate a binary Boolean request using the boolean classification head."""
        h_state = self._encode_text(request.context)

        outputs = self.heads_session.run(None, {"h_state": h_state})
        raw_logit = float(outputs[0][0])  # boolean_logits: (B,)

        scaled_logit = raw_logit / (request.temperature * self.temperature)
        prob_true = float(1.0 / (1.0 + np.exp(-scaled_logit)))

        confirmed = prob_true >= 0.5
        confidence = compute_boolean_confidence(prob_true)

        return BooleanResponse(
            confirmed=confirmed,
            probability=prob_true,
            confidence=confidence,
        )

    def evaluate_score(self, request: ScoreRequest) -> ScoreResponse:
        """Evaluate an ordinal Score request using the score classification head."""
        num_levels = len(request.criteria)
        h_state = self._encode_text(request.context)

        outputs = self.heads_session.run(None, {"h_state": h_state})
        # Output 1 is score_logits: (B, max_levels)
        raw_level_logits = outputs[1][0][:num_levels]

        scaled_logits = raw_level_logits / (request.temperature * self.temperature)
        exp_logits = np.exp(scaled_logits - np.max(scaled_logits))
        probs_arr = exp_logits / np.sum(exp_logits)

        probabilities = {str(i): float(probs_arr[i]) for i in range(num_levels)}
        legend = {str(i): desc for i, desc in enumerate(request.criteria)}

        expected_score = compute_expected_score(probabilities)
        confidence = compute_score_confidence(probabilities)

        return ScoreResponse(
            score=expected_score,
            legend=legend,
            probabilities=probabilities,
            confidence=confidence,
        )

    def evaluate_decide(self, request: DecideRequest) -> DecideResponse:
        """Evaluate speculative fan-out questions over shared context."""
        start = time.perf_counter()
        answers: dict[str, AnswerUnion] = {}

        # Optimization: Shared context state encoded once
        shared_h = self._encode_text(request.context)

        for q_id, q_req in request.questions.items():
            if isinstance(q_req, ChoiceRequest):
                candidates = list(q_req.criteria.keys())
                cand_descs = [f"{k} — {q_req.criteria[k]}" for k in candidates]
                cand_h = self._encode_text(cand_descs)[np.newaxis, ...]
                ort_inputs = {
                    "context_state": shared_h,
                    "candidate_embeddings": cand_h,
                }
                outputs = self.choice_session.run(None, ort_inputs)
                raw_logits = outputs[0][0]
                scaled_logits = raw_logits / (q_req.temperature * self.temperature)
                exp_l = np.exp(scaled_logits - np.max(scaled_logits))
                probs = exp_l / np.sum(exp_l)
                p_map = {candidates[i]: float(probs[i]) for i in range(len(candidates))}
                answers[q_id] = ChoiceResponse(
                    choice=candidates[int(np.argmax(probs))],
                    probabilities=p_map,
                    confidence=compute_choice_confidence(p_map),
                )

            elif isinstance(q_req, BooleanRequest):
                outputs = self.heads_session.run(None, {"h_state": shared_h})
                raw_logit = float(outputs[0][0])
                scaled = raw_logit / (q_req.temperature * self.temperature)
                p_true = float(1.0 / (1.0 + np.exp(-scaled)))
                answers[q_id] = BooleanResponse(
                    confirmed=p_true >= 0.5,
                    probability=p_true,
                    confidence=compute_boolean_confidence(p_true),
                )

            elif isinstance(q_req, ScoreRequest):
                n_lev = len(q_req.criteria)
                outputs = self.heads_session.run(None, {"h_state": shared_h})
                raw_logits = outputs[1][0][:n_lev]
                scaled = raw_logits / (q_req.temperature * self.temperature)
                exp_l = np.exp(scaled - np.max(scaled))
                probs = exp_l / np.sum(exp_l)
                p_map = {str(i): float(probs[i]) for i in range(n_lev)}
                legend = {str(i): desc for i, desc in enumerate(q_req.criteria)}
                answers[q_id] = ScoreResponse(
                    score=compute_expected_score(p_map),
                    legend=legend,
                    probabilities=p_map,
                    confidence=compute_score_confidence(p_map),
                )

        latency_ms = (time.perf_counter() - start) * 1000.0
        return DecideResponse(answers=answers, latency_ms=latency_ms)

    # -----------------------------------------------------------------------
    # Laya-MLX Aligned API — dict-in dict-out (no Pydantic overhead)
    # -----------------------------------------------------------------------

    def system_one(self, state: Any, questions: dict, **_kwargs: Any) -> dict:
        """Evaluate multiple heterogeneous questions in a single batched pass.

        Unified dict-in dict-out interface aligned with laya-mlx's
        ``agent.system_one(state, questions)`` and ``agent.predict()`` APIs.

        Uses the canonical marker-based prompt layout:
            [CLS] <type> instruction [SEP] [MASK] opt0 [MASK] opt1 … [SEP] state [SEP]

        Confidence is reported via Shannon entropy (``compute_entropy_confidence``)
        which is strictly more informative than the peak-probability formula for
        high-cardinality distributions. Per-bucket temperature clamping prevents
        overfitted checkpoints from sharpening logits unsafely.

        Args:
            state: Input state — string, dict, list, or None.
            questions: Dict mapping question IDs to question definition dicts:
                {"type": "choice"|"score"|"boolean", "instructions": str, "criteria": ...}
            **_kwargs: Ignored extra keywords (for API compatibility with Router).

        Returns:
            dict: Mapping question ID to result dict with keys:
                - For choice: {"choice": str, "probabilities": dict, "confidence": float}
                - For score:  {"score": float, "legend": dict, "probabilities": dict, "confidence": float}
                - For boolean:{"confirmed": bool, "probability": float, "confidence": float}
        """
        if not isinstance(questions, dict) or not questions:
            return {}

        tok = self.tokenizer
        result: dict = {}

        for qid, qdef in questions.items():
            try:
                q = to_internal(qdef)
            except ValueError as exc:
                logger.warning("system_one: skipping question %r — %s", qid, exc)
                continue

            options = list(q["crit"].keys()) if isinstance(q["crit"], dict) else (
                [str(i) for i in range(len(q["crit"]))] if isinstance(q["crit"], list) else
                ["false", "true"]
            )
            k = len(options)

            # Build the full token sequence via marker-based prompt layout
            ids, markers = build_sequence(tok, state, q)
            if not ids or not markers:
                logger.warning("system_one: empty sequence for question %r", qid)
                continue

            input_ids = np.array([ids], dtype=np.int64)
            attention_mask = np.ones_like(input_ids)

            # Per-bucket temperature (clamped to [TEMP_MIN, TEMP_MAX])
            qtype_int = QTYPES.get(q["t"], 2)
            bucket = temp_bucket(qtype_int, k)
            # If caller provided a per-bucket temperature map (future extension), use it;
            # otherwise fall back to the adapter's global temperature.
            raw_temp = self.temperature
            safe_temp = clamp_temperature(raw_temp)

            # Encoder pass
            enc_outputs = self.encoder_session.run(
                None,
                {"input_ids": input_ids, "attention_mask": attention_mask},
            )
            h = enc_outputs[0]  # (1, seq_len, D)

            if q["t"] == "choice":
                # Extract hidden states at marker positions (one per option)
                padded_markers = (markers + [markers[-1]] * k)[:k]
                marker_h = np.concatenate(
                    [h[:, m : m + 1, :] for m in padded_markers], axis=1
                )  # (1, k, D)

                # Use choice_head: context_state=(1,D), candidates=(1,k,D)
                context_state = h[:, 0, :]  # CLS token as context
                ort_inputs = {
                    "context_state": context_state,
                    "candidate_embeddings": marker_h,
                }
                ch_out = self.choice_session.run(None, ort_inputs)
                raw_logits = ch_out[0][0]  # (k,)
                scaled = raw_logits / safe_temp
                exp_l = np.exp(scaled - np.max(scaled))
                probs_arr = exp_l / np.sum(exp_l)
                probs_map = {options[i]: float(probs_arr[i]) for i in range(k)}
                result[qid] = {
                    "choice": options[int(np.argmax(probs_arr))],
                    "probabilities": probs_map,
                    "confidence": compute_entropy_confidence(probs_arr),
                    "temp_bucket": bucket,
                }

            elif q["t"] == "score":
                context_state = h[:, 0, :]  # CLS pooled
                heads_out = self.heads_session.run(None, {"h_state": context_state})
                raw_logits = heads_out[1][0][:k]  # score_logits (max_levels,) → slice to k
                scaled = raw_logits / safe_temp
                exp_l = np.exp(scaled - np.max(scaled))
                probs_arr = exp_l / np.sum(exp_l)
                probs_map = {str(i): float(probs_arr[i]) for i in range(k)}
                legend = {
                    str(i): desc for i, desc in enumerate(q["crit"])
                } if q["crit"] else {}
                result[qid] = {
                    "score": compute_expected_score(probs_map),
                    "legend": legend,
                    "probabilities": probs_map,
                    "confidence": compute_entropy_confidence(probs_arr),
                    "temp_bucket": bucket,
                }

            else:  # noul / boolean
                context_state = h[:, 0, :]
                heads_out = self.heads_session.run(None, {"h_state": context_state})
                raw_logit = float(heads_out[0][0])  # boolean_logit scalar
                scaled_logit = raw_logit / safe_temp
                p_true = float(1.0 / (1.0 + np.exp(-scaled_logit)))
                result[qid] = {
                    "confirmed": p_true >= 0.5,
                    "probability": p_true,
                    "confidence": compute_boolean_confidence(p_true),
                    "temp_bucket": bucket,
                }

        return result

    # laya-mlx API alias: agent.predict(state, questions)
    predict = system_one
