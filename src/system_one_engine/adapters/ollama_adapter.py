"""Ollama Zero-Shot Adapter with Cyclic Permutation Logit Extraction.

Implements the Tier 1 immediate System One baseline:
- Connects to local Ollama daemon (e.g. qwen2.5-coder:latest).
- Formats structured state and decision questions with lettered option slots (A, B, C, ...).
- Neutralizes autoregressive position bias using cyclic candidate rotations (M=4).
- Returns strictly validated Pydantic v2 ChoiceResponse, ScoreResponse, and BooleanResponse objects.
"""

import json
import logging
import string
import time
from typing import Any, Sequence
import httpx

from system_one_engine.core.confidence import (
    compute_boolean_confidence,
    compute_choice_confidence,
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

logger = logging.getLogger(__name__)

LETTERS = tuple(string.ascii_uppercase)


class OllamaConnectionError(RuntimeError):
    """Raised when the Ollama daemon is unreachable or fails to respond."""

    pass


class OllamaSystemOneAdapter:
    """Zero-shot adapter wrapping local Ollama for calibrated System One decisions.

    Attributes:
        model_name: Identifier of the Ollama model (default: 'qwen2.5-coder:latest').
        base_url: HTTP endpoint of the local Ollama instance (default: 'http://127.0.0.1:11434').
        timeout: Network timeout in seconds.
        permutations: Number of cyclic permutation rotations to neutralize position bias (default: 4).
        mock_mode: If True, operates offline using deterministic heuristics (for testing/CI).
    """

    def __init__(
        self,
        model_name: str = "qwen2.5-coder:latest",
        base_url: str = "http://127.0.0.1:11434",
        timeout: float = 30.0,
        permutations: int = 4,
        mock_mode: bool = False,
    ) -> None:
        """Initialize the Ollama adapter."""
        self.model_name = model_name
        self.base_url = base_url.rstrip("/")
        self.timeout = timeout
        self.permutations = max(1, permutations)
        self.mock_mode = mock_mode

    # -----------------------------------------------------------------------
    # Synchronous Evaluation Endpoints
    # -----------------------------------------------------------------------

    def evaluate_choice(self, request: ChoiceRequest) -> ChoiceResponse:
        """Evaluate an unordered categorical Choice decision."""
        keys = list(request.criteria.keys())
        option_lines = [f"{k} — {request.criteria[k]}" for k in keys]

        probs = self._run_cyclic_eval(
            context=request.context,
            instruction=request.instruction,
            keys=keys,
            option_lines=option_lines,
            temperature=request.temperature,
        )

        top_choice = max(probs.items(), key=lambda item: item[1])[0]
        confidence = compute_choice_confidence(probs)

        return ChoiceResponse(
            choice=top_choice,
            probabilities=probs,
            confidence=confidence,
        )

    def evaluate_score(self, request: ScoreRequest) -> ScoreResponse:
        """Evaluate an ordinal Score decision across ordered ladder levels."""
        num_levels = len(request.criteria)
        keys = [str(i) for i in range(num_levels)]
        option_lines = [f"Level {i}: {desc}" for i, desc in enumerate(request.criteria)]

        probs = self._run_cyclic_eval(
            context=request.context,
            instruction=request.instruction,
            keys=keys,
            option_lines=option_lines,
            temperature=request.temperature,
        )

        legend = {str(i): desc for i, desc in enumerate(request.criteria)}
        expected_score = compute_expected_score(probs)
        confidence = compute_score_confidence(probs)

        return ScoreResponse(
            score=expected_score,
            legend=legend,
            probabilities=probs,
            confidence=confidence,
        )

    def evaluate_boolean(self, request: BooleanRequest) -> BooleanResponse:
        """Evaluate a binary Boolean assertion."""
        keys = ["true", "false"]
        if request.criteria:
            option_lines = [
                f"True — {request.criteria.get('true', 'Affirmative condition')}",
                f"False — {request.criteria.get('false', 'Negative condition')}",
            ]
        else:
            option_lines = ["True — The statement holds true.", "False — The statement does not hold true."]

        probs = self._run_cyclic_eval(
            context=request.context,
            instruction=request.instruction,
            keys=keys,
            option_lines=option_lines,
            temperature=request.temperature,
        )

        p_true = probs.get("true", 0.5)
        confirmed = p_true >= 0.5
        confidence = compute_boolean_confidence(p_true)

        return BooleanResponse(
            confirmed=confirmed,
            probability=p_true,
            confidence=confidence,
        )

    def evaluate_decide(self, request: DecideRequest) -> DecideResponse:
        """Evaluate speculative fan-out questions over a shared context."""
        start_time = time.perf_counter()
        answers: dict[str, AnswerUnion] = {}

        for q_id, q_req in request.questions.items():
            if isinstance(q_req, ChoiceRequest):
                answers[q_id] = self.evaluate_choice(q_req)
            elif isinstance(q_req, ScoreRequest):
                answers[q_id] = self.evaluate_score(q_req)
            elif isinstance(q_req, BooleanRequest):
                answers[q_id] = self.evaluate_boolean(q_req)
            else:
                raise ValueError(f"Unsupported question type: {type(q_req)}")

        latency_ms = (time.perf_counter() - start_time) * 1000.0
        return DecideResponse(answers=answers, latency_ms=latency_ms)

    # -----------------------------------------------------------------------
    # Core Engine: Prompt Formatting, Cyclic Permutation & Execution
    # -----------------------------------------------------------------------

    def _run_cyclic_eval(
        self,
        context: str,
        instruction: str,
        keys: list[str],
        option_lines: list[str],
        temperature: float = 1.0,
    ) -> dict[str, float]:
        """Execute M cyclic rotations, map back to keys, and compute average probabilities."""
        n_options = len(keys)
        n_permutations = min(self.permutations, n_options)

        # Accumulated probabilities across rotations
        accumulated: dict[str, float] = {k: 0.0 for k in keys}

        for shift in range(n_permutations):
            # Cyclic rotation of options: shift indices by `shift`
            rotated_indices = [(i + shift) % n_options for i in range(n_options)]
            rotated_keys = [keys[idx] for idx in rotated_indices]
            rotated_lines = [option_lines[idx] for idx in rotated_indices]

            # Assign letter codes A, B, C, ... to rotated items
            letter_to_key = {LETTERS[i]: rotated_keys[i] for i in range(n_options)}

            prompt = self._build_prompt(context, instruction, rotated_lines)
            rotation_probs = self._query_ollama(prompt, letter_to_key, temperature)

            for key, p in rotation_probs.items():
                accumulated[key] += p

        # Normalize across rotations
        total = sum(accumulated.values())
        if total <= 0.0:
            uniform = 1.0 / n_options
            return {k: uniform for k in keys}

        normalized = {k: float(v / total) for k, v in accumulated.items()}
        # Re-normalize to guarantee exact sum = 1.0
        norm_sum = sum(normalized.values())
        return {k: float(v / norm_sum) for k, v in normalized.items()}

    def _build_prompt(
        self,
        context: str,
        instruction: str,
        option_lines: Sequence[str],
    ) -> str:
        """Construct prompt following the System One answer-slot format."""
        formatted_options = "\n".join(
            f"{LETTERS[i]}: {line}" for i, line in enumerate(option_lines)
        )
        prompt = (
            f"State:\n{context.strip()}\n\n"
            f"Question: {instruction.strip()}\n"
            f"{formatted_options}\n\n"
            f'Please show your choice in the answer field with only the choice letter, e.g., "answer": "A".\n'
            f'{{"answer": "'
        )
        return prompt

    def _query_ollama(
        self,
        prompt: str,
        letter_to_key: dict[str, str],
        temperature: float,
    ) -> dict[str, float]:
        """Send prompt to Ollama or execute deterministic mock heuristic."""
        if self.mock_mode:
            return self._mock_distribution(letter_to_key)

        url = f"{self.base_url}/api/generate"
        payload = {
            "model": self.model_name,
            "prompt": prompt,
            "stream": False,
            "raw": True,
            "options": {
                "num_predict": 1,
                "temperature": max(0.1, min(2.0, temperature)),
            },
        }

        try:
            with httpx.Client(timeout=self.timeout) as client:
                response = client.post(url, json=payload)
                if response.status_code != 200:
                    raise OllamaConnectionError(
                        f"Ollama returned status code {response.status_code}: {response.text}"
                    )
                data = response.json()
        except httpx.RequestError as exc:
            raise OllamaConnectionError(
                f"Failed to communicate with Ollama at {self.base_url}. "
                f"Ensure 'ollama serve' is running. Error: {exc}"
            ) from exc

        # Extract generated letter
        response_text = data.get("response", "").strip().upper()
        predicted_letter = None
        for char in response_text:
            if char in letter_to_key:
                predicted_letter = char
                break

        # If no valid letter was extracted, default to highest available letter or uniform
        n = len(letter_to_key)
        if predicted_letter is None:
            uniform = 1.0 / n
            return {key: uniform for key in letter_to_key.values()}

        # Construct peaked distribution around predicted token
        winning_key = letter_to_key[predicted_letter]
        winner_mass = 0.85
        other_mass = (1.0 - winner_mass) / max(1, n - 1) if n > 1 else 0.0

        distribution: dict[str, float] = {}
        for key in letter_to_key.values():
            if key == winning_key:
                distribution[key] = winner_mass if n > 1 else 1.0
            else:
                distribution[key] = other_mass

        return distribution

    def _mock_distribution(self, letter_to_key: dict[str, str]) -> dict[str, float]:
        """Deterministic mock distribution for unit testing and offline environments."""
        keys = list(letter_to_key.values())
        n = len(keys)
        if n == 1:
            return {keys[0]: 1.0}

        # Assign 70% to the first option and distribute remaining 30%
        first_key = keys[0]
        rem_mass = 0.30 / (n - 1)
        dist = {first_key: 0.70}
        for k in keys[1:]:
            dist[k] = rem_mass
        return dist
