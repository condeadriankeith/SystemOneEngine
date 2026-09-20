"""Mathematical confidence formulations for System One decision primitives.

Implements mathematically grounded metrics for:
- Choice Confidence: Peak probability lead over uniform random chance.
- Score Confidence: Mean Absolute Deviation (MAD) from modal index relative to uniform MAD.
- Boolean Confidence: Certainty divergence from maximal ambiguity (|P - 0.5| * 2).
- Expected Score: Probability-weighted continuous scalar across ordinal ladder levels.
"""

from typing import Sequence


def compute_choice_confidence(probabilities: Sequence[float] | dict[str, float]) -> float:
    """Compute the normalized Choice Confidence metric for an unordered categorical distribution.

    Formula:
        Confidence = (max(P) - 1/n) / (1 - 1/n)

    Behavior:
        - Uniform distribution: max(P) = 1/n -> 0.0
        - One-hot certainty: max(P) = 1.0 -> 1.0
        - Scaled proportionally to option count n.

    Args:
        probabilities: Sequence of probabilities or dictionary mapping candidate keys to probabilities.

    Returns:
        float: Normalized confidence in range [0.0, 1.0].
    """
    if isinstance(probabilities, dict):
        prob_values = list(probabilities.values())
    else:
        prob_values = list(probabilities)

    n = len(prob_values)
    if n <= 1:
        return 1.0

    peak_prob = max(prob_values)
    uniform_prob = 1.0 / n

    if peak_prob <= uniform_prob:
        return 0.0

    raw_conf = (peak_prob - uniform_prob) / (1.0 - uniform_prob)
    return float(max(0.0, min(1.0, raw_conf)))


def compute_score_confidence(probabilities: Sequence[float] | dict[str, float]) -> float:
    """Compute the ordinal Score Confidence based on Mean Absolute Deviation (MAD) around the mode.

    Because score levels possess an intrinsic geometric ordering (0 < 1 < ... < m-1),
    confidence must measure how tightly probability mass concentrates around the modal level,
    penalizing dispersed or bi-modal uncertainty.

    Steps:
        1. Identify candidate modes k* = argmax P(i). When tied, select the mode closest to center.
        2. Compute expected distance from mode: MAD_mode = sum_i P(i) * |i - k*|
        3. Compute MAD of uniform distribution centered at c = (m - 1) / 2:
           MAD_uniform = (1 / m) * sum_i |i - c|
        4. Normalize: Confidence = max(0.0, 1.0 - (MAD_mode / MAD_uniform))

    Args:
        probabilities: Sequence of level probabilities in ascending level order, or dictionary.

    Returns:
        float: Normalized score confidence in range [0.0, 1.0].
    """
    if isinstance(probabilities, dict):
        # Sort by key if keys are integer strings (e.g. '0', '1', '2')
        try:
            sorted_items = sorted(probabilities.items(), key=lambda item: int(item[0]))
            prob_values = [item[1] for item in sorted_items]
        except ValueError:
            prob_values = list(probabilities.values())
    else:
        prob_values = list(probabilities)

    m = len(prob_values)
    if m <= 1:
        return 1.0

    # Center of uniform distribution
    c = (m - 1) / 2.0

    # MAD of uniform distribution
    mad_uniform = sum(abs(i - c) for i in range(m)) / float(m)
    if mad_uniform == 0.0:
        return 1.0

    # Find all modal indices (handling ties)
    max_p = max(prob_values)
    modes = [idx for idx, p in enumerate(prob_values) if abs(p - max_p) < 1e-7]

    # Select the mode closest to the center c to avoid tie-breaking bias
    best_mode = min(modes, key=lambda idx: (abs(idx - c), idx))

    # Compute MAD around the chosen mode
    mad_mode = sum(prob * abs(i - best_mode) for i, prob in enumerate(prob_values))

    # Normalized ordinal concentration
    raw_confidence = 1.0 - (mad_mode / mad_uniform)
    return float(max(0.0, min(1.0, raw_confidence)))


def compute_expected_score(probabilities: Sequence[float] | dict[str, float]) -> float:
    """Compute the continuous expected score scalar as a probability-weighted expectation.

    Formula:
        Score = sum_{i=0}^{m-1} i * P(i)

    Args:
        probabilities: Sequence of level probabilities or dictionary of level index strings.

    Returns:
        float: Expected score in range [0.0, m-1].
    """
    if isinstance(probabilities, dict):
        try:
            sorted_items = sorted(probabilities.items(), key=lambda item: int(item[0]))
            prob_values = [item[1] for item in sorted_items]
        except ValueError:
            prob_values = list(probabilities.values())
    else:
        prob_values = list(probabilities)

    m = len(prob_values)
    if m == 0:
        return 0.0

    # Normalize if slightly drifting
    total_p = sum(prob_values)
    if total_p <= 0.0:
        return 0.0

    normalized_p = [p / total_p for p in prob_values]
    expected_val = sum(i * p for i, p in enumerate(normalized_p))
    return float(max(0.0, min(float(m - 1), expected_val)))


def compute_boolean_confidence(probability: float) -> float:
    """Compute Boolean confidence as distance from maximal uncertainty (P = 0.5).

    Formula:
        Confidence = |P - 0.5| * 2

    Args:
        probability: Calibrated probability of True/Affirmative in [0.0, 1.0].

    Returns:
        float: Decision certainty in range [0.0, 1.0].
    """
    clamped_p = max(0.0, min(1.0, float(probability)))
    return float(abs(clamped_p - 0.5) * 2.0)
