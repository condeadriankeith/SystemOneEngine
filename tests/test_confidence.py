"""Unit tests for mathematical confidence and expectation routines."""

import math
import pytest
from system_one_engine.core.confidence import (
    compute_boolean_confidence,
    compute_choice_confidence,
    compute_entropy_confidence,
    compute_expected_score,
    compute_score_confidence,
)


# ---------------------------------------------------------------------------
# Choice Confidence Tests
# ---------------------------------------------------------------------------


def test_choice_confidence_uniform():
    """Verify uniform distribution produces exactly 0.0 confidence."""
    # 2 choices
    assert compute_choice_confidence([0.5, 0.5]) == pytest.approx(0.0, abs=1e-5)
    # 4 choices
    assert compute_choice_confidence([0.25, 0.25, 0.25, 0.25]) == pytest.approx(0.0, abs=1e-5)
    # 10 choices
    assert compute_choice_confidence([0.1] * 10) == pytest.approx(0.0, abs=1e-5)


def test_choice_confidence_one_hot():
    """Verify one-hot distribution produces 1.0 confidence."""
    assert compute_choice_confidence([1.0, 0.0]) == pytest.approx(1.0, abs=1e-5)
    assert compute_choice_confidence([0.0, 1.0, 0.0, 0.0]) == pytest.approx(1.0, abs=1e-5)


def test_choice_confidence_scaling_with_n():
    """Verify choice confidence scales proportionally to the number of options n."""
    # 2 choices with P = 0.55: (0.55 - 0.50) / (1 - 0.50) = 0.05 / 0.50 = 0.10
    conf_2 = compute_choice_confidence([0.55, 0.45])
    assert conf_2 == pytest.approx(0.10, abs=1e-4)

    # 10 choices with P = 0.55: (0.55 - 0.10) / (1 - 0.10) = 0.45 / 0.90 = 0.50
    probs_10 = [0.55] + [0.05] * 9
    conf_10 = compute_choice_confidence(probs_10)
    assert conf_10 == pytest.approx(0.50, abs=1e-4)


def test_choice_confidence_dict_input():
    """Verify dictionary input works identically to sequence input."""
    data = {"option_a": 0.8, "option_b": 0.15, "option_c": 0.05}
    # n = 3, peak = 0.8, uniform = 1/3: (0.8 - 0.3333) / (1 - 0.3333) = 0.4667 / 0.6667 = 0.70
    conf = compute_choice_confidence(data)
    assert conf == pytest.approx(0.70, abs=1e-2)


# ---------------------------------------------------------------------------
# Score Confidence Tests
# ---------------------------------------------------------------------------


def test_score_confidence_one_hot():
    """Verify one-hot score distribution produces 1.0 confidence."""
    # Peak at level 0
    assert compute_score_confidence([1.0, 0.0, 0.0]) == pytest.approx(1.0, abs=1e-5)
    # Peak at level 1
    assert compute_score_confidence([0.0, 1.0, 0.0, 0.0]) == pytest.approx(1.0, abs=1e-5)


def test_score_confidence_uniform():
    """Verify uniform distribution produces 0.0 confidence."""
    # 3 levels: c = 1.0. MAD_mode = 2/3, MAD_uniform = 2/3 -> 1 - 1 = 0
    assert compute_score_confidence([1 / 3, 1 / 3, 1 / 3]) == pytest.approx(0.0, abs=1e-5)
    # 5 levels
    assert compute_score_confidence([0.2, 0.2, 0.2, 0.2, 0.2]) == pytest.approx(0.0, abs=1e-5)


def test_score_confidence_ordinal_proximity():
    """Verify probability clustered in adjacent levels has higher confidence than distant bi-modal."""
    # 4 levels: 0, 1, 2, 3
    # Adjacent levels (1 and 2)
    adjacent = [0.0, 0.5, 0.5, 0.0]
    # Distant bimodal extremes (0 and 3)
    distant = [0.5, 0.0, 0.0, 0.5]

    conf_adj = compute_score_confidence(adjacent)
    conf_dist = compute_score_confidence(distant)

    assert conf_adj > conf_dist
    assert conf_dist == pytest.approx(0.0, abs=1e-3)


# ---------------------------------------------------------------------------
# Expected Score Tests
# ---------------------------------------------------------------------------


def test_expected_score_discrete():
    """Verify expected score on discrete one-hot levels."""
    assert compute_expected_score([1.0, 0.0, 0.0]) == pytest.approx(0.0, abs=1e-5)
    assert compute_expected_score([0.0, 1.0, 0.0]) == pytest.approx(1.0, abs=1e-5)
    assert compute_expected_score([0.0, 0.0, 1.0]) == pytest.approx(2.0, abs=1e-5)


def test_expected_score_interpolation():
    """Verify smooth interpolation between levels."""
    # Split between 0 and 1 equally -> 0.5
    assert compute_expected_score([0.5, 0.5, 0.0]) == pytest.approx(0.5, abs=1e-5)
    # Split between 1 and 2 equally -> 1.5
    assert compute_expected_score([0.0, 0.5, 0.5]) == pytest.approx(1.5, abs=1e-5)
    # 3-level dictionary
    data = {"0": 0.2, "1": 0.3, "2": 0.5}
    # 0*0.2 + 1*0.3 + 2*0.5 = 0.3 + 1.0 = 1.3
    assert compute_expected_score(data) == pytest.approx(1.3, abs=1e-5)


# ---------------------------------------------------------------------------
# Boolean Confidence Tests
# ---------------------------------------------------------------------------


def test_boolean_confidence():
    """Verify Boolean confidence behavior."""
    # Complete ambiguity: P = 0.5 -> confidence = 0.0
    assert compute_boolean_confidence(0.5) == pytest.approx(0.0, abs=1e-5)
    # Absolute certainty
    assert compute_boolean_confidence(1.0) == pytest.approx(1.0, abs=1e-5)
    assert compute_boolean_confidence(0.0) == pytest.approx(1.0, abs=1e-5)
    # P = 0.85 -> |0.85 - 0.5| * 2 = 0.70
    assert compute_boolean_confidence(0.85) == pytest.approx(0.70, abs=1e-5)


# ---------------------------------------------------------------------------
# Entropy Confidence Tests (laya-mlx alignment)
# ---------------------------------------------------------------------------


def test_entropy_confidence_uniform_is_zero():
    """Uniform distribution has maximum entropy -> confidence = 0.0."""
    assert compute_entropy_confidence([0.5, 0.5]) == pytest.approx(0.0, abs=1e-6)
    assert compute_entropy_confidence([0.25, 0.25, 0.25, 0.25]) == pytest.approx(0.0, abs=1e-6)
    assert compute_entropy_confidence([0.1] * 10) == pytest.approx(0.0, abs=1e-6)


def test_entropy_confidence_one_hot_is_one():
    """One-hot distribution has zero entropy -> confidence = 1.0."""
    assert compute_entropy_confidence([1.0, 0.0]) == pytest.approx(1.0, abs=1e-6)
    assert compute_entropy_confidence([0.0, 1.0, 0.0, 0.0]) == pytest.approx(1.0, abs=1e-6)


def test_entropy_confidence_single_element():
    """Single-element distribution is trivially certain."""
    assert compute_entropy_confidence([1.0]) == pytest.approx(1.0, abs=1e-6)


def test_entropy_confidence_monotone():
    """More concentrated distribution -> higher confidence."""
    # [0.8, 0.2] is more concentrated than [0.6, 0.4]
    conf_high = compute_entropy_confidence([0.8, 0.2])
    conf_low = compute_entropy_confidence([0.6, 0.4])
    assert conf_high > conf_low


def test_entropy_confidence_dict_input():
    """Dict input is treated identically to equivalent list input."""
    probs_list = [0.7, 0.2, 0.1]
    probs_dict = {"a": 0.7, "b": 0.2, "c": 0.1}
    assert compute_entropy_confidence(probs_list) == pytest.approx(
        compute_entropy_confidence(probs_dict), abs=1e-10
    )


def test_entropy_confidence_output_bounded():
    """Output is always in [0.0, 1.0] for valid inputs."""
    import random
    random.seed(0)
    for _ in range(50):
        n = random.randint(2, 20)
        raw = [random.random() for _ in range(n)]
        total = sum(raw)
        probs = [p / total for p in raw]
        conf = compute_entropy_confidence(probs)
        assert 0.0 <= conf <= 1.0, f"Out of bounds: {conf} for {probs}"


def test_entropy_confidence_normalization_tolerance():
    """Slightly denormalized inputs should still return valid result."""
    # Probabilities that sum to 0.999 instead of 1.0
    probs = [0.499, 0.499, 0.001]
    conf = compute_entropy_confidence(probs)
    assert 0.0 <= conf <= 1.0


def test_entropy_confidence_all_zero_returns_zero():
    """All-zero probabilities cannot form a distribution -> return 0.0."""
    assert compute_entropy_confidence([0.0, 0.0, 0.0]) == pytest.approx(0.0, abs=1e-6)
