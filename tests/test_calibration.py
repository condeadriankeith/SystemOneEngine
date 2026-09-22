"""Unit tests for Expected Calibration Error (ECE) and TemperatureScaler."""

import numpy as np
import pytest
from system_one_engine.core.calibration import (
    TEMP_MAX,
    TEMP_MIN,
    TemperatureScaler,
    clamp_temperature,
    compute_ece,
    temp_bucket,
)


def test_compute_ece_perfect_calibration():
    """Verify compute_ece produces ~0.0 error for well-calibrated predictions."""
    # 1000 samples where prediction probability is 0.8, and 80% are correct
    np.random.seed(42)
    n = 1000
    confidences = np.full(n, 0.8)
    accuracies = np.random.binomial(1, 0.8, size=n)

    ece = compute_ece(confidences, accuracies, n_bins=10)
    # Due to sampling variance, ECE should be very close to 0.0 (< 0.02)
    assert ece < 0.03


def test_compute_ece_overconfident():
    """Verify compute_ece detects severe miscalibration."""
    # Confidence is 0.99, but actual accuracy is only 0.50
    confidences = np.full(100, 0.99)
    accuracies = np.array([1, 0] * 50)

    ece = compute_ece(confidences, accuracies, n_bins=10)
    # Expected error is |0.50 - 0.99| = 0.49
    assert ece == pytest.approx(0.49, abs=1e-2)


def test_compute_ece_empty_or_mismatch():
    """Verify edge cases for compute_ece."""
    # Empty inputs
    assert compute_ece([], []) == 0.0

    # Length mismatch
    with pytest.raises(ValueError):
        compute_ece([0.8, 0.9], [1])


def test_temperature_scaler_initialization():
    """Verify TemperatureScaler initialization constraints."""
    scaler = TemperatureScaler(temperature=1.5)
    assert scaler.temperature == 1.5

    with pytest.raises(ValueError):
        TemperatureScaler(temperature=0.0)

    with pytest.raises(ValueError):
        TemperatureScaler(temperature=-1.0)


def test_temperature_scaler_fit_and_recalibration(tmp_path):
    """Verify TemperatureScaler reduces ECE on overconfident synthetic logits."""
    np.random.seed(42)
    n_samples = 1500
    n_classes = 4

    # Generate synthetic uncalibrated logits scaled by a factor of 2.5 (overconfident)
    true_probs = np.random.dirichlet(np.ones(n_classes), size=n_samples)
    labels = np.array([np.random.choice(n_classes, p=p) for p in true_probs])

    # Uncalibrated logits = log(p) * 2.5 (making softmax sharper than true probabilities)
    uncalibrated_logits = (np.log(true_probs + 1e-12)) * 2.5

    # Compute pre-calibration ECE
    raw_exp = np.exp(uncalibrated_logits - np.max(uncalibrated_logits, axis=1, keepdims=True))
    raw_probs = raw_exp / np.sum(raw_exp, axis=1, keepdims=True)
    raw_preds = np.argmax(raw_probs, axis=1)
    raw_confs = np.max(raw_probs, axis=1)
    raw_accs = (raw_preds == labels).astype(int)

    pre_ece = compute_ece(raw_confs, raw_accs)

    # Fit Temperature Scaler
    scaler = TemperatureScaler()
    opt_temp = scaler.fit(uncalibrated_logits, labels)

    # Optimum temperature should be close to 2.5
    assert 1.8 <= opt_temp <= 3.2

    # Calibrate logits
    calibrated_probs = scaler.calibrate(uncalibrated_logits)
    cal_preds = np.argmax(calibrated_probs, axis=1)
    cal_confs = np.max(calibrated_probs, axis=1)
    cal_accs = (cal_preds == labels).astype(int)

    post_ece = compute_ece(cal_confs, cal_accs)

    # Ensure calibration improved significantly and adheres to PRD target (<= 0.05)
    assert post_ece < pre_ece
    assert post_ece <= 0.05

    # Test save and load serialization
    save_file = tmp_path / "calibration_temp.json"
    scaler.save(save_file)
    loaded_scaler = TemperatureScaler.load(save_file)
    assert loaded_scaler.temperature == pytest.approx(scaler.temperature, abs=1e-5)


# ---------------------------------------------------------------------------
# temp_bucket (laya-mlx alignment)
# ---------------------------------------------------------------------------


def test_temp_bucket_choice_k2():
    assert temp_bucket(0, 2) == "choice:2"


def test_temp_bucket_choice_k3():
    assert temp_bucket(0, 3) == "choice:3-5"


def test_temp_bucket_choice_k5():
    assert temp_bucket(0, 5) == "choice:3-5"


def test_temp_bucket_choice_k6():
    assert temp_bucket(0, 6) == "choice:6-10"


def test_temp_bucket_choice_k10():
    assert temp_bucket(0, 10) == "choice:6-10"


def test_temp_bucket_choice_k11():
    assert temp_bucket(0, 11) == "choice:11+"


def test_temp_bucket_score():
    assert temp_bucket(1, 4) == "score:3-5"


def test_temp_bucket_noul():
    assert temp_bucket(2, 2) == "noul:2"


def test_temp_bucket_unknown_qtype_defaults_to_choice():
    # qtype 99 is not in the _QTYPE_NAMES map -> falls back to "choice"
    result = temp_bucket(99, 3)
    assert result == "choice:3-5"


# ---------------------------------------------------------------------------
# clamp_temperature (laya-mlx alignment)
# ---------------------------------------------------------------------------


def test_clamp_temperature_within_bounds():
    assert clamp_temperature(1.0) == pytest.approx(1.0, abs=1e-9)
    assert clamp_temperature(2.5) == pytest.approx(2.5, abs=1e-9)


def test_clamp_temperature_below_min():
    # Values below TEMP_MIN (0.5) are raised to TEMP_MIN
    result = clamp_temperature(0.1)
    assert result == pytest.approx(TEMP_MIN, abs=1e-9)


def test_clamp_temperature_above_max():
    result = clamp_temperature(100.0)
    assert result == pytest.approx(TEMP_MAX, abs=1e-9)


def test_clamp_temperature_nan_returns_one():
    import math
    assert clamp_temperature(math.nan) == pytest.approx(1.0, abs=1e-9)


def test_clamp_temperature_inf_returns_one():
    import math
    assert clamp_temperature(math.inf) == pytest.approx(1.0, abs=1e-9)


def test_clamp_temperature_non_numeric_returns_one():
    assert clamp_temperature("hot") == pytest.approx(1.0, abs=1e-9)


def test_clamp_temperature_custom_bounds():
    assert clamp_temperature(3.0, lo=4.0, hi=8.0) == pytest.approx(4.0, abs=1e-9)
    assert clamp_temperature(10.0, lo=4.0, hi=8.0) == pytest.approx(8.0, abs=1e-9)
