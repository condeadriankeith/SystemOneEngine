"""Post-hoc probability calibration and Expected Calibration Error (ECE) metrics.

Provides:
- compute_ece: Expected Calibration Error computation across equal-width confidence bins.
- TemperatureScaler: Post-hoc temperature optimization (NLL minimization) to ensure
  the PRD requirement of ECE <= 0.05.
"""

import json
from pathlib import Path
from typing import Sequence
import numpy as np
from scipy.optimize import minimize_scalar


def compute_ece(
    confidences: Sequence[float] | np.ndarray,
    accuracies: Sequence[int | bool] | np.ndarray,
    n_bins: int = 10,
) -> float:
    """Calculate the Expected Calibration Error (ECE).

    Bins predictions into equal-width confidence intervals, computing the weighted
    absolute difference between average confidence and empirical accuracy in each bin:

        ECE = sum_{b=1}^{B} (|B_b| / N) * |acc(B_b) - conf(B_b)|

    Args:
        confidences: Sequence of predicted probabilities or top-class confidences in [0.0, 1.0].
        accuracies: Sequence of binary indicators (1/True if prediction was correct, 0/False otherwise).
        n_bins: Number of confidence bins (default 10).

    Returns:
        float: Expected Calibration Error in [0.0, 1.0].
    """
    conf_arr = np.asarray(confidences, dtype=np.float64)
    acc_arr = np.asarray(accuracies, dtype=np.float64)

    if len(conf_arr) == 0:
        return 0.0

    if len(conf_arr) != len(acc_arr):
        raise ValueError(
            f"Length mismatch: {len(conf_arr)} confidences vs {len(acc_arr)} accuracies."
        )

    # Define equal-width bin boundaries in [0.0, 1.0]
    bin_boundaries = np.linspace(0.0, 1.0, n_bins + 1)
    total_samples = len(conf_arr)
    ece = 0.0

    for i in range(n_bins):
        bin_lower = bin_boundaries[i]
        bin_upper = bin_boundaries[i + 1]

        # Include upper boundary on the last bin
        if i == n_bins - 1:
            in_bin = (conf_arr >= bin_lower) & (conf_arr <= bin_upper)
        else:
            in_bin = (conf_arr >= bin_lower) & (conf_arr < bin_upper)

        bin_count = int(np.sum(in_bin))
        if bin_count > 0:
            bin_acc = float(np.mean(acc_arr[in_bin]))
            bin_conf = float(np.mean(conf_arr[in_bin]))
            weight = bin_count / total_samples
            ece += weight * abs(bin_acc - bin_conf)

    return float(ece)


class TemperatureScaler:
    """Post-hoc temperature scaling model for probability calibration.

    Fits a single positive scalar T via Negative Log-Likelihood (cross-entropy) minimization
    over held-out validation data to recalibrate overconfident model predictions:
        P_calibrated = softmax(logits / T)
    """

    def __init__(self, temperature: float = 1.0) -> None:
        """Initialize temperature scaler with a starting temperature.

        Args:
            temperature: Scaling factor T > 0.0. Defaults to 1.0 (identity scaling).
        """
        if temperature <= 0.0:
            raise ValueError(f"Temperature must be strictly positive, got {temperature}")
        self.temperature: float = float(temperature)

    def fit(
        self,
        logits: np.ndarray,
        labels: np.ndarray,
        temp_bounds: tuple[float, float] = (0.05, 10.0),
    ) -> float:
        """Fit temperature parameter T by minimizing NLL on validation logits and labels.

        Args:
            logits: Array of shape (N, K) representing raw unnormalized model logits.
            labels: 1D array of integer class labels in range [0, K-1].
            temp_bounds: Search bounds for scalar temperature optimization (min, max).

        Returns:
            float: Optimized temperature parameter.
        """
        logits_arr = np.asarray(logits, dtype=np.float64)
        labels_arr = np.asarray(labels, dtype=np.int64)

        if logits_arr.ndim != 2:
            raise ValueError(f"Logits must be 2D array of shape (N, K), got shape {logits_arr.shape}")
        if len(logits_arr) != len(labels_arr):
            raise ValueError("Number of logits rows must match number of labels.")

        n_samples, n_classes = logits_arr.shape

        def nll_objective(temp: float) -> float:
            """Compute negative log-likelihood loss for a candidate temperature."""
            if temp <= 0.0:
                return float("inf")
            # Stable log-softmax calculation
            scaled_logits = logits_arr / temp
            max_logits = np.max(scaled_logits, axis=1, keepdims=True)
            log_sum_exp = max_logits + np.log(np.sum(np.exp(scaled_logits - max_logits), axis=1, keepdims=True))
            log_probs = scaled_logits - log_sum_exp

            # Select target class log-probabilities
            target_log_probs = log_probs[np.arange(n_samples), labels_arr]
            nll = -np.mean(target_log_probs)
            return float(nll)

        result = minimize_scalar(
            nll_objective,
            bounds=temp_bounds,
            method="bounded",
            options={"xatol": 1e-4, "maxiter": 100},
        )

        if not result.success:
            raise RuntimeError(f"Temperature scaling optimization failed: {result.message}")

        self.temperature = float(result.x)
        return self.temperature

    def calibrate(self, logits: np.ndarray) -> np.ndarray:
        """Scale logits by 1/T and apply softmax to produce calibrated probabilities.

        Args:
            logits: Array of shape (N, K) or 1D shape (K,).

        Returns:
            np.ndarray: Calibrated probability distribution summing to 1.0 along the last axis.
        """
        logits_arr = np.asarray(logits, dtype=np.float64)
        scaled_logits = logits_arr / self.temperature

        # Numerically stable softmax
        if scaled_logits.ndim == 1:
            max_logit = np.max(scaled_logits)
            exp_logits = np.exp(scaled_logits - max_logit)
            return exp_logits / np.sum(exp_logits)

        max_logits = np.max(scaled_logits, axis=-1, keepdims=True)
        exp_logits = np.exp(scaled_logits - max_logits)
        sum_exp = np.sum(exp_logits, axis=-1, keepdims=True)
        return exp_logits / sum_exp

    def save(self, filepath: str | Path) -> None:
        """Save calibrated temperature parameter to a JSON file."""
        path = Path(filepath)
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            json.dump({"temperature": self.temperature}, f, indent=2)

    @classmethod
    def load(cls, filepath: str | Path) -> "TemperatureScaler":
        """Load calibrated temperature parameter from a JSON file."""
        with open(filepath, "r", encoding="utf-8") as f:
            data = json.load(f)
        return cls(temperature=data["temperature"])
