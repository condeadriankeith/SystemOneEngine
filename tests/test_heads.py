"""Unit tests for DynamicChoiceHead, BooleanHead, and ScoreHead."""

import torch
import torch.nn.functional as F
from system_one_engine.models.heads import (
    BooleanHead,
    DynamicChoiceHead,
    ScoreHead,
)


# ---------------------------------------------------------------------------
# DynamicChoiceHead Tests
# ---------------------------------------------------------------------------


def test_dynamic_choice_head_forward_shape():
    """Verify output logits shape matches (batch_size, num_candidates)."""
    batch_size = 4
    num_candidates = 6
    hidden_size = 32

    head = DynamicChoiceHead(hidden_size=hidden_size, projection_dim=16)

    context_state = torch.randn(batch_size, hidden_size)
    candidate_embeddings = torch.randn(batch_size, num_candidates, hidden_size)

    logits = head(context_state, candidate_embeddings)
    assert logits.shape == (batch_size, num_candidates)


def test_dynamic_choice_head_permutation_equivariance():
    """Verify that reordering candidate embeddings yields an identically permuted logit vector.

    This mathematically proves zero positional bias.
    """
    batch_size = 2
    num_candidates = 5
    hidden_size = 32

    # Fixed seed for deterministic head weights
    torch.manual_seed(42)
    head = DynamicChoiceHead(hidden_size=hidden_size, dropout=0.0)
    head.eval()

    context_state = torch.randn(batch_size, hidden_size)
    candidate_embeddings = torch.randn(batch_size, num_candidates, hidden_size)

    # 1. Original forward pass
    with torch.no_grad():
        original_logits = head(context_state, candidate_embeddings)

    # 2. Permute candidates (e.g. reverse order: [4, 3, 2, 1, 0])
    perm = torch.tensor([4, 3, 2, 1, 0])
    permuted_candidates = candidate_embeddings[:, perm, :]

    with torch.no_grad():
        permuted_logits = head(context_state, permuted_candidates)

    # 3. Check that original_logits[:, perm] == permuted_logits
    expected_logits = original_logits[:, perm]
    assert torch.allclose(permuted_logits, expected_logits, atol=1e-5)


def test_dynamic_choice_head_masking():
    """Verify candidate_mask properly zeroes out probability for padded candidates."""
    batch_size = 2
    num_candidates = 4
    hidden_size = 32

    head = DynamicChoiceHead(hidden_size=hidden_size, dropout=0.0)
    head.eval()

    context_state = torch.randn(batch_size, hidden_size)
    candidate_embeddings = torch.randn(batch_size, num_candidates, hidden_size)

    # Mask out the last 2 candidates for batch item 0 (only 2 active candidates)
    mask = torch.tensor([
        [1, 1, 0, 0],
        [1, 1, 1, 1],
    ])

    logits = head(context_state, candidate_embeddings, candidate_mask=mask)
    probs = F.softmax(logits, dim=-1)

    # Masked positions should have near-zero probability
    assert probs[0, 2].item() < 1e-6
    assert probs[0, 3].item() < 1e-6
    assert probs[0, :2].sum().item() == pytest.approx(1.0, abs=1e-5)


# ---------------------------------------------------------------------------
# BooleanHead Tests
# ---------------------------------------------------------------------------


def test_boolean_head_forward():
    """Verify boolean head returns 1D logits and supports BCE loss."""
    batch_size = 4
    hidden_size = 32

    head = BooleanHead(hidden_size=hidden_size)
    context_state = torch.randn(batch_size, hidden_size)

    logits = head(context_state)
    assert logits.shape == (batch_size,)

    # Test BCE loss
    targets = torch.tensor([1.0, 0.0, 1.0, 0.0])
    loss = F.binary_cross_entropy_with_logits(logits, targets)
    assert loss.item() >= 0.0

    # Test backward pass
    loss.backward()
    for param in head.parameters():
        assert param.grad is not None


# ---------------------------------------------------------------------------
# ScoreHead Tests
# ---------------------------------------------------------------------------


def test_score_head_forward_and_bounds():
    """Verify ordinal score head produces valid logits, probabilities, and bounded expected score."""
    batch_size = 4
    hidden_size = 32
    max_levels = 10
    num_active_levels = 4

    head = ScoreHead(hidden_size=hidden_size, max_levels=max_levels, dropout=0.0)
    context_state = torch.randn(batch_size, hidden_size)

    masked_logits, probs, score = head(context_state, num_levels=num_active_levels)

    assert masked_logits.shape == (batch_size, max_levels)
    assert probs.shape == (batch_size, max_levels)
    assert score.shape == (batch_size,)

    # Inactive levels (>= 4) must have zero probability
    assert torch.all(probs[:, 4:] < 1e-6)

    # Active levels sum to 1.0
    active_sums = probs[:, :num_active_levels].sum(dim=-1)
    assert torch.allclose(active_sums, torch.ones_like(active_sums), atol=1e-5)

    # Expected score must be strictly within [0.0, num_active_levels - 1]
    assert torch.all(score >= 0.0)
    assert torch.all(score <= float(num_active_levels - 1))


import pytest
