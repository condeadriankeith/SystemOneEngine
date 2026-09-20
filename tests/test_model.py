"""Unit tests for SystemOneModel and mock data generation."""

import json
import pytest
import torch
from system_one_engine.models.backbone import PoolingStrategy, TinyBackbone
from system_one_engine.models.system_one_model import SystemOneModel
from system_one_engine.training.mock_generator import (
    generate_mock_dataset,
    save_mock_dataset,
)


def test_tiny_backbone_forward_pooling():
    """Verify TinyBackbone forward pass with both CLS and MEAN pooling."""
    batch_size = 2
    seq_len = 16
    hidden_size = 32

    # Mean pooling
    backbone_mean = TinyBackbone(
        vocab_size=100, hidden_size=hidden_size, pooling=PoolingStrategy.MEAN
    )
    input_ids = torch.randint(0, 100, (batch_size, seq_len))
    mask = torch.ones(batch_size, seq_len, dtype=torch.long)

    out_mean = backbone_mean(input_ids, mask)
    assert out_mean.shape == (batch_size, hidden_size)

    # CLS pooling
    backbone_cls = TinyBackbone(
        vocab_size=100, hidden_size=hidden_size, pooling=PoolingStrategy.CLS
    )
    out_cls = backbone_cls(input_ids, mask)
    assert out_cls.shape == (batch_size, hidden_size)


def test_system_one_model_choice_forward_and_backward():
    """Verify SystemOneModel Choice forward pass, loss calculation, and backprop."""
    batch_size = 2
    seq_len = 8
    num_candidates = 3
    cand_len = 6
    hidden_size = 32

    backbone = TinyBackbone(vocab_size=100, hidden_size=hidden_size)
    model = SystemOneModel(backbone=backbone, projection_dim=16)

    context_ids = torch.randint(0, 100, (batch_size, seq_len))
    context_mask = torch.ones(batch_size, seq_len, dtype=torch.long)

    cand_ids = torch.randint(0, 100, (batch_size, num_candidates, cand_len))
    cand_mask = torch.ones(batch_size, num_candidates, cand_len, dtype=torch.long)

    logits = model.forward_choice(context_ids, context_mask, cand_ids, cand_mask)
    assert logits.shape == (batch_size, num_candidates)

    # Target indices: (B,)
    targets = torch.tensor([1, 2], dtype=torch.long)
    loss = model.compute_loss("choice", logits, targets)
    assert loss.item() > 0.0

    loss.backward()
    # Ensure gradients flow to both backbone and choice head
    assert backbone.token_embeddings.weight.grad is not None
    assert model.choice_head.query_proj[0].weight.grad is not None


def test_system_one_model_boolean_forward_and_backward():
    """Verify SystemOneModel Boolean forward pass and backprop."""
    batch_size = 3
    seq_len = 8
    hidden_size = 32

    backbone = TinyBackbone(vocab_size=100, hidden_size=hidden_size)
    model = SystemOneModel(backbone=backbone)

    context_ids = torch.randint(0, 100, (batch_size, seq_len))
    context_mask = torch.ones(batch_size, seq_len, dtype=torch.long)

    logits = model.forward_boolean(context_ids, context_mask)
    assert logits.shape == (batch_size,)

    targets = torch.tensor([1, 0, 1], dtype=torch.float32)
    loss = model.compute_loss("boolean", logits, targets)
    assert loss.item() >= 0.0

    loss.backward()
    assert model.boolean_head.classifier[0].weight.grad is not None


def test_system_one_model_score_forward_and_backward():
    """Verify SystemOneModel Score forward pass and backprop."""
    batch_size = 3
    seq_len = 8
    hidden_size = 32
    num_levels = 4

    backbone = TinyBackbone(vocab_size=100, hidden_size=hidden_size)
    model = SystemOneModel(backbone=backbone, max_score_levels=10)

    context_ids = torch.randint(0, 100, (batch_size, seq_len))
    context_mask = torch.ones(batch_size, seq_len, dtype=torch.long)

    masked_logits, probs, score = model.forward_score(
        context_ids, context_mask, num_levels=num_levels
    )
    assert masked_logits.shape == (batch_size, 10)
    assert probs.shape == (batch_size, 10)
    assert score.shape == (batch_size,)

    # Target class index
    targets = torch.tensor([0, 2, 3], dtype=torch.long)
    loss = model.compute_loss("score", masked_logits, targets)
    assert loss.item() >= 0.0

    loss.backward()
    assert model.score_head.classifier[0].weight.grad is not None


def test_mock_data_generator_distribution_and_io(tmp_path):
    """Verify synthetic mock data generator produces 500 valid samples across tasks."""
    samples = generate_mock_dataset(num_samples=500, seed=123)
    assert len(samples) == 500

    task_types = {s["task_type"] for s in samples}
    assert task_types == {"choice", "boolean", "score"}

    # Save to file and verify roundtrip
    out_file = tmp_path / "mock_dataset.jsonl"
    save_mock_dataset(out_file, num_samples=100, seed=42)
    assert out_file.exists()

    with open(out_file, "r", encoding="utf-8") as f:
        loaded_lines = [json.loads(line) for line in f]
    assert len(loaded_lines) == 100
