"""Unit and mini-run tests for MultiTaskDataset, SystemOneTrainer, and Kaggle bundle."""

import json
import pytest
import torch
from system_one_engine.models.backbone import TinyBackbone
from system_one_engine.models.system_one_model import SystemOneModel
from system_one_engine.training.dataset import (
    MultiTaskDataset,
    SimpleVocabTokenizer,
    collate_multi_task_batch,
)
from system_one_engine.training.kaggle_bundle import generate_kaggle_notebook
from system_one_engine.training.mock_generator import generate_mock_dataset
from system_one_engine.training.trainer import SystemOneTrainer


def test_multi_task_dataset_mock_samples():
    """Verify MultiTaskDataset correctly indexes mock multi-task samples."""
    samples = generate_mock_dataset(num_samples=30, seed=42)
    dataset = MultiTaskDataset(samples)
    assert len(dataset) == 30

    first_sample = dataset[0]
    assert "task_type" in first_sample
    assert "context" in first_sample
    assert "instruction" in first_sample


def test_gold_dataset_jsonl_parsing():
    """Verify MultiTaskDataset can parse the ingested gold dataset."""
    gold_path = "data/datasets/system-one-all77.soft.jsonl"
    dataset = MultiTaskDataset.from_jsonl(gold_path)

    # Should contain thousands of individual question evaluations
    assert len(dataset) > 1000

    task_types = {s["task_type"] for s in dataset}
    assert "boolean" in task_types
    assert "score" in task_types


def test_simple_vocab_tokenizer():
    """Verify SimpleVocabTokenizer produces padded tensors."""
    tokenizer = SimpleVocabTokenizer(vocab_size=500, max_len=16)
    out = tokenizer(["hello world", "test sentence with multiple words"])

    assert out["input_ids"].shape == (2, 16)
    assert out["attention_mask"].shape == (2, 16)
    # Check padding token at the end of the shorter sentence
    assert out["input_ids"][0, -1].item() == 0
    assert out["attention_mask"][0, -1].item() == 0


def test_collate_multi_task_batch():
    """Verify collation of Choice and Boolean batches."""
    tokenizer = SimpleVocabTokenizer(vocab_size=500, max_len=16)

    # Choice batch
    choice_batch = [
        {
            "task_type": "choice",
            "context": "Context 1",
            "instruction": "Route query",
            "candidates": ["opt_a", "opt_b"],
            "target_index": 0,
        },
        {
            "task_type": "choice",
            "context": "Context 2",
            "instruction": "Route query",
            "candidates": ["opt_a", "opt_b", "opt_c"],
            "target_index": 2,
        },
    ]

    collated_choice = collate_multi_task_batch(choice_batch, tokenizer, max_cand_len=8)
    assert collated_choice["context_input_ids"].shape[0] == 2
    assert collated_choice["candidate_input_ids"].shape == (2, 3, 8)
    assert collated_choice["candidate_active_mask"].shape == (2, 3)
    assert collated_choice["candidate_active_mask"][0, 2].item() == 0  # 3rd candidate is padded for item 0
    assert collated_choice["targets"].shape == (2,)

    # Boolean batch
    bool_batch = [
        {"task_type": "boolean", "context": "Ctx A", "instruction": "Check", "target": 1.0},
        {"task_type": "boolean", "context": "Ctx B", "instruction": "Check", "target": 0.0},
    ]
    collated_bool = collate_multi_task_batch(bool_batch, tokenizer)
    assert collated_bool["targets"].shape == (2,)


def test_trainer_mini_run_cpu(tmp_path):
    """Verify end-to-end CPU training loop with loss calculation and checkpoint saving."""
    samples = generate_mock_dataset(num_samples=24, seed=42)
    train_ds = MultiTaskDataset(samples[:18])
    val_ds = MultiTaskDataset(samples[18:])

    tokenizer = SimpleVocabTokenizer(vocab_size=500, max_len=16)
    backbone = TinyBackbone(vocab_size=500, hidden_size=32)
    model = SystemOneModel(backbone=backbone, projection_dim=16)

    ckpt_dir = tmp_path / "checkpoints"

    trainer = SystemOneTrainer(
        model=model,
        tokenizer=tokenizer,
        train_dataset=train_ds,
        val_dataset=val_ds,
        batch_size=4,
        learning_rate=1e-3,
        num_epochs=1,
        checkpoint_dir=ckpt_dir,
        checkpoint_steps=100,
        device="cpu",
        use_amp=False,
    )

    history = trainer.train()

    assert len(history.epoch_losses) == 1
    assert history.epoch_losses[0] > 0.0
    assert (ckpt_dir / "best_model.pt").exists()


def test_generate_kaggle_notebook(tmp_path):
    """Verify generate_kaggle_notebook produces a valid Jupyter notebook."""
    nb_path = tmp_path / "test_kaggle.ipynb"
    result_path = generate_kaggle_notebook(nb_path)

    assert result_path.exists()
    with open(result_path, "r", encoding="utf-8") as f:
        nb_json = json.load(f)

    assert "cells" in nb_json
    assert len(nb_json["cells"]) >= 5
    assert nb_json["metadata"]["accelerator"] == "GPU"
