"""Multi-Task Training Loop for System One Decision Engine.

Supports:
- AdamW optimizer with weight decay and gradient clipping.
- Task-partitioned batching for Choice, Boolean, and Score heads.
- Automatic mixed-precision (AMP) for cloud GPU acceleration (Kaggle Dual T4/P100).
- Periodic 500-step checkpointing to local and persistent storage.
"""

from dataclasses import dataclass
import logging
from pathlib import Path
from typing import Any, Sequence
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Sampler

from system_one_engine.models.system_one_model import SystemOneModel
from system_one_engine.training.dataset import (
    MultiTaskDataset,
    collate_multi_task_batch,
)

logger = logging.getLogger(__name__)


@dataclass
class TrainingHistory:
    """Training metrics and epoch summaries."""

    epoch_losses: list[float]
    val_losses: list[float]
    best_val_loss: float
    checkpoints: list[Path]


class TaskBatchSampler(Sampler[list[int]]):
    """Batches samples by homogeneous task_type to allow vectorized forward passes."""

    def __init__(
        self,
        dataset: MultiTaskDataset,
        batch_size: int = 8,
        shuffle: bool = True,
    ) -> None:
        super().__init__()
        self.dataset = dataset
        self.batch_size = batch_size
        self.shuffle = shuffle

        # Group indices by task type
        self.task_indices: dict[str, list[int]] = {}
        for idx, sample in enumerate(dataset):
            t = sample["task_type"]
            self.task_indices.setdefault(t, []).append(idx)

    def __iter__(self):
        batches: list[list[int]] = []
        for task, indices in self.task_indices.items():
            pool = list(indices)
            if self.shuffle:
                # Deterministic torch shuffle or random
                perm = torch.randperm(len(pool)).tolist()
                pool = [pool[i] for i in perm]

            for i in range(0, len(pool), self.batch_size):
                batches.append(pool[i : i + self.batch_size])

        if self.shuffle:
            batch_perm = torch.randperm(len(batches)).tolist()
            batches = [batches[i] for i in batch_perm]

        yield from batches

    def __len__(self) -> int:
        count = 0
        for indices in self.task_indices.values():
            count += (len(indices) + self.batch_size - 1) // self.batch_size
        return count


class SystemOneTrainer:
    """Multi-task trainer orchestrating loss backpropagation, evaluation, and checkpointing."""

    def __init__(
        self,
        model: SystemOneModel,
        tokenizer: Any,
        train_dataset: MultiTaskDataset,
        val_dataset: MultiTaskDataset | None = None,
        batch_size: int = 8,
        learning_rate: float = 3e-5,
        weight_decay: float = 0.01,
        num_epochs: int = 3,
        checkpoint_dir: str | Path = "checkpoints",
        checkpoint_steps: int = 500,
        device: str | torch.device | None = None,
        use_amp: bool = True,
    ) -> None:
        """Initialize trainer."""
        self.model = model
        self.tokenizer = tokenizer
        self.train_dataset = train_dataset
        self.val_dataset = val_dataset
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay
        self.num_epochs = num_epochs
        self.checkpoint_dir = Path(checkpoint_dir)
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        self.checkpoint_steps = checkpoint_steps

        # Device configuration
        if device is None:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(device)

        self.model.to(self.device)
        self.use_amp = use_amp and self.device.type == "cuda"

        # Optimizer with weight decay
        no_decay = ["bias", "LayerNorm.weight", "layer_norm.weight"]
        optimizer_grouped_parameters = [
            {
                "params": [
                    p for n, p in self.model.named_parameters()
                    if not any(nd in n for nd in no_decay) and p.requires_grad
                ],
                "weight_decay": self.weight_decay,
            },
            {
                "params": [
                    p for n, p in self.model.named_parameters()
                    if any(nd in n for nd in no_decay) and p.requires_grad
                ],
                "weight_decay": 0.0,
            },
        ]
        self.optimizer = torch.optim.AdamW(
            optimizer_grouped_parameters, lr=self.learning_rate
        )
        self.scaler = torch.amp.GradScaler("cuda", enabled=self.use_amp)

    def train(self) -> TrainingHistory:
        """Execute full training loop across epochs with checkpointing."""
        history = TrainingHistory(
            epoch_losses=[],
            val_losses=[],
            best_val_loss=float("inf"),
            checkpoints=[],
        )

        train_sampler = TaskBatchSampler(
            self.train_dataset, batch_size=self.batch_size, shuffle=True
        )
        train_loader = DataLoader(
            self.train_dataset,
            batch_sampler=train_sampler,
            collate_fn=lambda b: collate_multi_task_batch(b, self.tokenizer),
        )

        global_step = 0

        for epoch in range(self.num_epochs):
            self.model.train()
            total_train_loss = 0.0
            num_batches = 0

            for batch in train_loader:
                global_step += 1
                loss = self._training_step(batch)
                total_train_loss += loss
                num_batches += 1

                # Periodic checkpointing
                if global_step % self.checkpoint_steps == 0:
                    ckpt_path = self.save_checkpoint(f"step_{global_step}.pt")
                    history.checkpoints.append(ckpt_path)

            avg_train_loss = total_train_loss / max(1, num_batches)
            history.epoch_losses.append(avg_train_loss)

            # Validation step
            val_loss = self.evaluate() if self.val_dataset else avg_train_loss
            history.val_losses.append(val_loss)

            if val_loss < history.best_val_loss:
                history.best_val_loss = val_loss
                best_path = self.save_checkpoint("best_model.pt")
                history.checkpoints.append(best_path)

            logger.info(
                "Epoch %d/%d: Train Loss = %.4f, Val Loss = %.4f",
                epoch + 1,
                self.num_epochs,
                avg_train_loss,
                val_loss,
            )

        return history

    def _training_step(self, batch: dict[str, Any]) -> float:
        """Execute forward pass, loss calculation, and backprop for one batch."""
        self.optimizer.zero_grad()

        task_type = batch["task_types"][0]
        ctx_ids = batch["context_input_ids"].to(self.device)
        ctx_mask = batch["context_attention_mask"].to(self.device)

        with torch.amp.autocast(device_type=self.device.type, enabled=self.use_amp):
            if task_type == "choice":
                cand_ids = batch["candidate_input_ids"].to(self.device)
                cand_mask = batch["candidate_attention_mask"].to(self.device)
                cand_act = batch["candidate_active_mask"].to(self.device)
                targets = batch["targets"].to(self.device)

                logits = self.model.forward_choice(
                    ctx_ids, ctx_mask, cand_ids, cand_mask, candidate_mask=cand_act
                )
                loss = self.model.compute_loss("choice", logits, targets)

            elif task_type == "boolean":
                targets = batch["targets"].to(self.device)
                logits = self.model.forward_boolean(ctx_ids, ctx_mask)
                loss = self.model.compute_loss("boolean", logits, targets)

            elif task_type == "score":
                targets = batch["targets"].to(self.device)
                num_levels = batch["num_levels"].to(self.device)
                logits, _, _ = self.model.forward_score(
                    ctx_ids, ctx_mask, num_levels=num_levels
                )
                loss = self.model.compute_loss("score", logits, targets)
            else:
                raise ValueError(f"Unknown task: {task_type}")

        self.scaler.scale(loss).backward()
        self.scaler.unscale_(self.optimizer)
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
        self.scaler.step(self.optimizer)
        self.scaler.update()

        return float(loss.item())

    def evaluate(self) -> float:
        """Run validation evaluation and compute mean loss."""
        if not self.val_dataset:
            return 0.0

        self.model.eval()
        val_sampler = TaskBatchSampler(
            self.val_dataset, batch_size=self.batch_size, shuffle=False
        )
        val_loader = DataLoader(
            self.val_dataset,
            batch_sampler=val_sampler,
            collate_fn=lambda b: collate_multi_task_batch(b, self.tokenizer),
        )

        total_loss = 0.0
        num_batches = 0

        with torch.no_grad():
            for batch in val_loader:
                task_type = batch["task_types"][0]
                ctx_ids = batch["context_input_ids"].to(self.device)
                ctx_mask = batch["context_attention_mask"].to(self.device)

                if task_type == "choice":
                    cand_ids = batch["candidate_input_ids"].to(self.device)
                    cand_mask = batch["candidate_attention_mask"].to(self.device)
                    cand_act = batch["candidate_active_mask"].to(self.device)
                    targets = batch["targets"].to(self.device)
                    logits = self.model.forward_choice(
                        ctx_ids, ctx_mask, cand_ids, cand_mask, candidate_mask=cand_act
                    )
                    loss = self.model.compute_loss("choice", logits, targets)

                elif task_type == "boolean":
                    targets = batch["targets"].to(self.device)
                    logits = self.model.forward_boolean(ctx_ids, ctx_mask)
                    loss = self.model.compute_loss("boolean", logits, targets)

                elif task_type == "score":
                    targets = batch["targets"].to(self.device)
                    num_levels = batch["num_levels"].to(self.device)
                    logits, _, _ = self.model.forward_score(
                        ctx_ids, ctx_mask, num_levels=num_levels
                    )
                    loss = self.model.compute_loss("score", logits, targets)
                else:
                    continue

                total_loss += float(loss.item())
                num_batches += 1

        return total_loss / max(1, num_batches)

    def save_checkpoint(self, filename: str) -> Path:
        """Save model state dict and optimizer state to checkpoint directory."""
        path = self.checkpoint_dir / filename
        torch.save(
            {
                "model_state_dict": self.model.state_dict(),
                "optimizer_state_dict": self.optimizer.state_dict(),
                "hidden_size": self.model.hidden_size,
            },
            path,
        )
        return path
