"""Multi-Task Dataset loader and collation for System One training.

Supports:
- Reading tabular multi-task records (Choice, Boolean, Score).
- Ingesting raw JSONL datasets (such as system-one-all77.soft.jsonl and synthetic mock data).
- Tokenization and tensor padding for dynamic candidate sets.
"""

import json
from pathlib import Path
from typing import Any, Callable, Sequence
import torch
from torch.utils.data import Dataset


def format_state_text(state: Any) -> str:
    """Format state object (dict or string) into standardized text representation."""
    if isinstance(state, str):
        return state.strip()
    return json.dumps(state, indent=2)


class SimpleVocabTokenizer:
    """Fast, deterministic tokenizer with fixed vocabulary for offline testing and CPU mini-runs."""

    def __init__(self, vocab_size: int = 2000, max_len: int = 128) -> None:
        """Initialize simple hash-based vocabulary tokenizer."""
        self.vocab_size = vocab_size
        self.max_len = max_len
        self.pad_token_id = 0
        self.cls_token_id = 1
        self.unk_token_id = 2

    def __call__(
        self,
        texts: str | list[str],
        max_length: int | None = None,
        padding: bool = True,
        truncation: bool = True,
        return_tensors: str | None = "pt",
    ) -> dict[str, torch.Tensor]:
        """Tokenize one or more text strings into padded input_ids and attention_mask."""
        if isinstance(texts, str):
            texts = [texts]

        limit = max_length or self.max_len
        batch_ids: list[list[int]] = []
        batch_masks: list[list[int]] = []

        for text in texts:
            words = text.strip().lower().split()
            # Deterministic hash to token ID in [3, vocab_size - 1]
            token_ids = [self.cls_token_id] + [
                (abs(hash(w)) % (self.vocab_size - 3)) + 3 for w in words
            ]
            if truncation and len(token_ids) > limit:
                token_ids = token_ids[:limit]

            mask = [1] * len(token_ids)

            if padding and len(token_ids) < limit:
                pad_count = limit - len(token_ids)
                token_ids.extend([self.pad_token_id] * pad_count)
                mask.extend([0] * pad_count)

            batch_ids.append(token_ids)
            batch_masks.append(mask)

        if return_tensors == "pt":
            return {
                "input_ids": torch.tensor(batch_ids, dtype=torch.long),
                "attention_mask": torch.tensor(batch_masks, dtype=torch.long),
            }
        raise ValueError(f"Unsupported return_tensors: {return_tensors}")


class MultiTaskDataset(Dataset):
    """PyTorch Dataset yielding standardized multi-task decision samples."""

    def __init__(self, samples: Sequence[dict[str, Any]]) -> None:
        """Initialize dataset with sample records.

        Args:
            samples: List of standardized sample dictionaries.
        """
        self.samples = list(samples)

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> dict[str, Any]:
        return self.samples[idx]

    @classmethod
    def from_jsonl(cls, filepath: str | Path) -> "MultiTaskDataset":
        """Load dataset from a JSONL file, automatically parsing both mock and gold formats."""
        path = Path(filepath)
        if not path.exists():
            raise FileNotFoundError(f"Dataset file not found: {path}")

        samples: list[dict[str, Any]] = []

        with open(path, "r", encoding="utf-8") as f:
            for line_idx, line in enumerate(f):
                line = line.strip()
                if not line:
                    continue
                record = json.loads(line)

                # Format A: Direct tabular mock sample
                if "task_type" in record:
                    samples.append(record)
                    continue

                # Format B: Gold dataset (system-one-all77.soft.jsonl)
                if "questions" in record and "state" in record:
                    state_text = format_state_text(record["state"])
                    expected_map = record.get("expected", {})
                    soft_targets_map = record.get("soft_targets", {})

                    for q_id, q_data in record["questions"].items():
                        q_type = q_data.get("type")
                        instruction = q_data.get("instructions", "")

                        if q_type == "noul":
                            # Boolean task
                            gold_bool = expected_map.get(q_id)
                            target_val = 1.0 if gold_bool else 0.0
                            soft_val = (
                                soft_targets_map.get(q_id, {}).get("yes")
                                if q_id in soft_targets_map
                                else None
                            )
                            samples.append({
                                "sample_id": f"{record.get('id', line_idx)}_{q_id}",
                                "task_type": "boolean",
                                "instruction": instruction,
                                "context": f"State:\n{state_text}\n\nQuestion: {instruction}",
                                "target": target_val,
                                "soft_target": soft_val,
                            })

                        elif q_type == "score":
                            # Score task
                            criteria = q_data.get("criteria", [])
                            gold_score = expected_map.get(q_id, 0)
                            soft_dist = None
                            if q_id in soft_targets_map:
                                soft_dist = [
                                    soft_targets_map[q_id].get(str(i), 0.0)
                                    for i in range(len(criteria))
                                ]
                            samples.append({
                                "sample_id": f"{record.get('id', line_idx)}_{q_id}",
                                "task_type": "score",
                                "instruction": instruction,
                                "context": f"State:\n{state_text}\n\nQuestion: {instruction}",
                                "criteria": criteria,
                                "num_levels": len(criteria),
                                "target": int(gold_score) if isinstance(gold_score, (int, float)) else 0,
                                "soft_target": soft_dist,
                            })

                        elif q_type == "choice":
                            # Choice task
                            criteria = q_data.get("criteria", {})
                            if isinstance(criteria, dict):
                                candidates = list(criteria.keys())
                            else:
                                candidates = list(criteria)
                            gold_choice = expected_map.get(q_id)
                            target_idx = (
                                candidates.index(gold_choice)
                                if gold_choice in candidates
                                else 0
                            )
                            soft_dist = None
                            if q_id in soft_targets_map:
                                soft_dist = [
                                    soft_targets_map[q_id].get(k, 0.0) for k in candidates
                                ]
                            samples.append({
                                "sample_id": f"{record.get('id', line_idx)}_{q_id}",
                                "task_type": "choice",
                                "instruction": instruction,
                                "context": f"State:\n{state_text}\n\nQuestion: {instruction}",
                                "candidates": candidates,
                                "target_index": target_idx,
                                "soft_target": soft_dist,
                            })

        return cls(samples=samples)


def collate_multi_task_batch(
    batch: Sequence[dict[str, Any]],
    tokenizer: Any,
    max_context_len: int = 128,
    max_cand_len: int = 32,
) -> dict[str, Any]:
    """Collate a heterogeneous or homogenous batch of multi-task decision samples.

    Args:
        batch: List of sample dictionaries.
        tokenizer: Tokenizer instance.
        max_context_len: Max sequence length for context text.
        max_cand_len: Max sequence length for candidate option strings.

    Returns:
        dict containing tokenized tensors and formatted targets partitioned by task.
    """
    contexts = [s["context"] for s in batch]
    ctx_encoding = tokenizer(
        contexts,
        max_length=max_context_len,
        padding=True,
        truncation=True,
        return_tensors="pt",
    )

    batch_tasks = [s["task_type"] for s in batch]

    collated: dict[str, Any] = {
        "context_input_ids": ctx_encoding["input_ids"],
        "context_attention_mask": ctx_encoding["attention_mask"],
        "task_types": batch_tasks,
        "raw_samples": batch,
    }

    # If homogeneous batch or partitioned by task:
    if all(t == "choice" for t in batch_tasks):
        # Handle dynamic candidate tensor: (B, max_k, cand_len)
        max_k = max(len(s["candidates"]) for s in batch)
        batch_size = len(batch)

        cand_ids = torch.zeros((batch_size, max_k, max_cand_len), dtype=torch.long)
        cand_mask = torch.zeros((batch_size, max_k, max_cand_len), dtype=torch.long)
        candidate_active_mask = torch.zeros((batch_size, max_k), dtype=torch.long)
        targets = torch.zeros(batch_size, dtype=torch.long)

        for b_idx, s in enumerate(batch):
            cands = s["candidates"]
            k = len(cands)
            candidate_active_mask[b_idx, :k] = 1
            targets[b_idx] = s.get("target_index", 0)

            c_enc = tokenizer(
                cands,
                max_length=max_cand_len,
                padding=True,
                truncation=True,
                return_tensors="pt",
            )
            cand_ids[b_idx, :k, : c_enc["input_ids"].shape[1]] = c_enc["input_ids"]
            cand_mask[b_idx, :k, : c_enc["attention_mask"].shape[1]] = c_enc["attention_mask"]

        collated["candidate_input_ids"] = cand_ids
        collated["candidate_attention_mask"] = cand_mask
        collated["candidate_active_mask"] = candidate_active_mask
        collated["targets"] = targets

    elif all(t == "boolean" for t in batch_tasks):
        targets = torch.tensor(
            [float(s.get("target", 0.0)) for s in batch], dtype=torch.float32
        )
        collated["targets"] = targets

    elif all(t == "score" for t in batch_tasks):
        targets = torch.tensor(
            [int(s.get("target", 0)) for s in batch], dtype=torch.long
        )
        num_levels = torch.tensor(
            [int(s.get("num_levels", 10)) for s in batch], dtype=torch.long
        )
        collated["targets"] = targets
        collated["num_levels"] = num_levels

    return collated
