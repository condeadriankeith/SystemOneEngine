"""Composite System One Multi-Task Decision Model.

Combines transformer encoder backbone with parallel decision heads:
- DynamicChoiceHead
- BooleanHead
- ScoreHead
"""

from typing import Literal
import torch
import torch.nn as nn
import torch.nn.functional as F
from system_one_engine.models.backbone import TransformerBackbone, TinyBackbone
from system_one_engine.models.heads import BooleanHead, DynamicChoiceHead, ScoreHead


class SystemOneModel(nn.Module):
    """Unified System One neural architecture for non-autoregressive decision classification.

    Attributes:
        backbone: Transformer encoder module extracting state vectors h_state.
        choice_head: Permutation-equivariant candidate selection head.
        boolean_head: Calibrated binary classification head.
        score_head: Ordinal multi-level classification and expectation head.
    """

    def __init__(
        self,
        backbone: TransformerBackbone | TinyBackbone,
        projection_dim: int | None = None,
        max_score_levels: int = 10,
        dropout: float = 0.1,
    ) -> None:
        """Initialize composite System One model.

        Args:
            backbone: Transformer backbone instance (ModernBERT, DeBERTa, or TinyBackbone).
            projection_dim: Projection dimension for choice dot-product head.
            max_score_levels: Maximum allowable discrete score levels (default 10).
            dropout: Dropout probability across heads.
        """
        super().__init__()
        self.backbone = backbone
        self.hidden_size: int = backbone.hidden_size

        self.choice_head = DynamicChoiceHead(
            hidden_size=self.hidden_size,
            projection_dim=projection_dim,
            dropout=dropout,
        )
        self.boolean_head = BooleanHead(
            hidden_size=self.hidden_size,
            dropout=dropout,
        )
        self.score_head = ScoreHead(
            hidden_size=self.hidden_size,
            max_levels=max_score_levels,
            dropout=dropout,
        )

    def encode(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
    ) -> torch.Tensor:
        """Encode tokenized input sequences into pooled state representations.

        Args:
            input_ids: Tensor of token IDs (batch_size, seq_len).
            attention_mask: Attention mask (batch_size, seq_len).

        Returns:
            torch.Tensor: Pooled state representations (batch_size, hidden_size).
        """
        return self.backbone(input_ids=input_ids, attention_mask=attention_mask)

    def forward_choice(
        self,
        context_input_ids: torch.Tensor,
        context_attention_mask: torch.Tensor,
        candidate_input_ids: torch.Tensor,
        candidate_attention_mask: torch.Tensor,
        candidate_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Evaluate a categorical Choice decision over dynamic candidate representations.

        Args:
            context_input_ids: Context tokens of shape (batch_size, seq_len).
            context_attention_mask: Context mask of shape (batch_size, seq_len).
            candidate_input_ids: Candidate tokens of shape (batch_size, num_candidates, cand_seq_len).
            candidate_attention_mask: Candidate mask of shape (batch_size, num_candidates, cand_seq_len).
            candidate_mask: Optional mask for valid candidate positions (batch_size, num_candidates).

        Returns:
            torch.Tensor: Logits across candidate options with shape (batch_size, num_candidates).
        """
        batch_size, num_candidates, cand_seq_len = candidate_input_ids.shape

        # 1. Encode context state: (B, D)
        h_state = self.encode(context_input_ids, context_attention_mask)

        # 2. Flatten and encode candidate options: (B * K, D)
        flat_cand_ids = candidate_input_ids.view(batch_size * num_candidates, cand_seq_len)
        flat_cand_mask = candidate_attention_mask.view(batch_size * num_candidates, cand_seq_len)
        flat_cand_emb = self.encode(flat_cand_ids, flat_cand_mask)

        # Reshape candidates to (B, K, D)
        cand_embeddings = flat_cand_emb.view(batch_size, num_candidates, self.hidden_size)

        # 3. Compute dot-product logits: (B, K)
        return self.choice_head(h_state, cand_embeddings, candidate_mask=candidate_mask)

    def forward_boolean(
        self,
        context_input_ids: torch.Tensor,
        context_attention_mask: torch.Tensor,
    ) -> torch.Tensor:
        """Evaluate a binary Boolean condition.

        Args:
            context_input_ids: Context tokens of shape (batch_size, seq_len).
            context_attention_mask: Context mask of shape (batch_size, seq_len).

        Returns:
            torch.Tensor: Binary logits of shape (batch_size,).
        """
        h_state = self.encode(context_input_ids, context_attention_mask)
        return self.boolean_head(h_state)

    def forward_score(
        self,
        context_input_ids: torch.Tensor,
        context_attention_mask: torch.Tensor,
        num_levels: int | torch.Tensor = 10,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Evaluate an ordinal Score ladder.

        Args:
            context_input_ids: Context tokens of shape (batch_size, seq_len).
            context_attention_mask: Context mask of shape (batch_size, seq_len).
            num_levels: Number of active levels m (2 <= m <= 10).

        Returns:
            tuple of (masked_logits, probabilities, expected_score).
        """
        h_state = self.encode(context_input_ids, context_attention_mask)
        return self.score_head(h_state, num_levels=num_levels)

    def compute_loss(
        self,
        task_type: Literal["choice", "boolean", "score"],
        predictions: torch.Tensor,
        targets: torch.Tensor,
    ) -> torch.Tensor:
        """Compute the appropriate loss function for the specified decision primitive.

        Args:
            task_type: Primitive task type ('choice', 'boolean', 'score').
            predictions: Model output predictions (logits or expected score).
            targets: Ground truth targets.

        Returns:
            torch.Tensor: Scalar loss tensor.
        """
        if task_type == "choice":
            # Cross-Entropy loss
            if targets.dtype == torch.long:
                return F.cross_entropy(predictions, targets)
            # Soft targets (cross-entropy with target probability distribution)
            log_probs = F.log_softmax(predictions, dim=-1)
            return -(targets * log_probs).sum(dim=-1).mean()

        elif task_type == "boolean":
            # Binary Cross-Entropy with Logits loss
            return F.binary_cross_entropy_with_logits(predictions, targets.float())

        elif task_type == "score":
            # Ordinal level cross-entropy if targets are discrete indices or distributions,
            # or Smooth L1 loss if targets are continuous expected scores
            if targets.dtype == torch.long:
                return F.cross_entropy(predictions, targets)
            elif targets.ndim == predictions.ndim and targets.shape[-1] == predictions.shape[-1]:
                # Soft level distribution
                log_probs = F.log_softmax(predictions, dim=-1)
                return -(targets * log_probs).sum(dim=-1).mean()
            else:
                # Continuous scalar target against predicted scalar
                return F.smooth_l1_loss(predictions, targets.float())

        raise ValueError(f"Unknown task_type: {task_type}")
