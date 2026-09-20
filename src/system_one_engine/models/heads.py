"""Parallel decision heads for System One Decision Engine.

Provides:
- DynamicChoiceHead: Permutation-equivariant candidate selection via scaled dot-product.
- BooleanHead: Calibrated binary classification head for affirmative assertions.
- ScoreHead: Ordinal level classification and continuous expected-value regression.
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F


class DynamicChoiceHead(nn.Module):
    """Permutation-equivariant dynamic Choice head using scaled dot-product attention.

    Computes normalized similarity between query context vector h_state and
    dynamically supplied candidate representations {e_c1, ..., e_ck}:

        q = W_q(h_state)
        k_i = W_k(e_c_i)
        logit_i = (q^T * k_i) / sqrt(d_proj)

    Because logits are computed via pairwise dot products with no positional bias,
    this head is permutation-equivariant by construction.
    """

    def __init__(
        self,
        hidden_size: int,
        projection_dim: int | None = None,
        dropout: float = 0.1,
    ) -> None:
        """Initialize dynamic choice head.

        Args:
            hidden_size: Input embedding dimension d.
            projection_dim: Latent dot-product dimension (defaults to hidden_size).
            dropout: Dropout probability.
        """
        super().__init__()
        self.hidden_size = hidden_size
        self.projection_dim = projection_dim or hidden_size
        self.scale = 1.0 / math.sqrt(self.projection_dim)

        self.query_proj = nn.Sequential(
            nn.Linear(self.hidden_size, self.projection_dim),
            nn.LayerNorm(self.projection_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(self.projection_dim, self.projection_dim),
        )

        self.candidate_proj = nn.Sequential(
            nn.Linear(self.hidden_size, self.projection_dim),
            nn.LayerNorm(self.projection_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(self.projection_dim, self.projection_dim),
        )

    def forward(
        self,
        context_state: torch.Tensor,
        candidate_embeddings: torch.Tensor,
        candidate_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Compute logits across dynamic candidates.

        Args:
            context_state: Pooled state representation with shape (batch_size, hidden_size).
            candidate_embeddings: Candidate embeddings with shape (batch_size, num_candidates, hidden_size).
            candidate_mask: Optional binary mask of shape (batch_size, num_candidates)
                            where 1 = valid candidate, 0 = padding.

        Returns:
            torch.Tensor: Logits across candidates with shape (batch_size, num_candidates).
        """
        # q: (B, 1, D_proj)
        q = self.query_proj(context_state).unsqueeze(1)
        # k: (B, K, D_proj)
        k = self.candidate_proj(candidate_embeddings)

        # Scaled dot-product: (B, 1, D_proj) x (B, D_proj, K) -> (B, 1, K) -> (B, K)
        logits = torch.bmm(q, k.transpose(1, 2)).squeeze(1) * self.scale

        if candidate_mask is not None:
            # Mask out invalid/padded candidate slots
            logits = logits.masked_fill(candidate_mask == 0, -1e9)

        return logits


class BooleanHead(nn.Module):
    """Calibrated binary classification head for True / False policy assertions."""

    def __init__(
        self,
        hidden_size: int,
        dropout: float = 0.1,
    ) -> None:
        """Initialize boolean head.

        Args:
            hidden_size: Input embedding dimension d.
            dropout: Dropout probability.
        """
        super().__init__()
        self.classifier = nn.Sequential(
            nn.Linear(hidden_size, hidden_size // 2),
            nn.LayerNorm(hidden_size // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size // 2, 1),
        )

    def forward(self, context_state: torch.Tensor) -> torch.Tensor:
        """Compute binary scalar logit.

        Args:
            context_state: Pooled state representation with shape (batch_size, hidden_size).

        Returns:
            torch.Tensor: Binary logits with shape (batch_size,).
        """
        logits = self.classifier(context_state)  # (B, 1)
        return logits.squeeze(-1)  # (B,)


class ScoreHead(nn.Module):
    """Ordinal multi-level classification and expectation regression head.

    Evaluates state against an ordered ladder (up to max_levels, default 10).
    Outputs level logits, softmax probabilities, and continuous expected scores:
        Score = sum_{i=0}^{m-1} i * P(i)
    """

    def __init__(
        self,
        hidden_size: int,
        max_levels: int = 10,
        dropout: float = 0.1,
    ) -> None:
        """Initialize ordinal score head.

        Args:
            hidden_size: Input embedding dimension d.
            max_levels: Maximum allowable discrete ordinal levels (2 to 10).
            dropout: Dropout probability.
        """
        super().__init__()
        self.max_levels = max_levels
        self.classifier = nn.Sequential(
            nn.Linear(hidden_size, hidden_size // 2),
            nn.LayerNorm(hidden_size // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size // 2, max_levels),
        )

    def forward(
        self,
        context_state: torch.Tensor,
        num_levels: int | torch.Tensor = 10,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Compute level logits, level probabilities, and continuous expected score.

        Args:
            context_state: Pooled state representation with shape (batch_size, hidden_size).
            num_levels: Number of active levels m (2 <= m <= max_levels).
                        Can be an integer or a 1D tensor of shape (batch_size,).

        Returns:
            tuple containing:
                - logits: Masked level logits of shape (batch_size, max_levels).
                - probabilities: Softmax probabilities of shape (batch_size, max_levels).
                - expected_score: Expected continuous score scalar of shape (batch_size,).
        """
        batch_size = context_state.size(0)
        device = context_state.device
        raw_logits = self.classifier(context_state)  # (B, max_levels)

        # Create level mask
        level_indices = torch.arange(self.max_levels, device=device).unsqueeze(0).expand(batch_size, -1)
        if isinstance(num_levels, int):
            mask = level_indices < num_levels
        else:
            mask = level_indices < num_levels.unsqueeze(-1)

        masked_logits = raw_logits.masked_fill(~mask, -1e9)
        probabilities = F.softmax(masked_logits, dim=-1)

        # Expected score: sum_{i=0}^{m-1} i * P(i)
        weights = level_indices.float()
        expected_score = torch.sum(probabilities * weights, dim=-1)

        return masked_logits, probabilities, expected_score
