"""Transformer backbone modules for System One Decision Engine.

Provides:
- PoolingStrategy: Supported pooling strategies (mean, cls).
- TransformerBackbone: Wraps Hugging Face encoder backbones (ModernBERT / DeBERTa-v3).
- TinyBackbone: Fast, lightweight 2-layer transformer for local CPU unit testing.
"""

from enum import Enum
from typing import Literal
import torch
import torch.nn as nn
from transformers import AutoConfig, AutoModel


class PoolingStrategy(str, Enum):
    """Pooling strategies for converting token sequences to state vectors."""

    CLS = "cls"
    MEAN = "mean"


class TransformerBackbone(nn.Module):
    """Wraps a pre-trained bidirectional transformer encoder (ModernBERT, DeBERTa-v3, BERT).

    Attributes:
        encoder: Hugging Face AutoModel encoder instance.
        hidden_size: Dimensionality d of the pooled output representation.
        pooling: PoolingStrategy (CLS or MEAN).
    """

    def __init__(
        self,
        model_name_or_path: str = "answerdotai/ModernBERT-base",
        pooling: PoolingStrategy | str = PoolingStrategy.MEAN,
    ) -> None:
        """Initialize transformer backbone.

        Args:
            model_name_or_path: Hugging Face model identifier or local checkpoint path.
            pooling: Pooling strategy to extract state vector ('mean' or 'cls').
        """
        super().__init__()
        self.pooling = PoolingStrategy(pooling)
        self.encoder = AutoModel.from_pretrained(model_name_or_path)
        self.hidden_size: int = self.encoder.config.hidden_size

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
    ) -> torch.Tensor:
        """Encode tokenized input sequences into pooled state representations h_state.

        Args:
            input_ids: Tensor of token IDs with shape (batch_size, seq_len).
            attention_mask: Binary mask with shape (batch_size, seq_len).

        Returns:
            torch.Tensor: Pooled state embeddings with shape (batch_size, hidden_size).
        """
        outputs = self.encoder(input_ids=input_ids, attention_mask=attention_mask)
        token_embeddings = outputs.last_hidden_state  # (B, L, D)

        if self.pooling == PoolingStrategy.CLS:
            # First token [CLS]
            return token_embeddings[:, 0, :]

        # Mean pooling with attention mask
        mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
        sum_embeddings = torch.sum(token_embeddings * mask_expanded, dim=1)
        sum_mask = torch.clamp(mask_expanded.sum(dim=1), min=1e-9)
        return sum_embeddings / sum_mask


class TinyBackbone(nn.Module):
    """Lightweight 2-layer transformer encoder for fast local unit testing on CPU.

    Provides identical output shapes without requiring multi-hundred megabyte HF model downloads.
    """

    def __init__(
        self,
        vocab_size: int = 1000,
        hidden_size: int = 64,
        num_layers: int = 2,
        num_heads: int = 4,
        max_seq_len: int = 128,
        pooling: PoolingStrategy | str = PoolingStrategy.MEAN,
    ) -> None:
        """Initialize tiny transformer backbone.

        Args:
            vocab_size: Size of vocabulary.
            hidden_size: Model dimension d.
            num_layers: Number of transformer encoder layers.
            num_heads: Number of attention heads.
            max_seq_len: Maximum sequence length.
            pooling: Pooling strategy.
        """
        super().__init__()
        self.hidden_size = hidden_size
        self.pooling = PoolingStrategy(pooling)
        self.token_embeddings = nn.Embedding(vocab_size, hidden_size)
        self.position_embeddings = nn.Embedding(max_seq_len, hidden_size)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_size,
            nhead=num_heads,
            dim_feedforward=hidden_size * 2,
            batch_first=True,
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.layer_norm = nn.LayerNorm(hidden_size)

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
    ) -> torch.Tensor:
        """Encode tokenized input sequences into pooled state representations.

        Args:
            input_ids: Tensor of token IDs with shape (batch_size, seq_len).
            attention_mask: Binary mask with shape (batch_size, seq_len).

        Returns:
            torch.Tensor: Pooled state embeddings with shape (batch_size, hidden_size).
        """
        batch_size, seq_len = input_ids.shape
        positions = torch.arange(seq_len, device=input_ids.device).unsqueeze(0).expand(batch_size, -1)

        x = self.token_embeddings(input_ids) + self.position_embeddings(positions)
        # In PyTorch TransformerEncoder, src_key_padding_mask requires True for padded tokens
        padding_mask = attention_mask == 0

        token_embeddings = self.encoder(x, src_key_padding_mask=padding_mask)
        token_embeddings = self.layer_norm(token_embeddings)

        if self.pooling == PoolingStrategy.CLS:
            return token_embeddings[:, 0, :]

        mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
        sum_embeddings = torch.sum(token_embeddings * mask_expanded, dim=1)
        sum_mask = torch.clamp(mask_expanded.sum(dim=1), min=1e-9)
        return sum_embeddings / sum_mask
