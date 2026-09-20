"""ONNX Export and Post-Training INT8 Quantization utilities.

Exports PyTorch SystemOne models to modular, decoupled ONNX computational graphs:
- encoder.onnx: Transforms token sequences into pooled state vectors h_state.
- heads.onnx: Computes boolean and score level logits from h_state.
- choice_head.onnx: Permutation-equivariant candidate similarity dot-product.

Applies post-training INT8 quantization for sub-25ms CPU execution on consumer hardware.
"""

import logging
from pathlib import Path
from typing import Any
import onnx
import onnxruntime as ort
from onnxruntime.quantization import QuantType, quantize_dynamic
import torch
import torch.nn as nn

from system_one_engine.models.system_one_model import SystemOneModel

logger = logging.getLogger(__name__)


class BooleanScoreHeadsWrapper(nn.Module):
    """Wrapper exporting Boolean and Score classification heads from pooled state."""

    def __init__(self, model: SystemOneModel) -> None:
        super().__init__()
        self.boolean_head = model.boolean_head
        self.score_classifier = model.score_head.classifier

    def forward(self, h_state: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Compute boolean logit and score level logits from pooled state.

        Args:
            h_state: (batch_size, hidden_size)

        Returns:
            tuple of (boolean_logits, score_logits)
        """
        bool_logits = self.boolean_head(h_state)
        score_logits = self.score_classifier(h_state)
        return bool_logits, score_logits


def export_encoder_to_onnx(
    model: SystemOneModel,
    output_path: str | Path,
    sample_seq_len: int = 16,
) -> Path:
    """Export the transformer backbone encoder to ONNX with dynamic axes.

    Args:
        model: SystemOneModel instance.
        output_path: Destination path for .onnx file.
        sample_seq_len: Sample sequence length for tracing dummy tensors.

    Returns:
        Path: Path to written ONNX file.
    """
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    model.eval()
    model.backbone.eval()

    dummy_ids = torch.randint(0, 100, (1, sample_seq_len), dtype=torch.long)
    dummy_mask = torch.ones(1, sample_seq_len, dtype=torch.long)

    dynamic_axes = {
        "input_ids": {0: "batch_size", 1: "seq_len"},
        "attention_mask": {0: "batch_size", 1: "seq_len"},
        "h_state": {0: "batch_size"},
    }

    torch.onnx.export(
        model.backbone,
        (dummy_ids, dummy_mask),
        str(path),
        input_names=["input_ids", "attention_mask"],
        output_names=["h_state"],
        dynamic_axes=dynamic_axes,
        opset_version=17,
        do_constant_folding=True,
        dynamo=False,
    )

    onnx_model = onnx.load(str(path))
    onnx.checker.check_model(onnx_model)
    logger.info("Exported encoder graph to %s", path)
    return path


def export_heads_to_onnx(
    model: SystemOneModel,
    output_path: str | Path,
) -> Path:
    """Export Boolean and Score decision heads to ONNX.

    Args:
        model: SystemOneModel instance.
        output_path: Destination path for .onnx file.

    Returns:
        Path: Path to written ONNX file.
    """
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    model.eval()

    wrapper = BooleanScoreHeadsWrapper(model)
    wrapper.eval()

    dummy_h = torch.randn(1, model.hidden_size)

    dynamic_axes = {
        "h_state": {0: "batch_size"},
        "boolean_logits": {0: "batch_size"},
        "score_logits": {0: "batch_size"},
    }

    torch.onnx.export(
        wrapper,
        dummy_h,
        str(path),
        input_names=["h_state"],
        output_names=["boolean_logits", "score_logits"],
        dynamic_axes=dynamic_axes,
        opset_version=17,
        do_constant_folding=True,
        dynamo=False,
    )

    onnx_model = onnx.load(str(path))
    onnx.checker.check_model(onnx_model)
    logger.info("Exported decision heads graph to %s", path)
    return path


def export_choice_head_to_onnx(
    model: SystemOneModel,
    output_path: str | Path,
    sample_num_cands: int = 3,
) -> Path:
    """Export dynamic choice dot-product head to ONNX.

    Args:
        model: SystemOneModel instance.
        output_path: Destination path for .onnx file.
        sample_num_cands: Number of dummy candidates for tracing.

    Returns:
        Path: Path to written ONNX file.
    """
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    model.eval()
    model.choice_head.eval()

    dummy_h = torch.randn(1, model.hidden_size)
    dummy_cands = torch.randn(1, sample_num_cands, model.hidden_size)

    dynamic_axes = {
        "context_state": {0: "batch_size"},
        "candidate_embeddings": {0: "batch_size", 1: "num_cands"},
        "choice_logits": {0: "batch_size", 1: "num_cands"},
    }

    torch.onnx.export(
        model.choice_head,
        (dummy_h, dummy_cands),
        str(path),
        input_names=["context_state", "candidate_embeddings"],
        output_names=["choice_logits"],
        dynamic_axes=dynamic_axes,
        opset_version=17,
        do_constant_folding=True,
        dynamo=False,
    )

    onnx_model = onnx.load(str(path))
    onnx.checker.check_model(onnx_model)
    logger.info("Exported choice head graph to %s", path)
    return path


def quantize_to_int8(
    model_input_path: str | Path,
    model_output_path: str | Path,
) -> Path:
    """Apply post-training dynamic INT8 quantization for ultra-fast CPU inference.

    Args:
        model_input_path: Path to FP32 ONNX model.
        model_output_path: Path to save INT8 quantized ONNX model.

    Returns:
        Path: Path to quantized model.
    """
    in_path = str(model_input_path)
    out_path = str(model_output_path)

    quantize_dynamic(
        model_input=in_path,
        model_output=out_path,
        weight_type=QuantType.QInt8,
    )
    logger.info("Successfully quantized %s -> %s (INT8)", in_path, out_path)
    return Path(out_path)


def export_and_quantize_all(
    model: SystemOneModel,
    output_dir: str | Path,
) -> dict[str, Path]:
    """Export and quantize all System One sub-graphs to INT8.

    Returns dictionary with paths to quantized models:
    - 'encoder': Path to encoder INT8 model
    - 'heads': Path to boolean/score heads INT8 model
    - 'choice_head': Path to choice head INT8 model
    """
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    enc_fp32 = out_dir / "encoder.onnx"
    heads_fp32 = out_dir / "heads.onnx"
    ch_fp32 = out_dir / "choice_head.onnx"

    export_encoder_to_onnx(model, enc_fp32)
    export_heads_to_onnx(model, heads_fp32)
    export_choice_head_to_onnx(model, ch_fp32)

    enc_int8 = out_dir / "encoder_int8.onnx"
    heads_int8 = out_dir / "heads_int8.onnx"
    ch_int8 = out_dir / "choice_head_int8.onnx"

    quantize_to_int8(enc_fp32, enc_int8)
    quantize_to_int8(heads_fp32, heads_int8)
    quantize_to_int8(ch_fp32, ch_int8)

    return {
        "encoder": enc_int8,
        "heads": heads_int8,
        "choice_head": ch_int8,
    }
