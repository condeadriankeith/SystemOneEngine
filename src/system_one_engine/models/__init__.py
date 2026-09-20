"""SystemOneEngine - Models module.

Defines PyTorch transformer backbones, dynamic candidate dot-product heads,
ordinal score heads, ONNX export, and the unified SystemOneModel.
"""

from system_one_engine.models.backbone import (
    PoolingStrategy,
    TinyBackbone,
    TransformerBackbone,
)
from system_one_engine.models.export import (
    BooleanScoreHeadsWrapper,
    export_and_quantize_all,
    export_choice_head_to_onnx,
    export_encoder_to_onnx,
    export_heads_to_onnx,
    quantize_to_int8,
)
from system_one_engine.models.heads import (
    BooleanHead,
    DynamicChoiceHead,
    ScoreHead,
)
from system_one_engine.models.system_one_model import SystemOneModel

__all__ = [
    "PoolingStrategy",
    "TransformerBackbone",
    "TinyBackbone",
    "DynamicChoiceHead",
    "BooleanHead",
    "ScoreHead",
    "SystemOneModel",
    "BooleanScoreHeadsWrapper",
    "export_encoder_to_onnx",
    "export_heads_to_onnx",
    "export_choice_head_to_onnx",
    "quantize_to_int8",
    "export_and_quantize_all",
]
