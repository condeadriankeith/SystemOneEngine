"""SystemOneEngine - Training module.

Provides multi-task dataset loaders, synthetic mock generators,
training orchestration, and Kaggle notebook generation.
"""

from system_one_engine.training.dataset import (
    MultiTaskDataset,
    SimpleVocabTokenizer,
    collate_multi_task_batch,
)
from system_one_engine.training.kaggle_bundle import generate_kaggle_notebook
from system_one_engine.training.mock_generator import (
    generate_mock_dataset,
    save_mock_dataset,
)
from system_one_engine.training.trainer import (
    SystemOneTrainer,
    TaskBatchSampler,
    TrainingHistory,
)

__all__ = [
    "MultiTaskDataset",
    "SimpleVocabTokenizer",
    "collate_multi_task_batch",
    "generate_mock_dataset",
    "save_mock_dataset",
    "SystemOneTrainer",
    "TaskBatchSampler",
    "TrainingHistory",
    "generate_kaggle_notebook",
]
