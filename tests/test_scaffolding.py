"""Tests for verifying Phase 1 environment scaffolding and package structure."""

import importlib
import pytest


def test_packages_importable():
    """Verify that all core third-party dependencies are installed and importable."""
    required_packages = [
        "torch",
        "transformers",
        "onnxruntime",
        "onnx",
        "pydantic",
        "sklearn",
        "fastapi",
        "uvicorn",
        "httpx",
        "numpy",
        "scipy",
    ]
    for package_name in required_packages:
        mod = importlib.import_module(package_name)
        assert mod is not None, f"Failed to import package: {package_name}"


def test_system_one_engine_submodules():
    """Verify all internal module packages exist and can be imported."""
    submodules = [
        "system_one_engine",
        "system_one_engine.core",
        "system_one_engine.models",
        "system_one_engine.adapters",
        "system_one_engine.training",
        "system_one_engine.server",
        "system_one_engine.client",
    ]
    for submodule in submodules:
        mod = importlib.import_module(submodule)
        assert mod is not None, f"Failed to import submodule: {submodule}"


def test_pydantic_v2():
    """Verify Pydantic version is 2.x as required by system standards."""
    import pydantic

    major_version = int(pydantic.__version__.split(".")[0])
    assert major_version >= 2, f"Pydantic version must be >= 2.0.0, found {pydantic.__version__}"


def test_torch_cpu_tensor_operations():
    """Verify PyTorch can perform basic CPU tensor operations without error."""
    import torch

    x = torch.randn(4, 16)
    w = torch.randn(16, 8)
    y = torch.matmul(x, w)
    assert y.shape == (4, 8)
