"""Integration-style pipelines (e.g. ResNet18 end-to-end)."""

from inference_engine.integration.resnet18_pipeline import (
    INPUT_SHAPE,
    INPUT_SHAPE_STR,
    run_resnet18_integration_pipeline,
)

__all__ = ["INPUT_SHAPE", "INPUT_SHAPE_STR", "run_resnet18_integration_pipeline"]
