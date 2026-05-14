"""Pytest integration tests for the ResNet18 end-to-end pipeline."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from inference_engine.integration.resnet18_pipeline import run_resnet18_integration_pipeline


pytest.importorskip("torchvision")


@pytest.mark.integration
def test_resnet18_pipeline_writes_reports(tmp_path: Path) -> None:
    report = run_resnet18_integration_pipeline(
        tmp_path,
        warmup_iterations=3,
        timed_iterations=10,
        seed=0,
        atol=1e-3,
        rtol=5e-3,
    )

    assert (tmp_path / "integration_report.json").is_file()
    assert (tmp_path / "validation.json").is_file()
    assert (tmp_path / "benchmark.json").is_file()
    assert (tmp_path / "summary.json").is_file()
    assert (tmp_path / "resnet18.pt").is_file()
    assert (tmp_path / "resnet18.onnx").is_file()

    val = json.loads((tmp_path / "validation.json").read_text(encoding="utf-8"))
    assert val["pytorch_vs_onnx"]["status"] == "PASS"

    bench = json.loads((tmp_path / "benchmark.json").read_text(encoding="utf-8"))
    assert bench["onnxruntime"]["backend"] == "onnxruntime"
    assert "comparison" in bench

    assert report["status"] == "completed"
    assert report["summary"]["overall"] == "PASS"
