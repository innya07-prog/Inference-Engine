"""Run ResNet18 integration pipeline: ``python -m inference_engine.integration <output_dir>``."""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

from inference_engine.integration.resnet18_pipeline import run_resnet18_integration_pipeline

logger = logging.getLogger(__name__)


def main() -> int:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "output_dir",
        nargs="?",
        default="integration_reports/resnet18",
        type=Path,
        help="Directory for checkpoints, ONNX/engine artifacts, and JSON reports.",
    )
    p.add_argument("--warmup", type=int, default=10)
    p.add_argument("--iterations", type=int, default=50)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--atol", type=float, default=1e-4)
    p.add_argument("--rtol", type=float, default=1e-3)
    p.add_argument("--export-device", default="cpu")
    p.add_argument("--onnx-parity-device", default="cpu")
    p.add_argument("--trt-device", default="cuda:0")
    args = p.parse_args()

    try:
        import torchvision  # noqa: F401
    except ImportError:
        logger.error("torchvision is required. Install: pip install -e \".[integration]\"")
        return 2

    try:
        report = run_resnet18_integration_pipeline(
            args.output_dir,
            warmup_iterations=args.warmup,
            timed_iterations=args.iterations,
            seed=args.seed,
            atol=args.atol,
            rtol=args.rtol,
            export_device=args.export_device,
            onnx_parity_device=args.onnx_parity_device,
            trt_device=args.trt_device,
        )
    except Exception:
        logger.exception("ResNet18 integration pipeline failed.")
        return 1

    summary = report.get("summary", {})
    print(json.dumps({"pass_fail": {k: summary.get(k) for k in ("pytorch_vs_onnx", "pytorch_vs_tensorrt", "overall")}}, indent=2))
    return 0 if summary.get("overall") == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
