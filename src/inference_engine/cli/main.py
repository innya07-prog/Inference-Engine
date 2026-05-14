"""Command-line entrypoint (``python -m inference_engine.cli.main`` or ``inference-engine``)."""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path
from typing import Sequence

import numpy as np
import torch

from inference_engine import __version__
from inference_engine.benchmarks.onnx_benchmark import OnnxBenchmarkError, benchmark_backend, write_benchmark_json
from inference_engine.builders.trt_engine_builder import TensorRTBuildError, build_tensorrt_engine
from inference_engine.exporters.onnx_exporter import OnnxExportError, export_to_onnx
from inference_engine.integration.resnet18_pipeline import run_resnet18_integration_pipeline
from inference_engine.models.test_mlp import TestMLP
from inference_engine.runtimes.onnx_runner import OnnxRunError, run_onnx_cli
from inference_engine.runtimes.tensorrt_runtime import TensorRTRuntime, TensorRTRuntimeError
from inference_engine.validators.onnx_parity import ParityValidationError, validate_onnx_parity
from inference_engine.validators.trt_parity import TensorRTParityValidationError, validate_trt_parity

logger = logging.getLogger(__name__)


def _parse_csv_shape(spec: str) -> tuple[int, ...]:
    parts = [p.strip() for p in spec.split(",") if p.strip()]
    if not parts:
        raise ValueError("Shape string is empty.")
    return tuple(int(p) for p in parts)


def _optional_csv_shape(spec: str | None) -> tuple[int, ...] | None:
    if spec is None or not str(spec).strip():
        return None
    return _parse_csv_shape(str(spec))


def configure_logging(verbose: int) -> None:
    level = logging.DEBUG if verbose >= 2 else logging.INFO
    logging.basicConfig(
        level=level,
        format="%(levelname)s %(name)s: %(message)s",
        force=True,
    )


def cmd_info(_args: argparse.Namespace) -> int:
    logger.info("inference-engine %s", __version__)
    print(f"inference-engine {__version__}")
    print(f"python {sys.version.split()[0]}")
    print(f"torch {torch.__version__}")
    return 0


def cmd_export(args: argparse.Namespace) -> int:
    out = Path(args.output_onnx).expanduser().resolve()
    ckpt = Path(args.checkpoint).expanduser().resolve()

    example_shape = tuple(int(x.strip()) for x in args.example_shape.split(",") if x.strip())
    if len(example_shape) < 1:
        logger.error("example_shape must list at least one dimension.")
        return 2

    x = torch.randn(*example_shape, device=args.device)

    module: torch.nn.Module | None = None
    if args.test_mlp:
        module = TestMLP()

    try:
        export_to_onnx(
            ckpt,
            out,
            example_input=x,
            module=module,
            dynamic_batch=not args.no_dynamic_batch,
            device=args.device,
            validate=not args.no_validate,
        )
    except (OnnxExportError, FileNotFoundError) as exc:
        logger.error("%s", exc)
        return 1
    except Exception:
        logger.exception("Unexpected error during ONNX export.")
        return 1

    logger.info("Export complete: %s", out)
    return 0


def cmd_run_onnx(args: argparse.Namespace) -> int:
    model = Path(args.model).expanduser().resolve()
    npy = Path(args.numpy).expanduser().resolve() if args.numpy else None
    try:
        summary = run_onnx_cli(
            model,
            numpy_path=npy,
            shape_spec=args.random_shape,
            dtype=args.dtype,
        )
    except OnnxRunError as exc:
        logger.error("%s", exc)
        return 1
    except Exception:
        logger.exception("Unexpected error during ONNX Runtime inference.")
        return 1

    print(json.dumps(summary, indent=2))
    return 0


def cmd_verify_onnx(args: argparse.Namespace) -> int:
    ckpt = Path(args.checkpoint).expanduser().resolve()
    onnx = Path(args.model).expanduser().resolve()

    module: torch.nn.Module | None = None
    if args.test_mlp:
        module = TestMLP()

    try:
        result = validate_onnx_parity(
            ckpt,
            onnx,
            args.shape,
            module=module,
            device=args.device,
            seed=args.seed,
            atol=args.atol,
            rtol=args.rtol,
            print_report=True,
        )
    except (ParityValidationError, FileNotFoundError) as exc:
        logger.error("%s", exc)
        return 1
    except Exception:
        logger.exception("Unexpected error during ONNX parity validation.")
        return 1

    return 0 if result.passed else 1


def cmd_verify_trt(args: argparse.Namespace) -> int:
    ckpt = Path(args.checkpoint).expanduser().resolve()
    eng = Path(args.engine).expanduser().resolve()

    module: torch.nn.Module | None = None
    if args.test_mlp:
        module = TestMLP()

    try:
        result = validate_trt_parity(
            ckpt,
            eng,
            args.shape,
            module=module,
            torch_device=args.torch_device,
            trt_device=args.device,
            seed=args.seed,
            atol=args.atol,
            rtol=args.rtol,
            print_report=True,
            output_tensor=args.output_tensor,
        )
    except (TensorRTParityValidationError, FileNotFoundError) as exc:
        logger.error("%s", exc)
        return 1
    except Exception:
        logger.exception("Unexpected error during TensorRT parity validation.")
        return 1

    return 0 if result.passed else 1


def cmd_integration_test_resnet18(args: argparse.Namespace) -> int:
    try:
        import torchvision  # noqa: F401
    except ImportError:
        logger.error("torchvision is required. Install: pip install -e \".[integration]\"")
        return 2

    out = Path(args.output_dir).expanduser().resolve()
    try:
        report = run_resnet18_integration_pipeline(
            out,
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
    print(json.dumps({"validation_metrics": summary.get("validation_metrics")}, indent=2))
    print(json.dumps({"benchmark_metrics": summary.get("benchmark_metrics")}, indent=2))
    print(
        json.dumps(
            {
                "pass_fail": {
                    k: summary[k]
                    for k in ("pytorch_vs_onnx", "pytorch_vs_tensorrt", "overall")
                    if k in summary
                }
            },
            indent=2,
        )
    )
    print(json.dumps({"artifacts": report.get("artifacts")}, indent=2))

    return 0 if summary.get("overall") == "PASS" else 1


def cmd_benchmark_onnx(args: argparse.Namespace) -> int:
    model = Path(args.model).expanduser().resolve()
    try:
        report = benchmark_backend(
            model,
            args.shape,
            warmup_iterations=args.warmup,
            timed_iterations=args.iterations,
            seed=args.seed,
            print_report=True,
        )
    except (OnnxBenchmarkError, FileNotFoundError) as exc:
        logger.error("%s", exc)
        return 1
    except Exception:
        logger.exception("Unexpected error during ONNX benchmark.")
        return 1

    if args.output_json is not None:
        try:
            write_benchmark_json(report, args.output_json)
        except OSError as exc:
            logger.error("Failed to write JSON: %s", exc)
            return 1

    return 0


def cmd_build_trt(args: argparse.Namespace) -> int:
    onnx = Path(args.model).expanduser().resolve()
    engine = Path(args.engine).expanduser().resolve()

    fp16: bool | None
    if args.fp16_mode == "auto":
        fp16 = None
    elif args.fp16_mode == "on":
        fp16 = True
    else:
        fp16 = False

    try:
        build_tensorrt_engine(
            onnx,
            engine,
            min_shape=_optional_csv_shape(args.min_shape),
            opt_shape=_optional_csv_shape(args.opt_shape),
            max_shape=_optional_csv_shape(args.max_shape),
            input_name=args.input_name,
            fp16=fp16,
            workspace_mib=int(args.workspace_mib),
            log_level=args.trt_log_level,
        )
    except (TensorRTBuildError, FileNotFoundError) as exc:
        logger.error("%s", exc)
        return 1
    except Exception:
        logger.exception("Unexpected error during TensorRT engine build.")
        return 1

    logger.info("TensorRT engine build complete: %s", engine)
    return 0


def cmd_run_trt(args: argparse.Namespace) -> int:
    engine = Path(args.engine).expanduser().resolve()
    try:
        shape = _parse_csv_shape(args.shape)
    except ValueError as exc:
        logger.error("Invalid --shape: %s", exc)
        return 2

    device = torch.device(args.device)
    try:
        rt = TensorRTRuntime(engine, device=device)
        in_names = rt.input_tensor_names
        if len(in_names) != 1:
            logger.error(
                "run-trt currently supports engines with exactly one input tensor; found %s: %s",
                len(in_names),
                in_names,
            )
            return 2
        rng = np.random.default_rng(int(args.seed))
        x = rng.standard_normal(shape, dtype=np.float32)
        x = np.ascontiguousarray(x)
        outputs = rt.infer({in_names[0]: x})
    except (TensorRTRuntimeError, FileNotFoundError) as exc:
        logger.error("%s", exc)
        return 1
    except Exception:
        logger.exception("Unexpected error during TensorRT inference.")
        return 1

    summary = {
        name: {
            "shape": list(arr.shape),
            "dtype": str(arr.dtype),
            "min": float(np.min(arr)) if arr.size else None,
            "max": float(np.max(arr)) if arr.size else None,
            "mean": float(np.mean(arr)) if arr.size else None,
        }
        for name, arr in outputs.items()
    }
    print(json.dumps(summary, indent=2))
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="inference-engine",
        description="PyTorch → ONNX / TensorRT inference engine CLI.",
    )
    parser.add_argument(
        "-v",
        "--verbose",
        action="count",
        default=0,
        help="Increase logging verbosity (-v INFO, -vv DEBUG).",
    )

    sub = parser.add_subparsers(dest="command", required=True)

    p_info = sub.add_parser("info", help="Print version and runtime information.")
    p_info.set_defaults(handler=cmd_info)

    p_exp = sub.add_parser("export", help="Export a PyTorch checkpoint to ONNX (torch.export path).")
    p_exp.add_argument("--checkpoint", required=True, type=Path, help="Path to .pt / .pth (Module or state_dict).")
    p_exp.add_argument("--output-onnx", required=True, type=Path, help="Destination .onnx path.")
    p_exp.add_argument(
        "--example-shape",
        default="2,128",
        help='Comma-separated example input shape, e.g. "2,128" (default avoids batch=1 specialization).',
    )
    p_exp.add_argument("--device", default="cpu", help="Torch device for export (default cpu).")
    p_exp.add_argument(
        "--test-mlp",
        action="store_true",
        help="When the checkpoint is a state_dict, load weights into packaged TestMLP.",
    )
    p_exp.add_argument("--no-dynamic-batch", action="store_true", help="Disable dynamic batch Dim on export.")
    p_exp.add_argument("--no-validate", action="store_true", help="Skip onnx.checker validation after export.")
    p_exp.set_defaults(handler=cmd_export)

    p_run = sub.add_parser("run-onnx", help="Run a single ONNX Runtime forward pass.")
    p_run.add_argument("--model", required=True, type=Path, help="Path to .onnx model.")
    g = p_run.add_mutually_exclusive_group(required=True)
    g.add_argument("--numpy", type=Path, help="Path to .npy for the first model input.")
    g.add_argument("--random-shape", help='Synthetic normal input shape, e.g. "1,128".')
    p_run.add_argument("--dtype", default="float32", help="Numpy dtype for input data (default float32).")
    p_run.set_defaults(handler=cmd_run_onnx)

    p_ver = sub.add_parser(
        "verify-onnx",
        help="Compare PyTorch checkpoint output to ONNX Runtime on the same random input.",
    )
    p_ver.add_argument("--checkpoint", required=True, type=Path, help="Path to .pt / .pth (Module or state_dict).")
    p_ver.add_argument("--model", required=True, type=Path, help="Path to exported .onnx model.")
    p_ver.add_argument(
        "--shape",
        required=True,
        help='Random input shape, comma-separated, e.g. "1,128".',
    )
    p_ver.add_argument("--device", default="cpu", help="Torch device for the reference forward (default cpu).")
    p_ver.add_argument(
        "--test-mlp",
        action="store_true",
        help="When the checkpoint is a state_dict, load weights into packaged TestMLP.",
    )
    p_ver.add_argument("--seed", type=int, default=0, help="NumPy RNG seed for the shared input (default 0).")
    p_ver.add_argument("--atol", type=float, default=1e-5, help="numpy.allclose absolute tolerance (default 1e-5).")
    p_ver.add_argument("--rtol", type=float, default=1e-4, help="numpy.allclose relative tolerance (default 1e-4).")
    p_ver.set_defaults(handler=cmd_verify_onnx)

    p_bench = sub.add_parser(
        "benchmark-onnx",
        help="Benchmark ONNX Runtime inference (CPUExecutionProvider, latency and throughput).",
    )
    p_bench.add_argument("--model", required=True, type=Path, help="Path to .onnx model.")
    p_bench.add_argument(
        "--shape",
        required=True,
        help='Random input shape including batch, comma-separated, e.g. "1,128".',
    )
    p_bench.add_argument("--warmup", type=int, default=10, help="Warmup iterations (default 10).")
    p_bench.add_argument("--iterations", type=int, default=100, help="Timed iterations (default 100).")
    p_bench.add_argument("--seed", type=int, default=0, help="RNG seed for input tensor (default 0).")
    p_bench.add_argument(
        "--output-json",
        type=Path,
        default=None,
        help="Optional path to write benchmark report as JSON.",
    )
    p_bench.set_defaults(handler=cmd_benchmark_onnx)

    p_btrt = sub.add_parser(
        "build-trt",
        help="Build a serialized TensorRT 10.x engine from ONNX (explicit batch, dynamic profiles, optional FP16).",
    )
    p_btrt.add_argument("--model", required=True, type=Path, help="Path to input .onnx model.")
    p_btrt.add_argument("--engine", required=True, type=Path, help="Output path for the serialized .engine file.")
    p_btrt.add_argument(
        "--min-shape",
        default=None,
        help='Optimization profile min shape, e.g. "1,128" (omit all three to auto-detect from ONNX).',
    )
    p_btrt.add_argument("--opt-shape", default=None, help='Profile opt shape, e.g. "4,128".')
    p_btrt.add_argument("--max-shape", default=None, help='Profile max shape, e.g. "32,128".')
    p_btrt.add_argument("--input-name", default=None, help="Override ONNX input tensor name for the profile.")
    p_btrt.add_argument(
        "--fp16",
        dest="fp16_mode",
        default="auto",
        choices=["auto", "on", "off"],
        help='FP16: "auto" enables when GPU reports fast FP16 (default), "on" forces, "off" disables.',
    )
    p_btrt.add_argument("--workspace-mib", type=int, default=1024, help="TensorRT WORKSPACE memory pool limit in MiB.")
    p_btrt.add_argument(
        "--trt-log-level",
        default="WARNING",
        help="TensorRT logger level name (e.g. WARNING, INFO, VERBOSE).",
    )
    p_btrt.set_defaults(handler=cmd_build_trt)

    p_rtrt = sub.add_parser(
        "run-trt",
        help="Run one TensorRT inference on the GPU (execute_async_v3 + CUDA stream sync).",
    )
    p_rtrt.add_argument("--engine", required=True, type=Path, help="Path to serialized .engine file.")
    p_rtrt.add_argument(
        "--shape",
        required=True,
        help='Random float32 input shape (comma-separated), e.g. "1,128". Must be within the engine profile.',
    )
    p_rtrt.add_argument("--seed", type=int, default=0, help="RNG seed for random input (default 0).")
    p_rtrt.add_argument("--device", default="cuda:0", help="Torch CUDA device, e.g. cuda:0 (default cuda:0).")
    p_rtrt.set_defaults(handler=cmd_run_trt)

    p_vtrt = sub.add_parser(
        "verify-trt",
        help="Compare PyTorch checkpoint output to TensorRT on the same random input (FP32 compare space).",
    )
    p_vtrt.add_argument("--checkpoint", required=True, type=Path, help="Path to .pt / .pth (Module or state_dict).")
    p_vtrt.add_argument("--engine", required=True, type=Path, help="Path to serialized TensorRT .engine file.")
    p_vtrt.add_argument(
        "--shape",
        required=True,
        help='Random float32 input shape (comma-separated), e.g. "1,128" or "8,128" for batch 8.',
    )
    p_vtrt.add_argument(
        "--device",
        default="cuda:0",
        help="CUDA device for TensorRT runtime (default cuda:0). Also used for Torch unless --torch-device is set.",
    )
    p_vtrt.add_argument(
        "--torch-device",
        default=None,
        help="Optional override for the Torch reference device (defaults to --device).",
    )
    p_vtrt.add_argument(
        "--test-mlp",
        action="store_true",
        help="When the checkpoint is a state_dict, load weights into packaged TestMLP.",
    )
    p_vtrt.add_argument("--seed", type=int, default=0, help="NumPy RNG seed for the shared input (default 0).")
    p_vtrt.add_argument("--atol", type=float, default=1e-5, help="numpy.allclose absolute tolerance (default 1e-5).")
    p_vtrt.add_argument("--rtol", type=float, default=1e-4, help="numpy.allclose relative tolerance (default 1e-4).")
    p_vtrt.add_argument(
        "--output-tensor",
        default=None,
        help="TRT output tensor name to compare when the engine has multiple outputs (defaults to first).",
    )
    p_vtrt.set_defaults(handler=cmd_verify_trt)

    p_r18 = sub.add_parser(
        "integration-test-resnet18",
        help="End-to-end ResNet18: checkpoint, ONNX, ORT/TRT, parity, benchmarks, JSON reports.",
    )
    p_r18.add_argument(
        "--output-dir",
        type=Path,
        default=Path("integration_reports/resnet18"),
        help="Directory for .pt, .onnx, .engine, validation.json, benchmark.json, summary.json.",
    )
    p_r18.add_argument("--warmup", type=int, default=10, help="Benchmark warmup iterations (default 10).")
    p_r18.add_argument("--iterations", type=int, default=50, help="Benchmark timed iterations (default 50).")
    p_r18.add_argument("--seed", type=int, default=0, help="RNG seed for parity and benchmarks (default 0).")
    p_r18.add_argument("--atol", type=float, default=1e-4, help="Parity absolute tolerance (default 1e-4).")
    p_r18.add_argument("--rtol", type=float, default=1e-3, help="Parity relative tolerance (default 1e-3).")
    p_r18.add_argument(
        "--export-device",
        default="cpu",
        help="Torch device for ONNX export example tensor (default cpu).",
    )
    p_r18.add_argument(
        "--onnx-parity-device",
        default="cpu",
        help="Torch device for PyTorch vs ONNX parity (default cpu).",
    )
    p_r18.add_argument(
        "--trt-device",
        default="cuda:0",
        help="CUDA device for TensorRT when CUDA + TensorRT are available (default cuda:0).",
    )
    p_r18.set_defaults(handler=cmd_integration_test_resnet18)

    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(list(argv) if argv is not None else None)
    configure_logging(args.verbose)

    handler = args.handler
    try:
        return int(handler(args))
    except KeyboardInterrupt:
        logger.warning("Interrupted.")
        return 130


if __name__ == "__main__":
    raise SystemExit(main())
