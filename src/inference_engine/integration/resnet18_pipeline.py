"""Complete ResNet18 integration pipeline (checkpoint → ONNX → ORT → TRT → parity → benchmarks)."""

from __future__ import annotations

import json
import logging
import sys
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn as nn

from inference_engine.benchmarks.onnx_benchmark import OnnxBenchmarkError, benchmark_backend
from inference_engine.benchmarks.tensorrt_benchmark import TensorRTBenchmarkError, benchmark_tensorrt_backend
from inference_engine.builders.trt_engine_builder import TensorRTBuildError, build_tensorrt_engine
from inference_engine.exporters.onnx_exporter import OnnxExportError, export_to_onnx
from inference_engine.runtimes.onnx_runner import run_onnx_once
from inference_engine.runtimes.tensorrt_runtime import TensorRTRuntime, TensorRTRuntimeError
from inference_engine.validators.onnx_parity import ParityValidationError, validate_onnx_parity
from inference_engine.validators.trt_parity import TensorRTParityValidationError, validate_trt_parity

logger = logging.getLogger(__name__)

INPUT_SHAPE: tuple[int, ...] = (1, 3, 224, 224)
INPUT_SHAPE_STR = "1,3,224,224"


def _benchmark_comparison(onnx_bench: dict[str, Any] | None, trt_bench: dict[str, Any] | None) -> dict[str, Any]:
    out: dict[str, Any] = {
        "onnx_throughput_samples_per_sec": None,
        "tensorrt_throughput_samples_per_sec": None,
        "speedup_tensorrt_vs_onnx": None,
    }
    try:
        o = (
            float(onnx_bench["throughput_samples_per_sec"])
            if onnx_bench and "throughput_samples_per_sec" in onnx_bench
            else None
        )
        t = (
            float(trt_bench["throughput_samples_per_sec"])
            if trt_bench and "throughput_samples_per_sec" in trt_bench
            else None
        )
    except (TypeError, ValueError, KeyError):
        return out
    out["onnx_throughput_samples_per_sec"] = o
    out["tensorrt_throughput_samples_per_sec"] = t
    if o is not None and t is not None and o > 0:
        out["speedup_tensorrt_vs_onnx"] = float(t / o)
    return out


def _tensorrt_import_ok() -> bool:
    try:
        import tensorrt  # noqa: F401

        return True
    except Exception:
        return False


def _build_resnet18() -> nn.Module:
    """ResNet18 without ImageNet weights (``weights=None`` / legacy ``pretrained=False``)."""
    try:
        from torchvision.models import resnet18
    except ImportError as exc:
        raise RuntimeError(
            "torchvision is required. Install: pip install -e \".[integration]\""
        ) from exc

    torch.manual_seed(0)
    np.random.seed(0)
    try:
        model = resnet18(weights=None)
    except TypeError:
        model = resnet18(pretrained=False)  # type: ignore[call-arg]
    model.eval()
    return model


def _parity_status_block(d: dict[str, Any] | None) -> str:
    if d is None:
        return "SKIP"
    st = d.get("status")
    if st in ("PASS", "FAIL"):
        return str(st)
    return "FAIL"


def _compute_summary(
    *,
    onnx_parity: dict[str, Any] | None,
    trt_parity: dict[str, Any] | None,
    skipped: dict[str, str],
) -> dict[str, Any]:
    pytorch_vs_onnx = _parity_status_block(onnx_parity)
    if trt_parity is None and skipped.get("tensorrt"):
        pytorch_vs_tensorrt = "SKIP"
    else:
        pytorch_vs_tensorrt = _parity_status_block(trt_parity)

    overall_ok = pytorch_vs_onnx == "PASS" and pytorch_vs_tensorrt in ("PASS", "SKIP")
    return {
        "pytorch_vs_onnx": pytorch_vs_onnx,
        "pytorch_vs_tensorrt": pytorch_vs_tensorrt,
        "overall": "PASS" if overall_ok else "FAIL",
    }


def _onnx_first_input_name(onnx_path: Path) -> str:
    import onnxruntime as ort

    sess = ort.InferenceSession(onnx_path.as_posix(), providers=ort.get_available_providers())
    return sess.get_inputs()[0].name


def run_resnet18_integration_pipeline(
    output_dir: str | Path,
    *,
    warmup_iterations: int = 10,
    timed_iterations: int = 50,
    seed: int = 0,
    export_device: str = "cpu",
    onnx_parity_device: str = "cpu",
    trt_device: str = "cuda:0",
    atol: float = 1e-4,
    rtol: float = 1e-3,
) -> dict[str, Any]:
    """
    Full ResNet18 path using exporters, builders, runtimes, and validators.

    Artifacts: ``resnet18.pt``, ``resnet18.onnx``, ``resnet18.engine`` (if TRT builds),
    ``validation.json``, ``benchmark.json``, ``summary.json``, ``integration_report.json``.
    """
    out = Path(output_dir).expanduser().resolve()
    out.mkdir(parents=True, exist_ok=True)

    ckpt_path = out / "resnet18.pt"
    onnx_path = out / "resnet18.onnx"
    engine_path = out / "resnet18.engine"

    stages: dict[str, Any] = {}
    skipped: dict[str, str] = {}

    meta: dict[str, Any] = {
        "event": "resnet18_integration",
        "started_at_utc": datetime.now(timezone.utc).isoformat(),
        "torch_version": torch.__version__,
        "python": sys.version.split()[0],
        "model": "torchvision.models.resnet18",
        "pretrained": False,
        "input_shape": list(INPUT_SHAPE),
        "export_device": export_device,
        "onnx_parity_device": onnx_parity_device,
        "trt_device": trt_device,
        "atol": atol,
        "rtol": rtol,
        "warmup_iterations": warmup_iterations,
        "timed_iterations": timed_iterations,
        "seed": seed,
        "stages": stages,
    }

    try:
        model = _build_resnet18()
        torch.save(model, ckpt_path)
        stages["1_generate_checkpoint"] = {"ok": True, "path": str(ckpt_path)}
        logger.info("Stage 1: saved ResNet18 checkpoint (pretrained=False) -> %s", ckpt_path)
    except Exception:
        logger.exception("Stage 1: checkpoint failed")
        stages["1_generate_checkpoint"] = {"ok": False}
        meta["status"] = "failed"
        meta["stages"] = stages
        _finalize_writes(out, meta, None, None, None)
        raise

    example = torch.randn(INPUT_SHAPE, device=export_device, dtype=torch.float32)
    try:
        export_to_onnx(
            ckpt_path,
            onnx_path,
            example_input=example,
            module=None,
            dynamic_batch=False,
            device=export_device,
            validate=True,
        )
        stages["2_export_onnx"] = {"ok": True, "path": str(onnx_path)}
        logger.info("Stage 2: exported ONNX -> %s", onnx_path)
    except (OnnxExportError, FileNotFoundError) as exc:
        logger.error("Stage 2: ONNX export failed: %s", exc)
        stages["2_export_onnx"] = {"ok": False, "error": str(exc)}
        meta["status"] = "failed_onnx_export"
        meta["error"] = str(exc)
        meta["stages"] = stages
        _finalize_writes(out, meta, None, None, None)
        raise
    except Exception:
        logger.exception("Stage 2: ONNX export unexpected error")
        stages["2_export_onnx"] = {"ok": False}
        meta["status"] = "failed_onnx_export"
        meta["stages"] = stages
        _finalize_writes(out, meta, None, None, None)
        raise

    try:
        in_name = _onnx_first_input_name(onnx_path)
        rng = np.random.default_rng(seed)
        x_np = np.ascontiguousarray(rng.standard_normal(INPUT_SHAPE, dtype=np.float32))
        _ = run_onnx_once(onnx_path, feeds={in_name: x_np})
        stages["3_onnx_runtime_inference"] = {"ok": True, "input_name": in_name}
        logger.info("Stage 3: ONNX Runtime inference OK (input=%s)", in_name)
    except Exception as exc:
        logger.error("Stage 3: ONNX Runtime inference failed: %s", exc)
        stages["3_onnx_runtime_inference"] = {"ok": False, "error": str(exc)}

    cuda_ok = torch.cuda.is_available()
    trt_ok = _tensorrt_import_ok()
    if not cuda_ok:
        skipped["tensorrt"] = "CUDA not available."
    elif not trt_ok:
        skipped["tensorrt"] = "TensorRT Python package not importable."
    else:
        try:
            build_tensorrt_engine(onnx_path, engine_path)
            stages["4_build_tensorrt_engine"] = {"ok": True, "path": str(engine_path)}
            logger.info("Stage 4: TensorRT engine -> %s", engine_path)
        except (TensorRTBuildError, FileNotFoundError) as exc:
            skipped["tensorrt"] = f"engine_build_failed: {exc}"
            stages["4_build_tensorrt_engine"] = {"ok": False, "error": str(exc)}
            logger.error("Stage 4: TensorRT build failed: %s", exc)

    if "tensorrt" not in skipped and engine_path.is_file():
        try:
            rt = TensorRTRuntime(engine_path, device=torch.device(trt_device))
            in_names = rt.input_tensor_names
            rng = np.random.default_rng(seed)
            x_np = np.ascontiguousarray(rng.standard_normal(INPUT_SHAPE, dtype=np.float32))
            outs = rt.infer({in_names[0]: x_np})
            stages["5_tensorrt_runtime_inference"] = {
                "ok": True,
                "input_name": in_names[0],
                "output_shapes": {k: list(v.shape) for k, v in outs.items()},
            }
            logger.info("Stage 5: TensorRT inference OK outputs=%s", list(outs.keys()))
        except (TensorRTRuntimeError, FileNotFoundError) as exc:
            logger.error("Stage 5: TensorRT inference failed: %s", exc)
            stages["5_tensorrt_runtime_inference"] = {"ok": False, "error": str(exc)}
        except Exception:
            logger.exception("Stage 5: TensorRT inference unexpected error")
            stages["5_tensorrt_runtime_inference"] = {"ok": False}
    else:
        stages["5_tensorrt_runtime_inference"] = {"ok": None, "skipped": skipped.get("tensorrt", "n/a")}

    onnx_parity: dict[str, Any] | None = None
    try:
        pr_onnx = validate_onnx_parity(
            ckpt_path,
            onnx_path,
            INPUT_SHAPE_STR,
            module=None,
            device=onnx_parity_device,
            seed=seed,
            atol=atol,
            rtol=rtol,
            print_report=False,
        )
        onnx_parity = asdict(pr_onnx)
        stages["6_validate_pytorch_vs_onnx"] = {"ok": onnx_parity.get("status") == "PASS", "status": onnx_parity.get("status")}
        logger.info("Stage 6: PyTorch vs ONNX -> %s", onnx_parity.get("status"))
    except (ParityValidationError, FileNotFoundError) as exc:
        onnx_parity = {"status": "ERROR", "error": str(exc)}
        stages["6_validate_pytorch_vs_onnx"] = {"ok": False, "error": str(exc)}
        logger.error("Stage 6: parity failed: %s", exc)
    except Exception:
        logger.exception("Stage 6: unexpected")
        onnx_parity = {"status": "ERROR", "error": "unexpected_error"}
        stages["6_validate_pytorch_vs_onnx"] = {"ok": False}

    trt_parity: dict[str, Any] | None = None
    if "tensorrt" not in skipped and engine_path.is_file():
        try:
            pr_trt = validate_trt_parity(
                ckpt_path,
                engine_path,
                INPUT_SHAPE_STR,
                module=None,
                torch_device=trt_device,
                trt_device=trt_device,
                seed=seed,
                atol=atol,
                rtol=rtol,
                print_report=False,
            )
            trt_parity = asdict(pr_trt)
            stages["7_validate_pytorch_vs_tensorrt"] = {"ok": trt_parity.get("status") == "PASS", "status": trt_parity.get("status")}
            logger.info("Stage 7: PyTorch vs TensorRT -> %s", trt_parity.get("status"))
        except (TensorRTParityValidationError, FileNotFoundError) as exc:
            trt_parity = {"status": "ERROR", "error": str(exc)}
            stages["7_validate_pytorch_vs_tensorrt"] = {"ok": False, "error": str(exc)}
            logger.error("Stage 7: TRT parity failed: %s", exc)
        except Exception:
            logger.exception("Stage 7: unexpected")
            trt_parity = {"status": "ERROR", "error": "unexpected_error"}
            stages["7_validate_pytorch_vs_tensorrt"] = {"ok": False}
    else:
        stages["7_validate_pytorch_vs_tensorrt"] = {"ok": None, "skipped": skipped.get("tensorrt", "n/a")}
        logger.info("Stage 7: skipped (%s)", skipped.get("tensorrt", "n/a"))

    onnx_bench: dict[str, Any] | None = None
    try:
        onnx_bench = benchmark_backend(
            onnx_path,
            INPUT_SHAPE_STR,
            warmup_iterations=warmup_iterations,
            timed_iterations=timed_iterations,
            seed=seed,
            print_report=False,
        )
        stages["8_benchmark_onnxruntime"] = {"ok": True}
        logger.info("Stage 8: ONNX Runtime benchmark done")
    except (OnnxBenchmarkError, FileNotFoundError) as exc:
        onnx_bench = {"backend": "onnxruntime", "error": str(exc)}
        stages["8_benchmark_onnxruntime"] = {"ok": False, "error": str(exc)}
        logger.error("Stage 8: benchmark failed: %s", exc)

    trt_bench: dict[str, Any] | None = None
    if "tensorrt" not in skipped and engine_path.is_file():
        try:
            trt_bench = benchmark_tensorrt_backend(
                engine_path,
                INPUT_SHAPE_STR,
                warmup_iterations=warmup_iterations,
                timed_iterations=timed_iterations,
                seed=seed,
                device=trt_device,
                print_report=False,
            )
            stages["9_benchmark_tensorrt"] = {"ok": True}
            logger.info("Stage 9: TensorRT benchmark done")
        except (TensorRTBenchmarkError, FileNotFoundError) as exc:
            trt_bench = {"backend": "tensorrt", "error": str(exc)}
            stages["9_benchmark_tensorrt"] = {"ok": False, "error": str(exc)}
            logger.error("Stage 9: TRT benchmark failed: %s", exc)
    else:
        stages["9_benchmark_tensorrt"] = {"ok": None, "skipped": skipped.get("tensorrt", "n/a")}

    validation_report: dict[str, Any] = {
        "event": "resnet18_validation",
        "model": "torchvision.models.resnet18",
        "pretrained": False,
        "input_shape": list(INPUT_SHAPE),
        "pytorch_vs_onnx": onnx_parity,
        "pytorch_vs_tensorrt": trt_parity,
        "skipped": skipped,
    }

    benchmark_report: dict[str, Any] = {
        "event": "resnet18_benchmark",
        "model": "torchvision.models.resnet18",
        "input_shape": list(INPUT_SHAPE),
        "onnxruntime": onnx_bench,
        "tensorrt": trt_bench,
        "comparison": _benchmark_comparison(onnx_bench, trt_bench),
        "skipped": skipped,
    }

    summary = _compute_summary(onnx_parity=onnx_parity, trt_parity=trt_parity, skipped=skipped)
    summary["validation_metrics"] = {
        "pytorch_vs_onnx": (
            {k: onnx_parity[k] for k in ("max_abs_error", "mean_abs_error", "cosine_similarity", "allclose", "status") if onnx_parity and k in onnx_parity}
        ),
        "pytorch_vs_tensorrt": (
            {
                k: trt_parity[k]
                for k in (
                    "max_abs_error",
                    "mean_abs_error",
                    "cosine_similarity",
                    "allclose",
                    "status",
                    "cosine_similarity_threshold",
                    "max_abs_error_threshold",
                )
                if trt_parity and k in trt_parity
            }
            if trt_parity
            else None
        ),
    }
    summary["benchmark_metrics"] = {
        "onnxruntime_latency_ms_mean": (onnx_bench or {}).get("latency_ms", {}).get("mean") if isinstance(onnx_bench, dict) else None,
        "onnxruntime_throughput_samples_per_sec": (onnx_bench or {}).get("throughput_samples_per_sec")
        if isinstance(onnx_bench, dict)
        else None,
        "tensorrt_latency_ms_mean": (trt_bench or {}).get("latency_ms", {}).get("mean") if isinstance(trt_bench, dict) else None,
        "tensorrt_throughput_samples_per_sec": (trt_bench or {}).get("throughput_samples_per_sec")
        if isinstance(trt_bench, dict)
        else None,
        "comparison": benchmark_report.get("comparison"),
    }

    meta["status"] = "completed"
    meta["stages"] = stages
    meta["artifacts"] = {
        "checkpoint": str(ckpt_path),
        "onnx": str(onnx_path),
        "engine": str(engine_path) if engine_path.is_file() else None,
        "validation_json": str(out / "validation.json"),
        "benchmark_json": str(out / "benchmark.json"),
        "summary_json": str(out / "summary.json"),
    }
    meta["validation"] = validation_report
    meta["benchmark"] = benchmark_report
    meta["summary"] = summary
    meta["skipped"] = skipped

    _finalize_writes(out, meta, validation_report, benchmark_report, summary)

    logger.info("integration_complete dir=%s summary=%s", out, json.dumps(summary, sort_keys=True))
    return meta


def _finalize_writes(
    out: Path,
    integration: dict[str, Any],
    validation: dict[str, Any] | None,
    benchmark: dict[str, Any] | None,
    summary: dict[str, Any] | None,
) -> None:
    (out / "integration_report.json").write_text(
        json.dumps(integration, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    if validation is not None:
        (out / "validation.json").write_text(json.dumps(validation, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    if benchmark is not None:
        (out / "benchmark.json").write_text(json.dumps(benchmark, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    if summary is not None:
        (out / "summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")


__all__ = ["INPUT_SHAPE", "INPUT_SHAPE_STR", "run_resnet18_integration_pipeline"]
