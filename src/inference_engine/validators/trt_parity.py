"""Compare PyTorch and TensorRT engine outputs (numerical parity, FP32 compare space)."""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

import numpy as np
import torch
import torch.nn as nn

from inference_engine.exporters.onnx_exporter import OnnxExportError, load_eager_from_checkpoint
from inference_engine.runtimes.tensorrt_runtime import TensorRTRuntime, TensorRTRuntimeError

logger = logging.getLogger(__name__)

Status = Literal["PASS", "FAIL"]
IntegerOutputPolicy = Literal["raise", "dequantize_reserved"]


class TensorRTParityValidationError(RuntimeError):
    """Raised when TensorRT parity validation cannot be completed."""


@dataclass(frozen=True)
class TrtParityResult:
    """Outcome and metrics from ``validate_trt_parity``."""

    status: Status
    max_abs_error: float
    mean_abs_error: float
    cosine_similarity: float
    allclose: bool
    atol: float
    rtol: float
    input_shape: tuple[int, ...]
    checkpoint_path: str
    engine_path: str
    torch_output_shape: tuple[int, ...]
    trt_output_shape: tuple[int, ...]
    seed: int
    torch_device: str
    trt_device: str
    compare_dtype: str
    trt_output_name: str
    integer_output_policy: str

    @property
    def passed(self) -> bool:
        return self.status == "PASS"


def _parse_shape(shape: tuple[int, ...] | str) -> tuple[int, ...]:
    if isinstance(shape, str):
        parts = [p.strip() for p in shape.split(",") if p.strip()]
        if not parts:
            raise TensorRTParityValidationError("input_shape must be a non-empty comma-separated string.")
        try:
            return tuple(int(p) for p in parts)
        except ValueError as exc:
            raise TensorRTParityValidationError(f"Invalid shape string {shape!r}: {exc}") from exc
    if not shape:
        raise TensorRTParityValidationError("input_shape must have at least one dimension.")
    return tuple(int(d) for d in shape)


def _torch_output_to_float32_numpy(t: torch.Tensor) -> np.ndarray:
    """FP16/bfloat16-safe promotion to float32 NumPy on CPU."""
    if not isinstance(t, torch.Tensor):
        raise TensorRTParityValidationError(f"Expected torch.Tensor, got {type(t)}.")
    with torch.no_grad():
        x = t.detach().cpu()
    if x.dtype in (torch.bfloat16, torch.float16):
        x = x.float()
    return np.asarray(x.numpy(), dtype=np.float32)


def _cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    a_f = np.ravel(a).astype(np.float64, copy=False)
    b_f = np.ravel(b).astype(np.float64, copy=False)
    denom = float(np.linalg.norm(a_f) * np.linalg.norm(b_f))
    if denom == 0.0:
        return 1.0 if np.array_equal(a_f, b_f) else 0.0
    return float(np.dot(a_f, b_f) / denom)


def _print_report(
    result: TrtParityResult,
    *,
    torch_summary: str,
    trt_summary: str,
) -> None:
    lines = [
        "",
        "=" * 72,
        "TensorRT parity validation report",
        "=" * 72,
        f"Checkpoint:        {result.checkpoint_path}",
        f"Engine:           {result.engine_path}",
        f"Input shape:      {tuple(result.input_shape)}",
        f"Random seed:      {result.seed}",
        f"Torch device:     {result.torch_device}",
        f"TRT device:       {result.trt_device}",
        f"Compare dtype:    {result.compare_dtype} (FP16-safe)",
        f"TRT output used:  {result.trt_output_name}",
        f"INT8 policy hook: {result.integer_output_policy}",
        f"atol / rtol:      {result.atol:g} / {result.rtol:g}",
        "-" * 72,
        f"Torch output:     shape={tuple(result.torch_output_shape)}  {torch_summary}",
        f"TRT output:       shape={tuple(result.trt_output_shape)}  {trt_summary}",
        "-" * 72,
        f"max |error|:      {result.max_abs_error:.6e}",
        f"mean |error|:     {result.mean_abs_error:.6e}",
        f"cosine similarity:{result.cosine_similarity:.8f}",
        f"np.allclose:      {result.allclose}",
        "-" * 72,
        f"RESULT:          {result.status}",
        "=" * 72,
        "",
    ]
    print("\n".join(lines))


def validate_trt_parity(
    checkpoint_path: str | Path,
    engine_path: str | Path,
    input_shape: tuple[int, ...] | str,
    *,
    module: nn.Module | None = None,
    torch_device: str | torch.device | None = None,
    trt_device: str | torch.device | None = None,
    seed: int = 0,
    atol: float = 1e-5,
    rtol: float = 1e-4,
    print_report: bool = True,
    output_tensor: str | None = None,
    integer_output_policy: IntegerOutputPolicy = "raise",
) -> TrtParityResult:
    """
    Run the same random float32 input through PyTorch (checkpoint) and a TensorRT engine, then compare in float32.

    ``TensorRTRuntime`` is used for TRT so dynamic-shape engines work via ``set_input_shape`` from the feed shape.
    Batch dimensions are part of ``input_shape`` (e.g. ``(8, 128)`` for batch 8).

    INT8 note
    ---------
    ``integer_output_policy`` is reserved for future INT8 / quantized output handling; today only floating
    comparisons are supported and non-float TRT outputs raise unless extended later.

    Parameters
    ----------
    checkpoint_path
        ``.pt`` / ``.pth`` with ``nn.Module`` or ``state_dict`` (pass ``module`` for the latter).
    engine_path
        Serialized TensorRT ``.engine`` file.
    input_shape
        Shape of the random normal input, e.g. ``(1, 128)`` or ``"1,128"``.
    module
        Eager module for ``state_dict`` checkpoints.
    torch_device, trt_device
        CUDA devices for reference Torch and TRT runtime. If ``torch_device`` is omitted, it defaults to ``trt_device``.
        If ``trt_device`` is omitted, it defaults to ``cuda:0`` when CUDA is available.
    seed, atol, rtol
        RNG seed and ``numpy.allclose`` tolerances (PASS iff ``allclose`` is true).
    print_report
        Print a human-readable report to stdout.
    output_tensor
        If the engine has multiple outputs, select which TRT output tensor name to compare (defaults to the first).
    integer_output_policy
        ``\"raise\"`` (default) or ``\"dequantize_reserved\"`` for future INT8 calibration flows (logged only today).

    Returns
    -------
    TrtParityResult

    Raises
    ------
    TensorRTParityValidationError
        On invalid configuration, dtype mismatches for validation, or runtime failures.
    FileNotFoundError
        If paths are missing.
    """
    if not torch.cuda.is_available():
        raise TensorRTParityValidationError("CUDA is required for TensorRT parity validation (install a CUDA PyTorch build).")

    trt_dev = torch.device(trt_device or "cuda:0")
    torch_dev = torch.device(torch_device) if torch_device is not None else trt_dev
    if trt_dev.type != "cuda" or torch_dev.type != "cuda":
        raise TensorRTParityValidationError("torch_device and trt_device must be CUDA devices for TensorRT parity.")

    ckpt = Path(checkpoint_path).expanduser().resolve()
    eng = Path(engine_path).expanduser().resolve()
    if not eng.is_file():
        raise FileNotFoundError(f"TensorRT engine not found: {eng}")

    shape = _parse_shape(input_shape)

    try:
        model = load_eager_from_checkpoint(ckpt, torch_dev, module)
    except OnnxExportError as exc:
        raise TensorRTParityValidationError(str(exc)) from exc

    rng = np.random.default_rng(seed)
    x_np = rng.standard_normal(shape, dtype=np.float32)
    x_np = np.ascontiguousarray(x_np)
    x_torch = torch.from_numpy(x_np.copy()).to(device=torch_dev, dtype=torch.float32)

    logger.info(
        "Starting TensorRT parity validation checkpoint=%s engine=%s shape=%s seed=%s torch=%s trt=%s atol=%s rtol=%s",
        ckpt,
        eng,
        shape,
        seed,
        torch_dev,
        trt_dev,
        atol,
        rtol,
    )

    with torch.no_grad():
        torch_out = model(x_torch)

    if isinstance(torch_out, (tuple, list)):
        if len(torch_out) != 1:
            raise TensorRTParityValidationError(
                "Parity check currently supports a single tensor output from PyTorch (tuple/list length must be 1)."
            )
        torch_out = torch_out[0]

    torch_np = _torch_output_to_float32_numpy(torch_out)

    try:
        rt = TensorRTRuntime(eng, device=trt_dev)
        in_names = rt.input_tensor_names
        if len(in_names) != 1:
            raise TensorRTParityValidationError(
                f"Parity currently supports engines with exactly one input tensor; found {len(in_names)}: {in_names}"
            )
        trt_outs = rt.infer({in_names[0]: x_np})
    except TensorRTRuntimeError as exc:
        raise TensorRTParityValidationError(str(exc)) from exc

    if not trt_outs:
        raise TensorRTParityValidationError("TensorRT runtime returned no outputs.")

    out_name: str
    if output_tensor is not None:
        if output_tensor not in trt_outs:
            raise TensorRTParityValidationError(f"output_tensor {output_tensor!r} not in TRT outputs {list(trt_outs)}.")
        out_name = output_tensor
    else:
        out_name = next(iter(trt_outs.keys()))
        if len(trt_outs) > 1:
            logger.info(
                "Engine has multiple outputs %s; comparing torch to TRT output %r (pass output_tensor= to override).",
                list(trt_outs.keys()),
                out_name,
            )

    raw = trt_outs[out_name]
    base = np.asarray(raw)
    if base.dtype.kind not in ("f", "c"):
        if integer_output_policy == "dequantize_reserved":
            raise TensorRTParityValidationError(
                "integer_output_policy='dequantize_reserved' is reserved for future INT8 dequantization paths."
            )
        raise TensorRTParityValidationError(
            f"TRT output {out_name!r} has dtype {base.dtype}; only floating outputs are supported for parity today."
        )

    trt_np = base.astype(np.float32, copy=False)

    if torch_np.shape != trt_np.shape:
        msg = f"Shape mismatch: torch {torch_np.shape} vs trt {trt_np.shape}"
        logger.error(msg)
        raise TensorRTParityValidationError(msg)

    diff = np.abs(torch_np - trt_np)
    max_abs = float(np.max(diff)) if diff.size else 0.0
    mean_abs = float(np.mean(diff)) if diff.size else 0.0
    cos_sim = _cosine_similarity(torch_np, trt_np)
    close = bool(np.allclose(torch_np, trt_np, atol=atol, rtol=rtol))
    status: Status = "PASS" if close else "FAIL"

    result = TrtParityResult(
        status=status,
        max_abs_error=max_abs,
        mean_abs_error=mean_abs,
        cosine_similarity=cos_sim,
        allclose=close,
        atol=float(atol),
        rtol=float(rtol),
        input_shape=shape,
        checkpoint_path=str(ckpt),
        engine_path=str(eng),
        torch_output_shape=tuple(int(x) for x in torch_np.shape),
        trt_output_shape=tuple(int(x) for x in trt_np.shape),
        seed=int(seed),
        torch_device=str(torch_dev),
        trt_device=str(trt_dev),
        compare_dtype="float32",
        trt_output_name=out_name,
        integer_output_policy=str(integer_output_policy),
    )

    structured: dict[str, Any] = {
        "event": "tensorrt_parity_validation",
        "status": result.status,
        "passed": result.passed,
        "max_abs_error": result.max_abs_error,
        "mean_abs_error": result.mean_abs_error,
        "cosine_similarity": result.cosine_similarity,
        "allclose": result.allclose,
        "atol": result.atol,
        "rtol": result.rtol,
        "input_shape": list(result.input_shape),
        "checkpoint_path": result.checkpoint_path,
        "engine_path": result.engine_path,
        "seed": result.seed,
        "torch_device": result.torch_device,
        "trt_device": result.trt_device,
        "compare_dtype": result.compare_dtype,
        "trt_output_name": result.trt_output_name,
        "integer_output_policy": result.integer_output_policy,
        "int8_validation": "reserved",
    }
    logger.info("parity_structured %s", json.dumps(structured, sort_keys=True))

    torch_summary = f"dtype=float32 min={float(torch_np.min()):.6g} max={float(torch_np.max()):.6g}"
    trt_summary = f"dtype=float32 min={float(trt_np.min()):.6g} max={float(trt_np.max()):.6g}"

    if print_report:
        _print_report(result, torch_summary=torch_summary, trt_summary=trt_summary)

    logger.info(
        "TensorRT parity finished: %s (max_abs=%.6e mean_abs=%.6e cos=%.6f allclose=%s)",
        result.status,
        result.max_abs_error,
        result.mean_abs_error,
        result.cosine_similarity,
        result.allclose,
    )

    return result


__all__ = [
    "IntegerOutputPolicy",
    "TensorRTParityValidationError",
    "TrtParityResult",
    "validate_trt_parity",
]
