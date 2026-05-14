"""Compare PyTorch and ONNX Runtime outputs (numerical parity)."""

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

logger = logging.getLogger(__name__)

Status = Literal["PASS", "FAIL"]


class ParityValidationError(RuntimeError):
    """Raised when parity validation cannot be completed (load, shape, or runtime errors)."""


@dataclass(frozen=True)
class ParityResult:
    """Outcome and metrics from ``validate_onnx_parity``."""

    status: Status
    max_abs_error: float
    mean_abs_error: float
    cosine_similarity: float
    allclose: bool
    atol: float
    rtol: float
    input_shape: tuple[int, ...]
    checkpoint_path: str
    onnx_path: str
    torch_output_shape: tuple[int, ...]
    onnx_output_shape: tuple[int, ...]
    seed: int

    @property
    def passed(self) -> bool:
        return self.status == "PASS"


def _parse_shape(shape: tuple[int, ...] | str) -> tuple[int, ...]:
    if isinstance(shape, str):
        parts = [p.strip() for p in shape.split(",") if p.strip()]
        if not parts:
            raise ParityValidationError("input_shape must be a non-empty tuple or comma-separated string.")
        try:
            return tuple(int(p) for p in parts)
        except ValueError as exc:
            raise ParityValidationError(f"Invalid shape string {shape!r}: {exc}") from exc
    if not shape:
        raise ParityValidationError("input_shape must have at least one dimension.")
    return tuple(int(d) for d in shape)


def _torch_output_to_numpy(t: torch.Tensor) -> np.ndarray:
    """Detach, move to CPU, promote to float32 for stable numpy comparison."""
    if not isinstance(t, torch.Tensor):
        raise ParityValidationError(f"Expected torch.Tensor, got {type(t)}.")
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


def _print_report(result: ParityResult, *, torch_summary: str, onnx_summary: str) -> None:
    lines = [
        "",
        "=" * 72,
        "ONNX parity validation report",
        "=" * 72,
        f"Checkpoint:        {result.checkpoint_path}",
        f"ONNX model:        {result.onnx_path}",
        f"Input shape:       {tuple(result.input_shape)}",
        f"Random seed:       {result.seed}",
        f"atol / rtol:       {result.atol:g} / {result.rtol:g}",
        "-" * 72,
        f"Torch output:      shape={tuple(result.torch_output_shape)}  {torch_summary}",
        f"ONNX output:       shape={tuple(result.onnx_output_shape)}  {onnx_summary}",
        "-" * 72,
        f"max |error|:       {result.max_abs_error:.6e}",
        f"mean |error|:      {result.mean_abs_error:.6e}",
        f"cosine similarity: {result.cosine_similarity:.8f}",
        f"np.allclose:       {result.allclose}",
        "-" * 72,
        f"RESULT:           {result.status}",
        "=" * 72,
        "",
    ]
    print("\n".join(lines))


def validate_onnx_parity(
    checkpoint_path: str | Path,
    onnx_path: str | Path,
    input_shape: tuple[int, ...] | str,
    *,
    module: nn.Module | None = None,
    device: str | torch.device = "cpu",
    seed: int = 0,
    atol: float = 1e-5,
    rtol: float = 1e-4,
    print_report: bool = True,
) -> ParityResult:
    """
    Run the same random input through PyTorch (checkpoint) and ONNX Runtime and compare outputs.

    Parameters
    ----------
    checkpoint_path
        ``.pt`` / ``.pth`` with ``nn.Module`` or ``state_dict`` (requires ``module`` for the latter).
    onnx_path
        Exported ``.onnx`` model (first graph output is compared to the PyTorch output).
    input_shape
        Shape of the random input, e.g. ``(1, 128)`` or ``"1,128"``.
    module
        Eager module used when the checkpoint is a ``state_dict`` dict.
    device
        Device for PyTorch inference.
    seed
        Seed for ``numpy.random.Generator`` (same draw used for both backends).
    atol, rtol
        Tolerances for ``numpy.allclose`` (and reflected in ``passed`` / ``status``).
    print_report
        If true, print a human-readable report to stdout.

    Returns
    -------
    ParityResult
        Includes ``status`` of ``\"PASS\"`` or ``\"FAIL\"`` (PASS iff ``allclose`` is true).

    Raises
    ------
    ParityValidationError
        On invalid inputs, shape mismatch, or missing optional dependencies.
    FileNotFoundError
        If paths are missing.
    """
    ckpt = Path(checkpoint_path).expanduser().resolve()
    onnx = Path(onnx_path).expanduser().resolve()
    if not onnx.is_file():
        raise FileNotFoundError(f"ONNX model not found: {onnx}")

    shape = _parse_shape(input_shape)
    dev = torch.device(device)

    try:
        model = load_eager_from_checkpoint(ckpt, dev, module)
    except OnnxExportError as exc:
        raise ParityValidationError(str(exc)) from exc

    try:
        import onnxruntime as ort
    except ImportError as exc:
        raise ParityValidationError(
            "onnxruntime is required for parity validation. Install project dependencies."
        ) from exc

    rng = np.random.default_rng(seed)
    x_np = rng.standard_normal(shape, dtype=np.float32)
    x_np = np.ascontiguousarray(x_np)
    x_torch = torch.from_numpy(x_np.copy()).to(device=dev, dtype=torch.float32)

    logger.info(
        "Starting ONNX parity validation checkpoint=%s onnx=%s shape=%s seed=%s device=%s atol=%s rtol=%s",
        ckpt,
        onnx,
        shape,
        seed,
        dev,
        atol,
        rtol,
    )

    with torch.no_grad():
        torch_out = model(x_torch)

    if isinstance(torch_out, (tuple, list)):
        if len(torch_out) != 1:
            raise ParityValidationError("Parity check currently supports a single tensor output from PyTorch.")
        torch_out = torch_out[0]

    torch_np = _torch_output_to_numpy(torch_out)

    try:
        session = ort.InferenceSession(onnx.as_posix(), providers=ort.get_available_providers())
    except Exception as exc:
        raise ParityValidationError(f"Failed to load ONNX with ONNX Runtime: {exc}") from exc

    inputs_meta = session.get_inputs()
    if not inputs_meta:
        raise ParityValidationError("ONNX model has no inputs.")
    in_name = inputs_meta[0].name
    ort_type = inputs_meta[0].type
    if ort_type not in ("tensor(float)", "tensor(float16)", "tensor(double)"):
        logger.warning("ONNX first input type is %s; feeding float32 numpy array.", ort_type)

    try:
        onnx_outputs = session.run(None, {in_name: x_np})
    except Exception as exc:
        raise ParityValidationError(f"ONNX Runtime inference failed: {exc}") from exc

    if not onnx_outputs:
        raise ParityValidationError("ONNX Runtime returned no outputs.")
    onnx_np = np.asarray(onnx_outputs[0], dtype=np.float32)

    if torch_np.shape != onnx_np.shape:
        msg = f"Shape mismatch: torch {torch_np.shape} vs onnx {onnx_np.shape}"
        logger.error(msg)
        raise ParityValidationError(msg)

    diff = np.abs(torch_np - onnx_np)
    max_abs = float(np.max(diff)) if diff.size else 0.0
    mean_abs = float(np.mean(diff)) if diff.size else 0.0
    cos_sim = _cosine_similarity(torch_np, onnx_np)
    close = bool(np.allclose(torch_np, onnx_np, atol=atol, rtol=rtol))
    status: Status = "PASS" if close else "FAIL"

    result = ParityResult(
        status=status,
        max_abs_error=max_abs,
        mean_abs_error=mean_abs,
        cosine_similarity=cos_sim,
        allclose=close,
        atol=float(atol),
        rtol=float(rtol),
        input_shape=shape,
        checkpoint_path=str(ckpt),
        onnx_path=str(onnx),
        torch_output_shape=tuple(int(x) for x in torch_np.shape),
        onnx_output_shape=tuple(int(x) for x in onnx_np.shape),
        seed=int(seed),
    )

    structured: dict[str, Any] = {
        "event": "onnx_parity_validation",
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
        "onnx_path": result.onnx_path,
        "seed": result.seed,
    }
    logger.info("parity_structured %s", json.dumps(structured, sort_keys=True))

    torch_summary = f"dtype=float32 min={float(torch_np.min()):.6g} max={float(torch_np.max()):.6g}"
    onnx_summary = f"dtype=float32 min={float(onnx_np.min()):.6g} max={float(onnx_np.max()):.6g}"

    if print_report:
        _print_report(result, torch_summary=torch_summary, onnx_summary=onnx_summary)

    logger.info(
        "Parity validation finished: %s (max_abs=%.6e mean_abs=%.6e cos=%.6f allclose=%s)",
        result.status,
        result.max_abs_error,
        result.mean_abs_error,
        result.cosine_similarity,
        result.allclose,
    )

    return result


__all__ = [
    "ParityResult",
    "ParityValidationError",
    "validate_onnx_parity",
]
