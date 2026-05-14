"""Minimal ONNX Runtime execution for validation and smoke tests."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import numpy as np

logger = logging.getLogger(__name__)


class OnnxRunError(RuntimeError):
    """Raised when ONNX Runtime cannot load or execute the model."""


def _parse_shape(spec: str) -> tuple[int, ...]:
    parts = [p.strip() for p in spec.split(",") if p.strip()]
    if not parts:
        raise OnnxRunError("Shape must be a comma-separated list of integers, e.g. 1,128.")
    try:
        return tuple(int(p) for p in parts)
    except ValueError as exc:
        raise OnnxRunError(f"Invalid shape spec {spec!r}: {exc}") from exc


def run_onnx_once(
    model_path: str | Path,
    *,
    feeds: dict[str, np.ndarray] | None = None,
    random_shape: tuple[int, ...] | None = None,
    random_dtype: str = "float32",
) -> dict[str, np.ndarray]:
    """
    Run a single forward pass with ONNX Runtime.

    Either pass ``feeds`` (input name -> ndarray), or ``random_shape`` to synthesize one input
    using the graph's first input name and dtype.
    """
    try:
        import onnxruntime as ort
    except ImportError as exc:
        raise OnnxRunError(
            "onnxruntime is required for run-onnx. Install dependencies from pyproject.toml."
        ) from exc

    path = Path(model_path).expanduser().resolve()
    if not path.is_file():
        raise OnnxRunError(f"ONNX model not found: {path}")

    providers: list[str] = []
    try:
        providers = ort.get_available_providers()
    except Exception:
        providers = []

    preferred: list[str] = []
    for p in ("CUDAExecutionProvider", "TensorrtExecutionProvider", "CPUExecutionProvider"):
        if p in providers:
            preferred.append(p)
    if not preferred:
        preferred = ["CPUExecutionProvider"]

    logger.info("Loading ONNX model from %s with providers %s", path, preferred)
    try:
        session = ort.InferenceSession(path.as_posix(), providers=preferred)
    except Exception as exc:
        raise OnnxRunError(f"Failed to create InferenceSession: {exc}") from exc

    if feeds is not None:
        input_feed = {k: np.asarray(v) for k, v in feeds.items()}
    elif random_shape is not None:
        inputs_meta = session.get_inputs()
        if not inputs_meta:
            raise OnnxRunError("Model has no inputs declared.")
        name = inputs_meta[0].name
        onnx_type = inputs_meta[0].type
        dtype = np.dtype(random_dtype)
        if "float" not in str(dtype):
            raise OnnxRunError("random_dtype must be a floating numpy dtype for synthetic inputs.")
        if onnx_type not in ("tensor(float)", "tensor(float16)", "tensor(double)"):
            logger.warning("First ONNX input type is %s; synthetic data uses %s.", onnx_type, dtype)
        rng = np.random.default_rng(0)
        input_feed = {name: rng.standard_normal(random_shape, dtype=dtype)}
    else:
        raise OnnxRunError("Provide either feeds=... or random_shape=...")

    try:
        outputs = session.get_outputs()
        out_names = [o.name for o in outputs]
        raw = session.run(out_names, input_feed)
    except Exception as exc:
        raise OnnxRunError(f"Inference failed: {exc}") from exc

    return dict(zip(out_names, raw))


def run_onnx_cli(
    model_path: Path,
    *,
    numpy_path: Path | None,
    shape_spec: str | None,
    dtype: str,
) -> dict[str, Any]:
    """CLI-oriented wrapper returning serializable summary data."""
    feeds: dict[str, np.ndarray] | None = None
    random_shape: tuple[int, ...] | None = None

    if numpy_path is not None:
        arr = np.load(numpy_path, allow_pickle=False)
        if not isinstance(arr, np.ndarray):
            raise OnnxRunError("--numpy must point to a .npy file containing an ndarray.")
        try:
            import onnxruntime as ort
        except ImportError as exc:
            raise OnnxRunError("onnxruntime is required.") from exc
        path = Path(model_path).expanduser().resolve()
        session = ort.InferenceSession(path.as_posix(), providers=ort.get_available_providers())
        in0 = session.get_inputs()[0].name
        feeds = {in0: arr.astype(np.dtype(dtype), copy=False)}
    elif shape_spec is not None:
        random_shape = _parse_shape(shape_spec)
    else:
        raise OnnxRunError("Provide either --numpy PATH or --random-shape 1,128,...")

    outputs = run_onnx_once(model_path, feeds=feeds, random_shape=random_shape, random_dtype=dtype)
    summary: dict[str, Any] = {}
    for name, arr in outputs.items():
        a = np.asarray(arr)
        summary[name] = {
            "shape": list(a.shape),
            "dtype": str(a.dtype),
            "min": float(np.min(a)) if a.size else None,
            "max": float(np.max(a)) if a.size else None,
            "mean": float(np.mean(a)) if a.size else None,
        }
    return summary
