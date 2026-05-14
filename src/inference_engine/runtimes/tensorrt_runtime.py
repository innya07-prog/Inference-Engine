"""TensorRT 10.x runtime: load serialized engine, run inference with ``execute_async_v3``."""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

import numpy as np
import torch

logger = logging.getLogger(__name__)


class TensorRTRuntimeError(RuntimeError):
    """Raised when TensorRT engine load or inference fails."""


def _require_tensorrt() -> Any:
    try:
        import tensorrt as trt
    except ImportError as exc:
        raise TensorRTRuntimeError(
            "TensorRT Python package is not installed. Install TensorRT 10.x for your CUDA version "
            "(see NVIDIA documentation) or: pip install -e \".[trt]\""
        ) from exc
    return trt


def _trt_dtype_to_torch(trt: Any, trt_dtype: Any) -> torch.dtype:
    mapping = {
        trt.float32: torch.float32,
        trt.float16: torch.float16,
        trt.int32: torch.int32,
        trt.int64: torch.int64,
        trt.bool: torch.bool,
        trt.int8: torch.int8,
        trt.uint8: torch.uint8,
    }
    return mapping.get(trt_dtype, torch.float32)


def _cuda_stream_handle(stream: torch.cuda.Stream) -> int:
    if hasattr(stream, "cuda_stream"):
        return int(stream.cuda_stream)
    cur = torch.cuda.current_stream(device=stream.device)
    if hasattr(cur, "cuda_stream"):
        return int(cur.cuda_stream)
    raise TensorRTRuntimeError(
        "Could not resolve a CUDA stream handle from PyTorch. Use a CUDA-enabled PyTorch build with Stream.cuda_stream."
    )


def _execute_async_v3(ctx: Any, stream_handle: int) -> bool:
    """Call ``execute_async_v3`` with compatible calling conventions across TensorRT Python builds."""
    try:
        return bool(ctx.execute_async_v3(stream_handle))
    except TypeError:
        pass
    try:
        return bool(ctx.execute_async_v3(stream_handle=stream_handle))
    except TypeError:
        return bool(ctx.execute_async_v3(stream=stream_handle))


def _tensor_to_numpy(t: torch.Tensor) -> np.ndarray:
    x = t.detach()
    if x.dtype in (torch.float16, torch.bfloat16):
        x = x.float()
    return np.asarray(x.cpu().numpy(), dtype=np.float32)


class TensorRTRuntime:
    """
    Load a serialized TensorRT engine and run inference using the TensorRT 10 named-tensor API.

    Uses ``set_input_shape`` (dynamic), ``set_tensor_address``, ``execute_async_v3``, and CUDA stream
    synchronization after enqueue.
    """

    def __init__(self, engine_path: str | Path, *, device: torch.device | None = None) -> None:
        trt = _require_tensorrt()
        if not torch.cuda.is_available():
            raise TensorRTRuntimeError("CUDA is required for TensorRT inference (torch.cuda.is_available() is False).")

        self._trt = trt
        self._device = device or torch.device("cuda", torch.cuda.current_device())
        self._engine_path = Path(engine_path).expanduser().resolve()
        if not self._engine_path.is_file():
            raise FileNotFoundError(f"TensorRT engine not found: {self._engine_path}")

        try:
            blob = self._engine_path.read_bytes()
        except OSError as exc:
            raise TensorRTRuntimeError(f"Failed to read engine file: {exc}") from exc

        trt_logger = trt.Logger(trt.Logger.WARNING)
        runtime = trt.Runtime(trt_logger)
        try:
            engine = runtime.deserialize_cuda_engine(blob)
        except Exception as exc:
            raise TensorRTRuntimeError(f"deserialize_cuda_engine failed: {exc}") from exc
        if engine is None:
            raise TensorRTRuntimeError("deserialize_cuda_engine returned None (incompatible engine or TRT version).")

        try:
            context = engine.create_execution_context()
        except Exception as exc:
            raise TensorRTRuntimeError(f"create_execution_context failed: {exc}") from exc

        self._engine = engine
        self._context = context
        self._stream = torch.cuda.Stream(device=self._device)
        self._torch_device = self._device

        meta = {
            "event": "tensorrt_runtime_loaded",
            "engine_path": str(self._engine_path),
            "device": str(self._device),
            "num_io_tensors": int(engine.num_io_tensors),
        }
        logger.info("tensorrt_runtime_structured %s", json.dumps(meta, sort_keys=True))
        logger.info("Loaded TensorRT engine from %s (%s tensors)", self._engine_path, engine.num_io_tensors)

    @property
    def engine_path(self) -> Path:
        return self._engine_path

    @property
    def input_tensor_names(self) -> list[str]:
        trt = self._trt
        engine = self._engine
        return [
            engine.get_tensor_name(i)
            for i in range(engine.num_io_tensors)
            if engine.get_tensor_mode(engine.get_tensor_name(i)) == trt.TensorIOMode.INPUT
        ]

    def infer(self, feeds: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
        """
        Run one inference step.

        Parameters
        ----------
        feeds
            Mapping of input tensor name -> host ``numpy.ndarray``. All engine inputs must be present.

        Returns
        -------
        dict[str, numpy.ndarray]
            Output tensor name -> host ``float32`` numpy array (FP16 outputs promoted for downstream tools).
        """
        trt = self._trt
        ctx = self._context
        engine = self._engine

        input_names = [
            engine.get_tensor_name(i)
            for i in range(engine.num_io_tensors)
            if engine.get_tensor_mode(engine.get_tensor_name(i)) == trt.TensorIOMode.INPUT
        ]
        output_names = [
            engine.get_tensor_name(i)
            for i in range(engine.num_io_tensors)
            if engine.get_tensor_mode(engine.get_tensor_name(i)) == trt.TensorIOMode.OUTPUT
        ]

        missing = [n for n in input_names if n not in feeds]
        if missing:
            raise TensorRTRuntimeError(f"Missing inputs for inference: {missing} (required {input_names})")

        structured_in = {k: {"shape": list(np.asarray(v).shape), "dtype": str(np.asarray(v).dtype)} for k, v in feeds.items()}
        logger.debug("tensorrt_infer_inputs %s", json.dumps(structured_in, sort_keys=True))

        device_tensors: dict[str, torch.Tensor] = {}

        try:
            for name in input_names:
                arr = np.ascontiguousarray(np.asarray(feeds[name]))
                tt = _trt_dtype_to_torch(trt, engine.get_tensor_dtype(name))
                if arr.dtype != np.dtype(np.float32) and tt == torch.float32:
                    arr = arr.astype(np.float32, copy=False)
                t = torch.from_numpy(arr).to(device=self._torch_device, dtype=tt)
                if not t.is_contiguous():
                    t = t.contiguous()
                device_tensors[name] = t
                shape_tuple = tuple(int(x) for x in t.shape)
                if not ctx.set_input_shape(name, shape_tuple):
                    raise TensorRTRuntimeError(f"set_input_shape rejected shape {shape_tuple} for {name!r}")

            if hasattr(ctx, "update_device_memory_size_for_shapes"):
                try:
                    ctx.update_device_memory_size_for_shapes()
                except Exception as exc:
                    logger.debug("update_device_memory_size_for_shapes failed (optional): %s", exc)

            for name in output_names:
                trt_dtype = engine.get_tensor_dtype(name)
                tt = _trt_dtype_to_torch(trt, trt_dtype)
                shape = tuple(int(x) for x in ctx.get_tensor_shape(name))
                if any(x <= 0 for x in shape):
                    raise TensorRTRuntimeError(f"Invalid/unknown output shape for {name!r}: {shape}")
                vol = int(trt.volume(shape))
                if vol <= 0:
                    raise TensorRTRuntimeError(f"Non-positive volume for output {name!r}: shape={shape}")
                device_tensors[name] = torch.empty(shape, device=self._torch_device, dtype=tt)

            for name in input_names + output_names:
                t = device_tensors[name]
                ctx.set_tensor_address(name, t.data_ptr())

            stream_handle = _cuda_stream_handle(self._stream)
            with torch.cuda.stream(self._stream):
                ok = _execute_async_v3(ctx, stream_handle)
            if not ok:
                raise TensorRTRuntimeError("execute_async_v3 returned False.")

            self._stream.synchronize()
        except TensorRTRuntimeError:
            raise
        except Exception as exc:
            raise TensorRTRuntimeError(f"Inference failed: {exc}") from exc

        outputs: dict[str, np.ndarray] = {n: _tensor_to_numpy(device_tensors[n]) for n in output_names}

        out_meta = {k: {"shape": list(v.shape), "dtype": str(v.dtype)} for k, v in outputs.items()}
        logger.info("tensorrt_infer_outputs %s", json.dumps(out_meta, sort_keys=True))
        return outputs


def infer_trt_engine(
    engine_path: str | Path,
    feeds: dict[str, np.ndarray],
    *,
    device: torch.device | None = None,
) -> dict[str, np.ndarray]:
    """Load engine and run one inference (useful for scripts and future benchmarking)."""
    rt = TensorRTRuntime(engine_path, device=device)
    return rt.infer(feeds)


__all__ = ["TensorRTRuntime", "TensorRTRuntimeError", "infer_trt_engine"]
