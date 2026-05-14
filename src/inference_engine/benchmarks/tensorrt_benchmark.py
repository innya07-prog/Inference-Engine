"""TensorRT CUDA inference benchmarking (reuses ``TensorRTRuntime``)."""

from __future__ import annotations

import json
import logging
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch

from inference_engine.runtimes.tensorrt_runtime import TensorRTRuntime, TensorRTRuntimeError

logger = logging.getLogger(__name__)


class TensorRTBenchmarkError(RuntimeError):
    """Raised when TensorRT benchmarking cannot be completed."""


def _parse_shape(shape: tuple[int, ...] | str) -> tuple[int, ...]:
    if isinstance(shape, str):
        parts = [p.strip() for p in shape.split(",") if p.strip()]
        if not parts:
            raise TensorRTBenchmarkError("input_shape must be a non-empty comma-separated list.")
        try:
            return tuple(int(p) for p in parts)
        except ValueError as exc:
            raise TensorRTBenchmarkError(f"Invalid input_shape string {shape!r}: {exc}") from exc
    if not shape:
        raise TensorRTBenchmarkError("input_shape must have at least one dimension.")
    return tuple(int(d) for d in shape)


def _percentiles_ms(latencies_ms: np.ndarray) -> tuple[float, float]:
    if latencies_ms.size == 0:
        return 0.0, 0.0
    p50 = float(np.percentile(latencies_ms, 50))
    p95 = float(np.percentile(latencies_ms, 95))
    return p50, p95


def benchmark_tensorrt_backend(
    engine_path: str | Path,
    input_shape: tuple[int, ...] | str,
    *,
    warmup_iterations: int = 10,
    timed_iterations: int = 100,
    seed: int = 0,
    device: str | torch.device | None = None,
    print_report: bool = False,
) -> dict[str, Any]:
    """
    Benchmark ``TensorRTRuntime.infer`` latency on CUDA (same input feed each iteration).

    Mirrors the structured fields produced by ``benchmark_backend`` for ONNX Runtime where practical.
    """
    if not torch.cuda.is_available():
        raise TensorRTBenchmarkError("CUDA is required for TensorRT benchmarking.")

    path = Path(engine_path).expanduser().resolve()
    if not path.is_file():
        raise FileNotFoundError(f"TensorRT engine not found: {path}")

    shape = _parse_shape(input_shape)
    batch_size = int(shape[0])
    dev = torch.device(device or "cuda:0")

    rng = np.random.default_rng(seed)
    x = np.ascontiguousarray(rng.standard_normal(shape, dtype=np.float32))

    try:
        rt = TensorRTRuntime(path, device=dev)
    except TensorRTRuntimeError as exc:
        raise TensorRTBenchmarkError(str(exc)) from exc

    in_names = rt.input_tensor_names
    if len(in_names) != 1:
        raise TensorRTBenchmarkError(
            f"Benchmark currently supports engines with exactly one input; found {len(in_names)}: {in_names}"
        )
    feed = {in_names[0]: x}

    logger.info(
        "tensorrt_benchmark_start engine=%s shape=%s warmup=%s timed=%s seed=%s device=%s",
        path,
        shape,
        warmup_iterations,
        timed_iterations,
        seed,
        dev,
    )

    try:
        for i in range(warmup_iterations):
            rt.infer(feed)
            if logger.isEnabledFor(logging.DEBUG):
                logger.debug("tensorrt warmup %s/%s", i + 1, warmup_iterations)
    except TensorRTRuntimeError as exc:
        raise TensorRTBenchmarkError(f"Warmup failed: {exc}") from exc

    latencies_ms: list[float] = []
    try:
        t_wall0 = time.perf_counter()
        for _ in range(timed_iterations):
            t0 = time.perf_counter()
            rt.infer(feed)
            latencies_ms.append((time.perf_counter() - t0) * 1000.0)
        wall_s = time.perf_counter() - t_wall0
    except TensorRTRuntimeError as exc:
        raise TensorRTBenchmarkError(f"Timed inference failed: {exc}") from exc

    arr = np.asarray(latencies_ms, dtype=np.float64)
    mean_ms = float(np.mean(arr))
    min_ms = float(np.min(arr))
    max_ms = float(np.max(arr))
    p50_ms, p95_ms = _percentiles_ms(arr)
    total_samples = batch_size * timed_iterations
    throughput = float(total_samples / wall_s) if wall_s > 0 else 0.0

    report: dict[str, Any] = {
        "event": "tensorrt_benchmark",
        "backend": "tensorrt",
        "execution_device": str(dev),
        "engine_path": str(path),
        "input_shape": list(shape),
        "batch_size": batch_size,
        "warmup_iterations": int(warmup_iterations),
        "timed_iterations": int(timed_iterations),
        "seed": int(seed),
        "wall_time_timed_phase_s": float(wall_s),
        "latency_ms": {
            "mean": mean_ms,
            "min": min_ms,
            "max": max_ms,
            "p50": p50_ms,
            "p95": p95_ms,
        },
        "throughput_samples_per_sec": throughput,
    }

    logger.info("tensorrt_benchmark_structured %s", json.dumps(report, sort_keys=True))

    if print_report:
        lat = report["latency_ms"]
        print(
            "\n".join(
                [
                    "",
                    "=" * 72,
                    "TensorRT benchmark (CUDA)",
                    "=" * 72,
                    f"Engine:            {path}",
                    f"Input shape:       {tuple(shape)}",
                    f"Timed wall:        {wall_s:.6f} s",
                    f"mean latency:      {lat['mean']:.4f} ms",
                    f"throughput:       {throughput:.2f} samples/sec",
                    "=" * 72,
                    "",
                ]
            )
        )

    return report


__all__ = ["TensorRTBenchmarkError", "benchmark_tensorrt_backend"]
