"""ONNX Runtime inference benchmarking (CPU, latency and throughput)."""

from __future__ import annotations

import json
import logging
import time
from pathlib import Path
from typing import Any

import numpy as np

logger = logging.getLogger(__name__)

CPU_PROVIDER = "CPUExecutionProvider"


class OnnxBenchmarkError(RuntimeError):
    """Raised when ONNX Runtime benchmarking cannot be completed."""


def _parse_shape(shape: tuple[int, ...] | str) -> tuple[int, ...]:
    if isinstance(shape, str):
        parts = [p.strip() for p in shape.split(",") if p.strip()]
        if not parts:
            raise OnnxBenchmarkError("input_shape must be a non-empty comma-separated list.")
        try:
            return tuple(int(p) for p in parts)
        except ValueError as exc:
            raise OnnxBenchmarkError(f"Invalid input_shape string {shape!r}: {exc}") from exc
    if not shape:
        raise OnnxBenchmarkError("input_shape must have at least one dimension.")
    return tuple(int(d) for d in shape)


def _session_cpu(model_path: Path):
    try:
        import onnxruntime as ort
    except ImportError as exc:
        raise OnnxBenchmarkError(
            "onnxruntime is required for benchmarking. Install project dependencies (pyproject.toml)."
        ) from exc

    if CPU_PROVIDER not in ort.get_available_providers():
        raise OnnxBenchmarkError(
            f"{CPU_PROVIDER} is not available in this onnxruntime build. Available: {ort.get_available_providers()}"
        )

    logger.info("Creating InferenceSession with providers=[%s] for %s", CPU_PROVIDER, model_path)
    try:
        return ort.InferenceSession(
            model_path.as_posix(),
            providers=[CPU_PROVIDER],
        )
    except Exception as exc:
        raise OnnxBenchmarkError(f"Failed to load ONNX model: {exc}") from exc


def _first_input_feed(session: Any, shape: tuple[int, ...], rng: np.random.Generator) -> dict[str, np.ndarray]:
    inputs_meta = session.get_inputs()
    if not inputs_meta:
        raise OnnxBenchmarkError("ONNX model declares no inputs.")
    name = inputs_meta[0].name
    x = rng.standard_normal(shape, dtype=np.float32)
    x = np.ascontiguousarray(x)
    return {name: x}


def _percentiles_ms(latencies_ms: np.ndarray) -> tuple[float, float]:
    if latencies_ms.size == 0:
        return 0.0, 0.0
    p50 = float(np.percentile(latencies_ms, 50))
    p95 = float(np.percentile(latencies_ms, 95))
    return p50, p95


def benchmark_backend(
    model_path: str | Path,
    input_shape: tuple[int, ...] | str,
    *,
    warmup_iterations: int = 10,
    timed_iterations: int = 100,
    seed: int = 0,
    print_report: bool = True,
) -> dict[str, Any]:
    """
    Benchmark ONNX Runtime inference using ``CPUExecutionProvider`` only.

    Uses ``time.perf_counter()`` around each ``session.run`` during warmup and timed phases.
    The same randomly generated input feed is reused across all iterations (steady-state latency).

    Parameters
    ----------
    model_path
        Path to ``.onnx`` model.
    input_shape
        Full input tensor shape including batch (e.g. ``(8, 128)`` or ``"8,128"``). Batch size is ``input_shape[0]``.
    warmup_iterations
        Number of untimed warmup runs (default 10).
    timed_iterations
        Number of timed runs used for statistics (default 100).
    seed
        Seed for the random input generator.
    print_report
        If true, print a formatted summary to stdout.

    Returns
    -------
    dict
        JSON-serializable benchmark report (see ``print_benchmark_report`` for keys).

    Raises
    ------
    OnnxBenchmarkError
        On invalid arguments, missing deps, or runtime failures.
    FileNotFoundError
        If ``model_path`` does not exist.
    """
    path = Path(model_path).expanduser().resolve()
    if not path.is_file():
        raise FileNotFoundError(f"ONNX model not found: {path}")

    if warmup_iterations < 0:
        raise OnnxBenchmarkError("warmup_iterations must be >= 0.")
    if timed_iterations < 1:
        raise OnnxBenchmarkError("timed_iterations must be >= 1.")

    shape = _parse_shape(input_shape)
    batch_size = int(shape[0])

    session = _session_cpu(path)
    output_names = [o.name for o in session.get_outputs()]
    rng = np.random.default_rng(seed)
    feed = _first_input_feed(session, shape, rng)

    logger.info(
        "benchmark_backend start model=%s shape=%s warmup=%s timed=%s seed=%s outputs=%s",
        path,
        shape,
        warmup_iterations,
        timed_iterations,
        seed,
        output_names,
    )

    try:
        for i in range(warmup_iterations):
            session.run(None, feed)
            if logger.isEnabledFor(logging.DEBUG):
                logger.debug("warmup %s/%s", i + 1, warmup_iterations)
    except Exception as exc:
        raise OnnxBenchmarkError(f"Warmup inference failed: {exc}") from exc

    latencies_ms: list[float] = []
    try:
        t_wall0 = time.perf_counter()
        for _ in range(timed_iterations):
            t0 = time.perf_counter()
            session.run(None, feed)
            latencies_ms.append((time.perf_counter() - t0) * 1000.0)
        wall_s = time.perf_counter() - t_wall0
    except Exception as exc:
        raise OnnxBenchmarkError(f"Timed inference failed: {exc}") from exc

    arr = np.asarray(latencies_ms, dtype=np.float64)
    mean_ms = float(np.mean(arr))
    min_ms = float(np.min(arr))
    max_ms = float(np.max(arr))
    p50_ms, p95_ms = _percentiles_ms(arr)

    total_samples = batch_size * timed_iterations
    throughput = float(total_samples / wall_s) if wall_s > 0 else 0.0

    report: dict[str, Any] = {
        "event": "onnx_benchmark",
        "backend": "onnxruntime",
        "execution_provider": CPU_PROVIDER,
        "model_path": str(path),
        "input_shape": list(shape),
        "batch_size": batch_size,
        "warmup_iterations": int(warmup_iterations),
        "timed_iterations": int(timed_iterations),
        "seed": int(seed),
        "output_names": output_names,
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

    logger.info(
        "benchmark_backend complete mean_ms=%.4f p50_ms=%.4f p95_ms=%.4f throughput_samples_per_sec=%.2f",
        mean_ms,
        p50_ms,
        p95_ms,
        throughput,
    )
    logger.info("benchmark_structured %s", json.dumps(report, sort_keys=True))

    if print_report:
        print_benchmark_report(report)

    return report


def print_benchmark_report(report: dict[str, Any]) -> None:
    """Print a human-readable benchmark summary from a report dict."""
    lat = report.get("latency_ms", {})
    lines = [
        "",
        "=" * 72,
        "ONNX Runtime benchmark (CPUExecutionProvider)",
        "=" * 72,
        f"Model:              {report.get('model_path', '')}",
        f"Input shape:        {tuple(report.get('input_shape', ()))}  (batch_size={report.get('batch_size', '')})",
        f"Warmup iterations:  {report.get('warmup_iterations', '')}",
        f"Timed iterations:   {report.get('timed_iterations', '')}",
        f"Timed wall time:    {report.get('wall_time_timed_phase_s', 0.0):.6f} s",
        "-" * 72,
        f"Latency mean:       {lat.get('mean', 0.0):.4f} ms",
        f"Latency min:      {lat.get('min', 0.0):.4f} ms",
        f"Latency max:      {lat.get('max', 0.0):.4f} ms",
        f"Latency p50:      {lat.get('p50', 0.0):.4f} ms",
        f"Latency p95:      {lat.get('p95', 0.0):.4f} ms",
        "-" * 72,
        f"Throughput:       {report.get('throughput_samples_per_sec', 0.0):.2f} samples/sec",
        "=" * 72,
        "",
    ]
    print("\n".join(lines))


def write_benchmark_json(report: dict[str, Any], json_path: str | Path) -> Path:
    """Write the benchmark report dict to ``json_path`` (UTF-8)."""
    out = Path(json_path).expanduser().resolve()
    out.parent.mkdir(parents=True, exist_ok=True)
    payload = json.dumps(report, indent=2, sort_keys=True)
    out.write_text(payload + "\n", encoding="utf-8")
    logger.info("Wrote benchmark JSON to %s", out)
    return out


__all__ = [
    "CPU_PROVIDER",
    "OnnxBenchmarkError",
    "benchmark_backend",
    "print_benchmark_report",
    "write_benchmark_json",
]
