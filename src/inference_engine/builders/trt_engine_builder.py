"""Build TensorRT 10.x serialized engines from ONNX (explicit batch, dynamic profiles, optional FP16)."""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)


class TensorRTBuildError(RuntimeError):
    """Raised when TensorRT engine build or ONNX parsing fails."""


def _require_tensorrt():
    try:
        import tensorrt as trt  # noqa: F401
    except ImportError as exc:
        raise TensorRTBuildError(
            "TensorRT Python package is not installed. Install TensorRT 10.x wheels for your "
            "CUDA version (see NVIDIA TensorRT install guide) and optional extra: pip install -e \".[trt]\""
        ) from exc


def _onnx_first_input(onnx_path: Path) -> tuple[str, list[int | str]]:
    try:
        import onnx
    except ImportError as exc:
        raise TensorRTBuildError("The 'onnx' package is required to inspect ONNX inputs.") from exc

    model = onnx.load(onnx_path.as_posix())
    if not model.graph.input:
        raise TensorRTBuildError("ONNX model has no inputs.")
    inp = model.graph.input[0]
    name = inp.name
    dims: list[int | str] = []
    for d in inp.type.tensor_type.shape.dim:
        which = d.WhichOneof("value")
        if which == "dim_param":
            dims.append(str(d.dim_param))
        elif which == "dim_value":
            dims.append(int(d.dim_value))
        else:
            dims.append(-1)
    return name, dims


def _resolve_profile_shapes(
    onnx_dims: list[int | str],
    min_shape: tuple[int, ...] | None,
    opt_shape: tuple[int, ...] | None,
    max_shape: tuple[int, ...] | None,
) -> tuple[tuple[int, ...], tuple[int, ...], tuple[int, ...]]:
    rank = len(onnx_dims)
    has_dynamic = any(
        (isinstance(d, str) and d not in ("",))
        or d in (-1, "?")
        or (isinstance(d, int) and d < 0)
        for d in onnx_dims
    )

    if min_shape and opt_shape and max_shape:
        for label, sh in (("min", min_shape), ("opt", opt_shape), ("max", max_shape)):
            if len(sh) != rank:
                raise TensorRTBuildError(f"{label}_shape rank {len(sh)} != ONNX input rank {rank}.")
        return min_shape, opt_shape, max_shape

    if min_shape or opt_shape or max_shape:
        raise TensorRTBuildError("Either provide all three profile shapes (min/opt/max) or omit all three.")

    if not has_dynamic:
        static = []
        for d in onnx_dims:
            if isinstance(d, int) and d > 0:
                static.append(int(d))
            else:
                raise TensorRTBuildError(f"Invalid ONNX static dimension in profile inference: {onnx_dims!r}")
        t = tuple(static)
        return t, t, t

    # Dynamic: default batch 1/4/32 on dim0; remaining dims must be positive constants in ONNX.
    rest: list[int] = []
    for d in onnx_dims[1:]:
        if isinstance(d, int) and d > 0:
            rest.append(int(d))
        else:
            raise TensorRTBuildError(
                "Dynamic non-batch dimensions are not auto-configured; pass explicit --min-shape/--opt-shape/--max-shape."
            )
    return (1, *rest), (4, *rest), (32, *rest)


def build_tensorrt_engine(
    onnx_path: str | Path,
    engine_path: str | Path,
    *,
    min_shape: tuple[int, ...] | None = None,
    opt_shape: tuple[int, ...] | None = None,
    max_shape: tuple[int, ...] | None = None,
    input_name: str | None = None,
    fp16: bool | None = None,
    workspace_mib: int = 1024,
    log_level: int | str = "WARNING",
) -> Path:
    """
    Parse ONNX, configure explicit-batch dynamic profile (when needed), optionally enable FP16, and serialize engine.

    TensorRT 10.x uses ``IBuilder.build_serialized_network`` and ``IBuilderConfig.set_memory_pool_limit`` for workspace.

    Parameters
    ----------
    onnx_path
        Path to ONNX model.
    engine_path
        Destination ``.engine`` path.
    min_shape, opt_shape, max_shape
        Optimization profile for the first ONNX input (required if ONNX has non-batch dynamic dims without defaults).
    input_name
        Override profile input name (defaults to first ONNX graph input).
    fp16
        If ``None``, enable FP16 when ``builder.platform_has_fast_fp16`` is true; if ``False``, never; if ``True``, require fast FP16.
    workspace_mib
        Workspace memory pool limit in MiB.
    log_level
        TensorRT logger level (``trt.Logger`` severity name or int).

    Returns
    -------
    pathlib.Path
        Resolved ``engine_path`` after write.

    Raises
    ------
    TensorRTBuildError
        On parser errors, unsupported ONNX, or build failures.
    FileNotFoundError
        If ``onnx_path`` is missing.
    """
    _require_tensorrt()
    import tensorrt as trt

    onnx_path = Path(onnx_path).expanduser().resolve()
    engine_path = Path(engine_path).expanduser().resolve()
    if not onnx_path.is_file():
        raise FileNotFoundError(f"ONNX model not found: {onnx_path}")

    if isinstance(log_level, str):
        sev = getattr(trt.Logger, log_level.upper(), trt.Logger.WARNING)
    else:
        sev = int(log_level)
    trt_logger = trt.Logger(sev)

    inferred_name, onnx_dims = _onnx_first_input(onnx_path)
    profile_input = input_name or inferred_name
    min_s, opt_s, max_s = _resolve_profile_shapes(onnx_dims, min_shape, opt_shape, max_shape)

    builder = trt.Builder(trt_logger)
    # TensorRT 10: implicit/explicit batch flags removed; ONNX is explicit batch.
    network = builder.create_network(0)
    parser = trt.OnnxParser(network, trt_logger)

    logger.info(
        "Parsing ONNX for TensorRT build onnx=%s engine_out=%s profile_input=%s min=%s opt=%s max=%s",
        onnx_path,
        engine_path,
        profile_input,
        min_s,
        opt_s,
        max_s,
    )

    try:
        data = onnx_path.read_bytes()
    except OSError as exc:
        raise TensorRTBuildError(f"Failed to read ONNX file: {exc}") from exc

    if not parser.parse(data):
        errors: list[str] = []
        try:
            n = int(parser.num_errors)
        except Exception:
            n = 0
        for i in range(n):
            try:
                errors.append(str(parser.get_error(i)))
            except Exception:
                errors.append(f"<error {i}>")
        if not errors:
            errors.append("Unknown ONNX parse failure (no parser errors exposed).")
        raise TensorRTBuildError("ONNX parse failed:\n" + "\n".join(errors))

    config = builder.create_builder_config()
    workspace_bytes = int(workspace_mib) * (1 << 20)
    try:
        config.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, workspace_bytes)
    except Exception as exc:
        raise TensorRTBuildError(f"Could not set WORKSPACE memory pool limit: {exc}") from exc

    profile = builder.create_optimization_profile()
    try:
        profile.set_shape(profile_input, min_s, opt_s, max_s)
    except Exception as exc:
        raise TensorRTBuildError(f"set_shape failed for {profile_input!r}: {exc}") from exc
    try:
        config.add_optimization_profile(profile)
    except Exception as exc:
        raise TensorRTBuildError(f"add_optimization_profile failed: {exc}") from exc

    use_fp16: bool
    if fp16 is None:
        use_fp16 = bool(builder.platform_has_fast_fp16)
    else:
        use_fp16 = bool(fp16)
        if use_fp16 and not builder.platform_has_fast_fp16:
            raise TensorRTBuildError("FP16 requested but this platform does not report fast FP16 support.")

    if use_fp16:
        config.set_flag(trt.BuilderFlag.FP16)
        logger.info("Enabling FP16 (platform_has_fast_fp16=%s)", builder.platform_has_fast_fp16)
    else:
        logger.info("FP16 disabled (fp16=%s, platform_has_fast_fp16=%s)", fp16, builder.platform_has_fast_fp16)

    structured: dict[str, Any] = {
        "event": "tensorrt_build_start",
        "onnx_path": str(onnx_path),
        "engine_path": str(engine_path),
        "profile_input": profile_input,
        "min_shape": list(min_s),
        "opt_shape": list(opt_s),
        "max_shape": list(max_s),
        "fp16": use_fp16,
        "workspace_mib": int(workspace_mib),
    }
    logger.info("tensorrt_build_structured %s", json.dumps(structured, sort_keys=True))

    try:
        serialized = builder.build_serialized_network(network, config)
    except Exception as exc:
        raise TensorRTBuildError(f"build_serialized_network raised: {exc}") from exc

    if serialized is None or len(serialized) == 0:
        raise TensorRTBuildError("build_serialized_network returned empty engine bytes.")

    engine_path.parent.mkdir(parents=True, exist_ok=True)
    try:
        engine_path.write_bytes(serialized)
    except OSError as exc:
        raise TensorRTBuildError(f"Failed to write engine file: {exc}") from exc

    logger.info("Wrote TensorRT engine (%s bytes) to %s", len(serialized), engine_path)
    logger.info(
        "tensorrt_build_done %s",
        json.dumps(
            {
                "event": "tensorrt_build_done",
                "engine_path": str(engine_path),
                "bytes": len(serialized),
                "fp16": use_fp16,
            },
            sort_keys=True,
        ),
    )
    return engine_path


__all__ = ["TensorRTBuildError", "build_tensorrt_engine"]
