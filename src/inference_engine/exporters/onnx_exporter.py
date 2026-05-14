"""ONNX export via ``torch.export`` and the dynamo ONNX backend."""

from __future__ import annotations

import contextlib
import logging
import sys
from pathlib import Path
from typing import Iterator, Sequence

import torch
import torch.nn as nn
from torch.export import Dim, export

logger = logging.getLogger(__name__)


@contextlib.contextmanager
def _stdio_utf8() -> Iterator[None]:
    """Avoid UnicodeEncodeError on Windows when torch.onnx prints UTF-8 symbols to the console."""
    out_enc = getattr(sys.stdout, "encoding", None)
    err_enc = getattr(sys.stderr, "encoding", None)
    reconfigured = False
    try:
        if hasattr(sys.stdout, "reconfigure"):
            sys.stdout.reconfigure(encoding="utf-8", errors="replace")
        if hasattr(sys.stderr, "reconfigure"):
            sys.stderr.reconfigure(encoding="utf-8", errors="replace")
        reconfigured = True
    except Exception:
        logger.debug("Could not reconfigure stdio to UTF-8; continuing.", exc_info=True)

    try:
        yield
    finally:
        if not reconfigured:
            return
        try:
            if hasattr(sys.stdout, "reconfigure") and out_enc:
                sys.stdout.reconfigure(encoding=out_enc, errors="replace")
            if hasattr(sys.stderr, "reconfigure") and err_enc:
                sys.stderr.reconfigure(encoding=err_enc, errors="replace")
        except Exception:
            logger.debug("Could not restore stdio encoding.", exc_info=True)


class OnnxExportError(RuntimeError):
    """Raised when ONNX export or validation fails after recoverable checks."""


def _normalize_example_args(example_input: torch.Tensor | tuple[torch.Tensor, ...]) -> tuple[torch.Tensor, ...]:
    if isinstance(example_input, torch.Tensor):
        return (example_input,)
    if not example_input:
        raise OnnxExportError("example_input must be a tensor or a non-empty tuple of tensors.")
    if not all(isinstance(t, torch.Tensor) for t in example_input):
        raise OnnxExportError("example_input tuple must contain only torch.Tensor instances.")
    return tuple(example_input)


def _ensure_non_singleton_batch_for_dynamic(
    example_args: tuple[torch.Tensor, ...],
    *,
    dynamic_batch: bool,
    batch_dim: int,
) -> tuple[torch.Tensor, ...]:
    """``torch.export`` may specialize batch==1; duplicate rows when marking batch dynamic."""
    if not dynamic_batch or not example_args:
        return example_args
    x0 = example_args[0]
    if x0.dim() == 0 or x0.shape[batch_dim] != 1:
        return example_args
    dup = torch.cat([x0, x0], dim=batch_dim)
    return (dup,) + tuple(example_args[1:])


def _load_torch_model(model_path: Path, device: torch.device, module: nn.Module | None) -> nn.Module:
    if not model_path.is_file():
        raise FileNotFoundError(f"Model path does not exist or is not a file: {model_path}")

    logger.info("Loading PyTorch artifact from %s", model_path)
    try:
        loaded = torch.load(model_path, map_location=device, weights_only=False)
    except Exception as exc:
        raise OnnxExportError(f"Failed to load torch artifact from {model_path}: {exc}") from exc

    if isinstance(loaded, nn.Module):
        model = loaded
    elif isinstance(loaded, dict) and "state_dict" in loaded:
        if module is None:
            raise OnnxExportError(
                "Checkpoint contains 'state_dict' but no eager nn.Module was provided. "
                "Pass module=<your nn.Module> so weights can be loaded before export."
            )
        try:
            module.load_state_dict(loaded["state_dict"], strict=True)
        except Exception as exc:
            raise OnnxExportError(f"state_dict could not be loaded into the provided module: {exc}") from exc
        model = module
    else:
        raise OnnxExportError(
            "Unsupported artifact type. Expected torch.nn.Module or a dict with key 'state_dict'."
        )

    if isinstance(model, torch.jit.ScriptModule):
        raise OnnxExportError(
            "TorchScript modules are not supported with torch.export-based ONNX export. "
            "Provide an eager torch.nn.Module or a state_dict checkpoint plus module=."
        )

    model = model.to(device)
    model.eval()
    return model


load_eager_from_checkpoint = _load_torch_model


def _validate_onnx_file(onnx_path: Path) -> None:
    try:
        import onnx
        from onnx import checker
    except ImportError as exc:
        raise OnnxExportError(
            "ONNX validation requires the 'onnx' package. Install project dependencies (see requirements.txt)."
        ) from exc

    logger.info("Validating ONNX model at %s", onnx_path)
    try:
        proto = onnx.load(onnx_path.as_posix(), load_external_data=True)
    except Exception as exc:
        raise OnnxExportError(f"Failed to read ONNX file for validation: {exc}") from exc

    try:
        checker.check_model(proto, full_check=True)
    except Exception as exc:
        raise OnnxExportError(f"Exported ONNX failed validation: {exc}") from exc


def export_to_onnx(
    model_path: str | Path,
    onnx_path: str | Path,
    *,
    example_input: torch.Tensor | tuple[torch.Tensor, ...],
    module: nn.Module | None = None,
    input_names: Sequence[str] | None = None,
    output_names: Sequence[str] | None = None,
    opset_version: int = 17,
    dynamic_batch: bool = True,
    batch_dim: int = 0,
    batch_dim_min: int | None = 1,
    batch_dim_max: int | None = 2048,
    device: str | torch.device = "cpu",
    validate: bool = True,
    external_data: bool = False,
) -> Path:
    """
    Export an eager ``torch.nn.Module`` (or state_dict checkpoint + ``module``) to ONNX using ``torch.export``.

    The graph is produced with ``torch.export.export`` and serialized with ``torch.onnx.export`` (``dynamo=True``).
    The requested ``opset_version`` is passed through to the exporter (default 17).

    Parameters
    ----------
    model_path
        ``.pt`` / ``.pth`` file containing either a pickled ``nn.Module`` or a dict with ``state_dict``.
    onnx_path
        Destination ``.onnx`` path (parent directories are created as needed).
    example_input
        Example forward arguments (single tensor or tuple). When ``dynamic_batch`` is true and the leading
        batch dimension is 1, it is internally duplicated along ``batch_dim`` so the batch axis is not specialized.
    module
        Required when ``model_path`` resolves to a state_dict checkpoint.
    input_names, output_names
        Optional ONNX tensor names. Defaults: ``input`` (or ``input0``, ``input1``, … for multi-input) and ``output``.
        If your exported program returns multiple tensors, supply ``output_names`` with the same arity.
    opset_version
        ONNX opset (default 17). Some PyTorch builds target a newer internal opset and convert down; you may see a
        warning even when the serialized graph reports opset 17.
    dynamic_batch
        If true, marks the batch dimension of the first input as dynamic using ``torch.export.Dim``.
    batch_dim
        Index of the batch dimension on the first input tensor (commonly 0).
    batch_dim_min, batch_dim_max
        Optional bounds for the dynamic batch ``Dim`` passed to ``torch.export``.
    device
        Map/load tensors on this device before tracing.
    validate
        If true, run ``onnx.checker.check_model`` on the written file.
    external_data
        Passed through to ``torch.onnx.export`` (default false for a single portable ``.onnx`` file).

    Returns
    -------
    pathlib.Path
        Resolved path to the written ONNX model.

    Raises
    ------
    OnnxExportError
        On unsupported checkpoints, export failures, or failed ONNX validation.
    FileNotFoundError
        If ``model_path`` is missing.
    """
    model_path = Path(model_path).expanduser().resolve()
    onnx_path = Path(onnx_path).expanduser().resolve()
    dev = torch.device(device)

    example_args = _normalize_example_args(example_input)
    example_args = tuple(t.detach().to(dev) for t in example_args)
    example_args = _ensure_non_singleton_batch_for_dynamic(
        example_args, dynamic_batch=dynamic_batch, batch_dim=batch_dim
    )

    model = _load_torch_model(model_path, dev, module)

    in_names: list[str]
    if input_names is None:
        if len(example_args) == 1:
            in_names = ["input"]
        else:
            in_names = [f"input{i}" for i in range(len(example_args))]
    else:
        in_names = list(input_names)
        if len(in_names) != len(example_args):
            raise OnnxExportError(
                f"input_names length ({len(in_names)}) must match number of example tensors ({len(example_args)})."
            )

    out_names: list[str]
    if output_names is None:
        out_names = ["output"]
    else:
        out_names = list(output_names)

    dynamic_shapes: tuple[dict[int, Dim], ...] | None
    if dynamic_batch:
        if not example_args[0].dim():
            raise OnnxExportError("dynamic_batch requires the first input to have at least one dimension.")
        if batch_dim < 0 or batch_dim >= example_args[0].dim():
            raise OnnxExportError(f"batch_dim={batch_dim} is out of range for first input rank {example_args[0].dim()}.")

        batch = Dim("batch", min=batch_dim_min, max=batch_dim_max)
        per_arg: list[dict[int, Dim]] = [{} for _ in example_args]
        per_arg[0][batch_dim] = batch
        dynamic_shapes = tuple(per_arg)
        logger.info(
            "Exporting with dynamic batch on first input dim %s (Dim 'batch', min=%s max=%s)",
            batch_dim,
            batch_dim_min,
            batch_dim_max,
        )
    else:
        dynamic_shapes = None
        logger.info("Exporting with static shapes (no dynamic batch).")

    try:
        exported = export(
            model,
            example_args,
            dynamic_shapes=dynamic_shapes,
        )
    except Exception as exc:
        raise OnnxExportError(f"torch.export.export failed: {exc}") from exc

    onnx_path.parent.mkdir(parents=True, exist_ok=True)

    try:
        with _stdio_utf8():
            torch.onnx.export(
                exported,
                example_args,
                onnx_path.as_posix(),
                opset_version=opset_version,
                input_names=in_names,
                output_names=out_names,
                dynamic_shapes=dynamic_shapes,
                external_data=external_data,
                dynamo=True,
            )
    except ImportError as exc:
        if getattr(exc, "name", None) in {"onnxscript", "onnx_ir"}:
            raise OnnxExportError(
                "The dynamo ONNX exporter requires optional dependencies (e.g. onnxscript). "
                "Install full requirements: pip install -r requirements.txt"
            ) from exc
        raise OnnxExportError(f"ONNX export failed due to a missing dependency: {exc}") from exc
    except Exception as exc:
        raise OnnxExportError(f"torch.onnx.export failed: {exc}") from exc

    logger.info("Wrote ONNX graph to %s (opset=%s)", onnx_path, opset_version)

    if validate:
        _validate_onnx_file(onnx_path)

    return onnx_path
