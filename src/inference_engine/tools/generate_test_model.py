"""Generate ``weights/test_model.pt`` using the packaged :class:`~inference_engine.models.TestMLP`."""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

import torch

from inference_engine.models.test_mlp import TestMLP

logger = logging.getLogger("inference_engine.tools.generate_test_model")


def default_out_path() -> Path:
    return Path.cwd() / "weights" / "test_model.pt"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "-o",
        "--output",
        type=Path,
        default=default_out_path(),
        help="Output ``.pt`` path (default: ./weights/test_model.pt relative to current working directory).",
    )
    parser.add_argument("-v", "--verbose", action="count", default=0)
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(levelname)s %(name)s: %(message)s",
    )

    torch.manual_seed(0)
    model = TestMLP(in_features=128, hidden=64, out_features=10)
    model.eval()

    x = torch.randn(1, 128)
    with torch.no_grad():
        y = model(x)
    if y.shape != (1, 10):
        raise RuntimeError(f"unexpected output shape {tuple(y.shape)}")

    args.output.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "state_dict": model.state_dict(),
        "in_features": model.in_features,
        "hidden": model.hidden,
        "out_features": model.out_features,
        "arch": "TestMLP",
        "torch_version": torch.__version__,
    }
    torch.save(payload, args.output)
    logger.info("Saved %s (input %s, output %s)", args.output, tuple(x.shape), tuple(y.shape))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
