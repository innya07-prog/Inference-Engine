"""Allow ``python -m inference_engine.cli`` (same entry as console script)."""

from inference_engine.cli.main import main

if __name__ == "__main__":
    raise SystemExit(main())
