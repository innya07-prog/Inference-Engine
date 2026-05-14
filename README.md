# inference-engine

A modular **PyTorch → ONNX → TensorRT** inference and deployment toolkit. It packages export, ONNX Runtime execution, TensorRT 10 engine build and CUDA inference, numerical parity checks, benchmarking hooks, structured logging, and a command-line interface suitable for CI and local workflows.

---

## Architecture overview

```text
┌─────────────────┐     torch.export + dynamo ONNX      ┌──────────────┐
│ PyTorch (.pt)   │ ─────────────────────────────────────► │ ONNX (.onnx) │
└────────┬────────┘                                        └──────┬───────┘
         │                                                          │
         │ validate_onnx_parity                                     │ ONNX Runtime
         │ (Torch vs ORT)                                           │ (run / benchmark)
         │                                                          │
         │ validate_trt_parity                                      │ TensorRT 10
         └────────────────────────────────────────────────────────► │ build + infer
                                                                    └──────────────┘
```

| Layer | Package location | Responsibility |
|--------|------------------|----------------|
| **Export** | `inference_engine.exporters` | Load checkpoints, `torch.export`, serialize ONNX (`dynamo=True`) |
| **ONNX Runtime** | `inference_engine.runtimes.onnx_runner` | Single-session inference, CLI summaries |
| **TensorRT build** | `inference_engine.builders` | Parse ONNX, optimization profile, FP16 option, serialize `.engine` |
| **TensorRT runtime** | `inference_engine.runtimes.tensorrt_runtime` | Deserialize engine, GPU buffers, `execute_async_v3` |
| **Validation** | `inference_engine.validators` | PyTorch vs ONNX / vs TensorRT parity metrics |
| **Benchmarks** | `inference_engine.benchmarks` | ORT (CPU EP); TRT via same runtime (used by integration pipeline) |
| **Integration** | `inference_engine.integration` | ResNet18 end-to-end reports (`.pt`, `.onnx`, `.engine`, JSON) |
| **CLI** | `inference_engine.cli.main` | Subcommands for each stage |

---

## Project structure

```text
inference-engine/
├── pyproject.toml          # package metadata, deps, extras, console script
├── requirements.txt        # editable install hint (-e .)
├── docker/
│   └── Dockerfile          # NGC PyTorch base + pip install .
├── src/
│   └── inference_engine/
│       ├── __init__.py
│       ├── exporters/      # ONNX export
│       ├── runtimes/       # ONNX Runtime, TensorRT runtime
│       ├── builders/       # TensorRT engine from ONNX
│       ├── validators/     # Parity checks
│       ├── benchmarks/   # Latency / throughput helpers
│       ├── integration/    # ResNet18 pipeline + module entrypoint
│       ├── models/         # TestMLP and small reference nets
│       ├── tools/          # e.g. generate_test_model
│       ├── cli/            # argparse CLI
│       └── serving/        # placeholder for future HTTP/ASGI
└── tests/
    └── integration/        # ResNet18 pipeline tests (@pytest.mark.integration)
```

---

## Installation

**Requirements:** Python **3.10+**, a compatible **PyTorch** install (`torch>=2.2`). ONNX export uses the Dynamo exporter (`onnxscript`, `onnx-ir`, `onnx` are declared in `pyproject.toml`).

From the repository root:

```bash
python -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate
pip install -U pip
pip install -e .
```

**Optional extras**

| Extra | Command | Purpose |
|--------|---------|---------|
| Integration / ResNet18 tests | `pip install -e ".[integration]"` | `torchvision`, `pytest` |
| TensorRT (Python wheel matching your CUDA) | `pip install -e ".[trt]"` | `tensorrt>=10` |
| Future HTTP serving deps | `pip install -e ".[serve]"` | FastAPI stack (serving package is not fully wired yet) |

---

## CLI tooling

The console script is **`inference-engine`**. Equivalent module invocations:

```bash
python -m inference_engine.cli --help
python -m inference_engine.cli.main info
```

Global verbosity:

```bash
inference-engine -v info          # INFO logging
inference-engine -vv export ...  # DEBUG
```

### Commands at a glance

| Command | Role |
|---------|------|
| `info` | Version and runtime info |
| `export` | Checkpoint → ONNX |
| `run-onnx` | Single ONNX Runtime forward |
| `verify-onnx` | PyTorch vs ONNX parity |
| `benchmark-onnx` | ORT latency/throughput (**CPU** execution provider) |
| `build-trt` | ONNX → serialized TensorRT engine |
| `run-trt` | Single TensorRT inference (CUDA) |
| `verify-trt` | PyTorch vs TensorRT parity |
| `integration-test-resnet18` | Full ResNet18 pipeline + JSON artifacts |

---

## ONNX workflow

**1. Export** (example: packaged `TestMLP` checkpoint with `--test-mlp`):

```bash
inference-engine export \
  --checkpoint weights/test_model.pt \
  --output-onnx models/test_model.onnx \
  --example-shape 2,128 \
  --device cpu
```

**2. Run one inference** (random normal input):

```bash
inference-engine run-onnx --model models/test_model.onnx --random-shape 1,128
```

Or from a NumPy file aligned with the first ONNX input:

```bash
inference-engine run-onnx --model models/test_model.onnx --numpy input.npy --dtype float32
```

**3. Optional:** generate a small reference checkpoint:

```bash
python -m inference_engine.tools.generate_test_model -o weights/test_model.pt
```

Export uses **`torch.export`** and **`torch.onnx.export(..., dynamo=True)`**. Dynamic batch is enabled by default on the first input’s batch dimension (with logic to avoid batch=1 specialization).

---

## TensorRT workflow

> **TensorRT portability**  
> Serialized **`.engine` files are not portable** across GPUs, driver versions, and TensorRT build versions. **Build the engine on the same class of machine** (CUDA + TRT version) where you intend to run inference. Do not expect an engine built on one host to load on arbitrary other hosts.

**1. Install TensorRT** for your platform (see NVIDIA docs). With a matching pip wheel:

```bash
pip install -e ".[trt]"
```

**2. Build an engine from ONNX**

```bash
inference-engine build-trt \
  --model models/test_model.onnx \
  --engine models/test_model.engine \
  --fp16 auto
```

For dynamic shapes beyond the built-in defaults, supply **`--min-shape`**, **`--opt-shape`**, and **`--max-shape`** together (comma-separated, full rank).

**3. Run inference**

```bash
inference-engine run-trt --engine models/test_model.engine --shape 1,128 --device cuda:0
```

**4. Parity vs PyTorch** (requires CUDA and a compatible engine):

```bash
inference-engine verify-trt \
  --checkpoint weights/test_model.pt \
  --engine models/test_model.engine \
  --shape 1,128 \
  --test-mlp
```

---

## Validation examples

**PyTorch vs ONNX Runtime** (same random input, `numpy.allclose`-style tolerances):

```bash
inference-engine verify-onnx \
  --checkpoint weights/test_model.pt \
  --model models/test_model.onnx \
  --shape 1,128 \
  --device cpu \
  --test-mlp \
  --atol 1e-5 \
  --rtol 1e-4
```

**PyTorch vs TensorRT** (first TRT output compared to Torch reference; FP16-safe promotion on the Torch side):

```bash
inference-engine verify-trt \
  --checkpoint path/to/model.pt \
  --engine path/to/model.engine \
  --shape 1,3,224,224 \
  --device cuda:0
```

Structured logs include JSON-friendly lines (e.g. `parity_structured`, `tensorrt_build_structured`) for ingestion by log pipelines.

---

## Benchmark examples

### ONNX Runtime (CPU)

The benchmark subcommand uses **`CPUExecutionProvider` only** for stable, comparable numbers (not necessarily the same providers as `run-onnx`, which may prefer CUDA when available).

```bash
inference-engine benchmark-onnx \
  --model models/test_model.onnx \
  --shape 8,128 \
  --warmup 10 \
  --iterations 100 \
  --seed 0 \
  --output-json reports/onnx_bench.json
```

**Example printed report** (values are illustrative; your hardware will differ):

```text
========================================================================
ONNX Runtime benchmark (CPUExecutionProvider)
========================================================================
Model:              /path/to/models/test_model.onnx
Input shape:        (8, 128)  (batch_size=8)
Warmup iterations:  10
Timed iterations:   100
Timed wall time:    0.052341 s
------------------------------------------------------------------------
Latency mean:       0.4123 ms
Latency min:        0.3891 ms
Latency max:        0.5012 ms
Latency p50:        0.4088 ms
Latency p95:        0.4456 ms
------------------------------------------------------------------------
Throughput:         15284.32 samples/sec
========================================================================
```

JSON written to `--output-json` mirrors fields such as `latency_ms.mean`, `throughput_samples_per_sec`, and `execution_provider`.

### TensorRT

There is **no standalone `benchmark-trt` CLI** today. TensorRT benchmarking runs **inside the ResNet18 integration pipeline** (writes `benchmark.json` alongside ONNX metrics) or via Python:

```python
from inference_engine.benchmarks.tensorrt_benchmark import benchmark_tensorrt_backend

report = benchmark_tensorrt_backend(
    "models/model.engine",
    "1,128",
    warmup_iterations=10,
    timed_iterations=100,
    device="cuda:0",
    print_report=True,
)
```

---

## ResNet18 integration testing

End-to-end pipeline: torchvision **ResNet18** (`weights=None` / non-pretrained), fixed input **1×3×224×224**, checkpoint save, ONNX export, ORT run, optional TRT build/infer, parity, benchmarks, and JSON reports.

```bash
pip install -e ".[integration]"

inference-engine integration-test-resnet18 \
  --output-dir ./integration_reports/resnet18 \
  --warmup 10 \
  --iterations 50 \
  --export-device cpu \
  --onnx-parity-device cpu \
  --trt-device cuda:0
```

**Artifacts** (under `--output-dir`):

- `resnet18.pt`, `resnet18.onnx`, `resnet18.engine` (if TRT succeeds)
- `validation.json`, `benchmark.json`, `summary.json`, `integration_report.json`

**Module entrypoint:**

```bash
python -m inference_engine.integration --help
```

**Pytest** (requires `torchvision`, marked slow / optional GPU):

```bash
pytest tests/integration -m integration -v
```

---

## Docker usage

```bash
docker build -f docker/Dockerfile -t inference-engine:local .
docker run --rm inference-engine:local info
```

The sample [`docker/Dockerfile`](docker/Dockerfile) uses **`nvcr.io/nvidia/pytorch:24.01-py3`** and runs **`pip install .`** (core dependencies only). **TensorRT is not installed by that Dockerfile by default**; for TRT inside the image you must extend the image with TensorRT versions matched to the base CUDA stack, or use an NVIDIA TensorRT–oriented base image and align `pip install -e ".[trt]"` with available wheels.

GPU inference from the container typically requires NVIDIA Container Toolkit and runtime flags, for example:

```bash
docker run --rm --gpus all inference-engine:local inference-engine info
```

---

## GPU requirements

| Capability | Requirement |
|------------|-------------|
| **ONNX export / ORT / CPU parity** | CPU sufficient |
| **ONNX Runtime CUDA EP** | CUDA build of ONNX Runtime + GPU drivers (provider order is chosen automatically in `run-onnx`) |
| **TensorRT build & run** | NVIDIA GPU, CUDA, **TensorRT 10.x** Python package compatible with your driver/CUDA, **`torch.cuda.is_available()`** |

---

## Known limitations

- **TensorRT engine portability** — see warning above; treat engines as **build artifacts**, not shared binaries.
- **Single-output focus** — ONNX parity compares the **first** ONNX output to a **single** PyTorch tensor output; multi-output models need custom handling.
- **CLI `run-trt` / TRT benchmark helper** — oriented toward engines with **one dynamic input**; multi-input engines may need the Python API.
- **ONNX benchmark** — **CPU-only**; not directly comparable to `run-onnx` on a machine where ORT selects CUDA.
- **Checkpoints** — export loads full pickles with `weights_only=False` for flexibility; only use **trusted** checkpoint files.
- **`serving` package** — reserved for future HTTP/ASGI; not a complete service yet.

---

## Future roadmap

- Hardened **HTTP/ASGI** serving path under `[serve]` with documented API.
- **`benchmark-trt`** (or unified `benchmark`) CLI mirroring `benchmark-onnx`.
- **CI** workflows (CPU tests + optional GPU/TRT nightly).
- **Multi-input / multi-output** parity and TRT profile tooling.
- **Pinned lockfiles** or optional `requirements-lock.txt` for reproducible deploys.
- Richer **documentation** for CUDA/TRT version matrices and Docker variants (CPU vs GPU vs TRT).

---

## License

Add a `LICENSE` file at the repository root when you open-source the project; it is not bundled here by default.
