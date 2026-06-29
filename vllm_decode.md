# vLLM Repository Decoded

This document is a contributor-oriented map of this repository. It explains
what the major code files do, which files are most important, and how a request
flows through vLLM from API/CLI entry point to model execution and output.

The repository is large: this snapshot contains about 5,457 files, including
3,581 Python files, 237 Rust files, CUDA/C++ kernels, docs, examples, tests, and
CI/build assets. Instead of listing every generated fixture or test helper one by
one, this README groups the code by subsystem and calls out the files that are
important for understanding or changing behavior.

## What vLLM Is

vLLM is a high-throughput inference and serving engine for large language
models. It provides:

- A Python API for offline inference through `vllm.LLM`.
- A CLI command through the package script `vllm`.
- OpenAI-compatible HTTP serving for chat, completions, embeddings, responses,
  tokenization, transcription, and related APIs.
- A V1 execution engine with request scheduling, paged KV-cache management,
  multi-GPU/distributed execution, speculative decoding, LoRA, multimodal input,
  structured outputs, metrics, and custom kernels.

## Top-Level Repository Layout

| Path | Purpose |
| --- | --- |
| `vllm/` | Main Python package and runtime implementation. This is where most application logic lives. |
| `vllm/v1/` | Current V1 engine path: request processing, scheduler, engine core, executors, workers, sampling, KV cache, metrics, structured output. |
| `vllm/entrypoints/` | Public APIs and servers: Python `LLM`, CLI, OpenAI-compatible server, generate/pooling/speech APIs. |
| `vllm/model_executor/` | Model loading/execution code, model definitions, layers, attention backends, quantization, sampling, custom op wrappers. |
| `vllm/config/` | Strongly typed configuration objects used to build engine/model/runtime config. |
| `vllm/distributed/` | Distributed execution helpers, parallel state, KV transfer, collective communication. |
| `vllm/multimodal/` | Multimodal registry, parsing, preprocessing, and budgeting. |
| `vllm/lora/` | LoRA request/config/runtime support. |
| `vllm/compilation/` | Torch compile, CUDA graph, compiler cache, and graph-capture helpers. |
| `vllm/platforms/` | Hardware/platform abstraction for CPU, CUDA, ROCm, XPU, TPU, etc. |
| `vllm/tokenizers/` | Tokenizer abstractions and tokenizer-group management. |
| `vllm/tool_parsers/` | Tool/function-call parsing for chat APIs. |
| `vllm/reasoning/` | Reasoning-content parsers and protocol helpers. |
| `vllm/renderers/` | Converts user-level chat/completion inputs into engine-ready prompts and back. |
| `vllm/plugins/` | Plugin loading hooks and plugin interfaces. |
| `csrc/` | Native C++/CUDA/ROCm/CPU kernels and PyTorch extension bindings. |
| `rust/` | Rust crates used for parsing, tokenization, rendering, and related fast paths. |
| `benchmarks/` | Benchmark scripts for serving, throughput, latency, attention, kernels, prefix cache, etc. |
| `examples/` | Example applications, scripts, and chat templates. |
| `tests/` | Unit, integration, model, entrypoint, engine, scheduler, and platform tests. |
| `docs/` | Documentation source. |
| `requirements/` | Dependency sets for CUDA, CPU, ROCm, XPU, TPU, testing, linting, docs, and development. |
| `docker/` | Dockerfiles and container build definitions. |
| `cmake/`, `CMakeLists.txt`, `setup.py`, `pyproject.toml` | Build system for Python package and native extensions. |

## Most Important Files

### Packaging and Build

| File | What it does |
| --- | --- |
| `pyproject.toml` | Defines the Python project metadata, Python versions, build requirements, package discovery, Ruff/Mypy/Pytest config, and the console script `vllm = vllm.entrypoints.cli.main:main`. |
| `setup.py` | Main package build script. Coordinates extension builds and package metadata that cannot live only in `pyproject.toml`. |
| `CMakeLists.txt` | Top-level native extension build configuration. |
| `cmake/*.cmake` | Platform-specific and external-project build helpers. |
| `build_rust.sh` | Rust build helper for Rust-backed components. |
| `requirements/*.txt` | Runtime and development dependency groups by platform and purpose. |

### Public Python API

| File | What it does |
| --- | --- |
| `vllm/__init__.py` | Public package surface. Lazily exposes `LLM`, `LLMEngine`, `AsyncLLMEngine`, `EngineArgs`, `SamplingParams`, output types, and model registry symbols. |
| `vllm/entrypoints/llm.py` | Implements the main offline Python API class `LLM`. Users instantiate this for local batched generation. |
| `vllm/sampling_params.py` | Defines sampling controls such as temperature, top-p, top-k, max tokens, stop strings, logprobs, penalties, and structured-output related knobs. |
| `vllm/pooling_params.py` | Defines request parameters for pooling/embedding/classification-style tasks. |
| `vllm/outputs.py` | Public output data structures returned by generation, embeddings, pooling, classification, and scoring APIs. |

### CLI and HTTP Serving

| File | What it does |
| --- | --- |
| `vllm/entrypoints/cli/main.py` | Root `vllm` command. Registers subcommands such as `serve`, `bench`, `collect-env`, `run-batch`, and OpenAI serving commands. |
| `vllm/entrypoints/cli/serve.py` | CLI wrapper for server startup. |
| `vllm/entrypoints/openai/api_server.py` | Main OpenAI-compatible FastAPI server. Builds the app, registers routers, creates the async engine client, and starts serving. |
| `vllm/entrypoints/openai/cli_args.py` | Server argument definitions and validation. |
| `vllm/entrypoints/openai/*/api_router.py` | Route registration for chat, completion, responses, models, pooling, transcription, etc. |
| `vllm/entrypoints/openai/*/protocol.py` | Request/response schemas for each API family. |
| `vllm/entrypoints/openai/*/serving.py` | Converts HTTP requests into engine requests and streams or returns API responses. |
| `vllm/entrypoints/serve/utils/*.py` | Shared server utilities: logging, errors, SSL, health/lifespan, request logging, metrics, API helpers. |

### V1 Engine Core

| File | What it does |
| --- | --- |
| `vllm/v1/engine/llm_engine.py` | Synchronous engine wrapper. Converts user inputs to engine-core requests, sends them to `EngineCore`, then converts engine-core outputs back into public outputs. |
| `vllm/v1/engine/async_llm.py` | Async engine used by online serving. Handles asynchronous request submission, streaming, cancellation, and shutdown. |
| `vllm/v1/engine/core.py` | Inner execution loop. Owns executor, scheduler, KV-cache initialization, request stepping, and batch execution. |
| `vllm/v1/engine/core_client.py` | Client abstraction around `EngineCore`, supporting in-process and multiprocess engine-core modes. |
| `vllm/v1/engine/input_processor.py` | Converts raw prompts/chat/rendered inputs into `EngineCoreRequest` objects. |
| `vllm/v1/engine/output_processor.py` | Converts `EngineCoreOutputs` into user-facing `RequestOutput`/streaming outputs. |
| `vllm/v1/request.py` | Internal request state machine and request metadata. |
| `vllm/v1/outputs.py` | Internal model-runner and scheduler output structures. |

### Scheduler and KV Cache

| File | What it does |
| --- | --- |
| `vllm/v1/core/sched/scheduler.py` | Main scheduler. Chooses which requests/tokens run each step, enforces limits, handles waiting/running/finished queues, prefix cache, encoder cache, speculative decoding, and KV connectors. |
| `vllm/v1/core/sched/request_queue.py` | Request queue implementations and scheduling policy selection. |
| `vllm/v1/core/sched/output.py` | Data structures emitted by the scheduler for workers/model runners. |
| `vllm/v1/core/kv_cache_manager.py` | Allocates, frees, and tracks KV-cache blocks for requests. |
| `vllm/v1/core/block_pool.py` | Low-level pool of cache blocks. |
| `vllm/v1/core/kv_cache_utils.py` | KV-cache sizing, hashing, block-size resolution, and helper utilities. |
| `vllm/v1/core/encoder_cache_manager.py` | Cache manager for encoder/multimodal inputs. |
| `vllm/v1/kv_cache_interface.py` | Shared KV-cache configuration and interface types. |

### Executors and Workers

| File | What it does |
| --- | --- |
| `vllm/v1/executor/__init__.py` | Selects the correct executor class from the runtime config. |
| `vllm/v1/executor/abstract.py` | Executor interface used by the engine core. |
| `vllm/v1/executor/uniproc_executor.py` | Single-process executor path. |
| `vllm/v1/executor/multiproc_executor.py` | Multi-process local executor path. |
| `vllm/v1/executor/ray_executor.py`, `ray_executor_v2.py` | Ray-backed distributed executor paths. |
| `vllm/v1/worker/worker_base.py` | Worker interface/base class. |
| `vllm/v1/worker/gpu_worker.py` | GPU worker lifecycle and execution coordination. |
| `vllm/v1/worker/gpu_model_runner.py` | GPU model runner entry point for executing scheduled batches. |
| `vllm/v1/worker/gpu/model_runner.py` | Newer detailed GPU model-runner implementation and supporting logic. |
| `vllm/v1/worker/cpu_worker.py`, `cpu_model_runner.py` | CPU execution path. |
| `vllm/v1/worker/xpu_worker.py`, `xpu_model_runner.py` | Intel XPU execution path. |
| `vllm/v1/worker/gpu/sample/sampler.py` | GPU-side token sampling logic. |
| `vllm/v1/worker/gpu/spec_decode/*` | Speculative decoding implementations. |

### Model Execution and Kernels

| Path/File | What it does |
| --- | --- |
| `vllm/model_executor/models/` | Model architecture implementations and model registry entries. Look here to add or debug model support. |
| `vllm/model_executor/layers/` | Reusable neural-network layers: attention, normalization, quantized layers, MoE, rotary embeddings, logits processors, samplers. |
| `vllm/model_executor/layers/quantization/` | Quantization method implementations and dispatch. |
| `vllm/model_executor/input_processing.py` | Model-side input preparation. |
| `vllm/model_executor/model_loader/` | Loading Hugging Face/model weights into vLLM model classes. |
| `vllm/_custom_ops.py`, `_xpu_ops.py`, `_aiter_ops.py` | Python wrappers around native custom operations. |
| `csrc/` | Native kernels and PyTorch extension bindings for CUDA, ROCm, CPU, quantization, all-reduce, attention, MoE, and memory management. |
| `vllm/kernels/` | Python-side kernel integration and Triton/CUDA kernel helpers. |
| `vllm/vllm_flash_attn/` | vLLM flash-attention interface wrappers. |

### Config, Platform, and Utilities

| Path/File | What it does |
| --- | --- |
| `vllm/config/` | Config classes for model, cache, scheduler, parallelism, LoRA, compilation, quantization, observability, KV transfer, etc. |
| `vllm/engine/arg_utils.py` | Converts CLI/Python engine arguments into engine config objects. |
| `vllm/envs.py`, `vllm/env_override.py` | Environment-variable definitions and early overrides. |
| `vllm/platforms/` | Hardware detection and feature flags. |
| `vllm/logger.py`, `vllm/logging_utils/` | Logging setup and structured diagnostic helpers. |
| `vllm/utils/` | Shared utility functions. This is broad; check callers before changing behavior. |
| `vllm/usage/` | Usage telemetry/context helpers. |
| `vllm/tracing/` | OpenTelemetry/tracing integration. |

### Feature-Specific Areas

| Path | What it does |
| --- | --- |
| `vllm/lora/` | LoRA loading, request objects, resolver plugins, and runtime integration. |
| `vllm/multimodal/` | Multimodal input registration, mapping, preprocessing, encoder budgets, and caches. |
| `vllm/reasoning/` | Reasoning parser selection and parsing logic for models that emit reasoning traces. |
| `vllm/tool_parsers/` | Tool-call parsers for model-specific function calling formats. |
| `vllm/renderers/` | Online/offline input rendering and de-rendering. |
| `vllm/structured_output/`, `vllm/v1/structured_output/` | Structured output constraints and state management. |
| `vllm/distributed/kv_transfer/` | KV-cache transfer connectors for disaggregated/prefill-decode style deployments. |
| `vllm/distributed/ec_transfer/` | Expert-cache or elastic/expert transfer connector logic. |

## Core Request Flow

### Offline Python Flow: `LLM.generate(...)`

1. User imports `LLM` from `vllm`.
2. `vllm/__init__.py` lazily resolves `LLM` to `vllm/entrypoints/llm.py`.
3. `LLM.__init__` builds `EngineArgs`, creates a `VllmConfig`, and constructs `vllm.v1.engine.llm_engine.LLMEngine`.
4. `LLM.generate(...)`/offline mixins prepare prompts and `SamplingParams`.
5. `LLMEngine.add_request(...)` uses `InputProcessor` to convert prompts into `EngineCoreRequest` objects.
6. `EngineCoreClient` sends requests to `EngineCore`.
7. `EngineCore` asks `Scheduler` to choose which request tokens run next.
8. `Executor` dispatches scheduled work to one or more workers.
9. Worker/model runner executes the model, attention/KV-cache operations, logits processing, and sampling.
10. `EngineCore` receives `ModelRunnerOutput`, updates scheduler/request state, and returns `EngineCoreOutputs`.
11. `OutputProcessor` detokenizes/aggregates outputs into `RequestOutput`.
12. `LLM.generate(...)` returns completed outputs to the caller.

### Online Server Flow: OpenAI-Compatible API

1. User runs `vllm serve ...` or an OpenAI server command.
2. `pyproject.toml` maps the `vllm` executable to `vllm.entrypoints.cli.main:main`.
3. `vllm/entrypoints/cli/main.py` registers command modules and dispatches to the selected command.
4. Serving code enters `vllm/entrypoints/openai/api_server.py`.
5. `build_app(...)` creates a FastAPI app and registers routers for models, chat, completions, responses, generate, pooling, transcription, tokenization, and vLLM-specific endpoints.
6. `build_async_engine_client(...)` converts CLI args to `AsyncEngineArgs`, then creates `AsyncLLM`.
7. Request routers parse HTTP requests using their `protocol.py` schemas.
8. `serving.py` files convert API requests into vLLM engine requests.
9. `AsyncLLM` submits requests to the V1 engine and streams updates as tokens are produced.
10. API serving code formats engine outputs into OpenAI-compatible JSON or server-sent events.

### Engine Loop Flow

At a high level, each engine step does this:

```text
new/queued requests
    -> InputProcessor
    -> EngineCore request queue
    -> Scheduler selects tokens and allocates KV blocks
    -> Executor sends batch to worker(s)
    -> ModelRunner runs forward pass and sampling
    -> Scheduler updates request/KV state from outputs
    -> OutputProcessor builds user-facing outputs
```

The main files for this loop are:

- `vllm/v1/engine/llm_engine.py`
- `vllm/v1/engine/async_llm.py`
- `vllm/v1/engine/core.py`
- `vllm/v1/core/sched/scheduler.py`
- `vllm/v1/core/kv_cache_manager.py`
- `vllm/v1/executor/*.py`
- `vllm/v1/worker/*model_runner*.py`
- `vllm/v1/engine/output_processor.py`

## How To Navigate Common Tasks

### Add or Debug a Model

Start in:

- `vllm/model_executor/models/`
- `vllm/model_executor/models/registry.py` or registry-related files
- `vllm/model_executor/model_loader/`
- `tests/models/`

Most model-specific behavior belongs in `model_executor/models/`. Shared layers
belong in `model_executor/layers/`.

### Change Scheduling Behavior

Start in:

- `vllm/v1/core/sched/scheduler.py`
- `vllm/v1/core/sched/request_queue.py`
- `vllm/v1/core/sched/output.py`
- `vllm/v1/core/kv_cache_manager.py`
- `tests/v1/`

Scheduling changes usually affect correctness, throughput, streaming behavior,
prefix caching, and distributed behavior, so tests should cover request ordering,
preemption, aborts, cache allocation/freeing, and edge cases.

### Change CLI or Server API Behavior

Start in:

- `vllm/entrypoints/cli/main.py`
- `vllm/entrypoints/cli/*.py`
- `vllm/entrypoints/openai/api_server.py`
- `vllm/entrypoints/openai/*/protocol.py`
- `vllm/entrypoints/openai/*/serving.py`
- `tests/entrypoints/`

Protocol files define schemas. Serving files translate schemas to engine calls.
API router files connect routes to serving implementations.

### Change Sampling

Start in:

- `vllm/sampling_params.py`
- `vllm/v1/worker/gpu/sample/`
- `vllm/model_executor/layers/sampler.py` and related sampler/logits files
- `tests/sampling/` or relevant tests under `tests/`

Sampling changes often need deterministic tests and tests for streaming/logprobs.

### Change KV Cache or Prefix Caching

Start in:

- `vllm/v1/core/kv_cache_manager.py`
- `vllm/v1/core/block_pool.py`
- `vllm/v1/core/kv_cache_utils.py`
- `vllm/v1/core/sched/scheduler.py`
- `benchmarks/benchmark_prefix_caching.py`
- `tests/v1/`

This area is performance-critical and correctness-critical because it controls
which request owns which cache blocks.

### Add a Native Kernel or Custom Op

Start in:

- `csrc/`
- `vllm/_custom_ops.py`
- `vllm/model_executor/layers/`
- `CMakeLists.txt`
- `cmake/`
- `tests/kernels/` or relevant model/layer tests

Native changes may need CUDA/ROCm/CPU variants and build-system updates.

### Work on Distributed Execution

Start in:

- `vllm/distributed/`
- `vllm/v1/executor/`
- `vllm/v1/worker/`
- `vllm/v1/engine/core.py`
- `tests/distributed/`

Executor choice, worker lifecycle, process groups, tensor/pipeline/data
parallelism, and KV transfer all interact here.

## Important Data Structures

| Data structure | Location | Role |
| --- | --- | --- |
| `EngineArgs`, `AsyncEngineArgs` | `vllm/engine/arg_utils.py` | User/CLI configuration before final engine config creation. |
| `VllmConfig` | `vllm/config/` | Central runtime configuration object passed across engine, scheduler, executor, and workers. |
| `SamplingParams` | `vllm/sampling_params.py` | User-facing text generation controls. |
| `PoolingParams` | `vllm/pooling_params.py` | User-facing pooling/embedding controls. |
| `EngineCoreRequest` | `vllm/v1/engine/` | Internal request sent into the engine core. |
| `Request` | `vllm/v1/request.py` | Scheduler/runtime request state. |
| `SchedulerOutput` | `vllm/v1/core/sched/output.py` | Batch plan emitted by scheduler for model execution. |
| `ModelRunnerOutput` | `vllm/v1/outputs.py` | Output from worker/model runner back to engine core. |
| `RequestOutput` | `vllm/outputs.py` | Public generation output returned to users. |

## Testing and Validation Areas

| Path | What to look for |
| --- | --- |
| `tests/v1/` | V1 engine, scheduler, KV-cache, worker, and output behavior. |
| `tests/entrypoints/` | CLI/API server behavior and request/response validation. |
| `tests/models/` | Model compatibility and architecture-specific tests. |
| `tests/kernels/` | Kernel and low-level numerical tests. |
| `tests/distributed/` | Multi-GPU/distributed correctness. |
| `benchmarks/` | Performance checks for throughput, serving latency, prefix cache, attention, kernels. |

## Mental Model

Think of vLLM as five layers:

1. **Interface layer**: Python API, CLI, HTTP routes.
2. **Request translation layer**: renderers, input processors, protocol adapters.
3. **Engine layer**: request lifecycle, async/sync client, engine core.
4. **Scheduling/cache layer**: batching, token budgets, preemption, KV-cache blocks.
5. **Execution layer**: executor, workers, model runners, kernels, sampling.

When debugging, locate the layer where the symptom appears, then move one layer
inward:

- Bad HTTP schema or response shape: `entrypoints/openai`.
- Bad prompt conversion: `renderers`, `entrypoints/chat_utils.py`, `InputProcessor`.
- Request stuck or wrong batching: `v1/core/sched`.
- Wrong generated token/logprobs: sampler/model runner/model executor.
- Memory/cache issue: `v1/core/kv_cache*`, worker block tables, native kernels.
- Multi-GPU issue: `distributed`, `v1/executor`, `v1/worker`.

## Suggested First Reading Order

For a new contributor trying to understand this repository quickly:

1. `pyproject.toml`
2. `vllm/__init__.py`
3. `vllm/entrypoints/llm.py`
4. `vllm/entrypoints/cli/main.py`
5. `vllm/entrypoints/openai/api_server.py`
6. `vllm/v1/engine/llm_engine.py`
7. `vllm/v1/engine/async_llm.py`
8. `vllm/v1/engine/core.py`
9. `vllm/v1/core/sched/scheduler.py`
10. `vllm/v1/core/kv_cache_manager.py`
11. `vllm/v1/executor/abstract.py`
12. `vllm/v1/executor/uniproc_executor.py`
13. `vllm/v1/worker/gpu_model_runner.py`
14. `vllm/model_executor/models/`
15. `vllm/model_executor/layers/`

That sequence gives the clearest path from public usage to actual model
execution.

