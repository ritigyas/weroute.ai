# vLLM Networking, Serving, Docker, CI, and Deployment Architecture

## Purpose

This document explains the operational side of the vLLM repository: network serving, API protocols, HTTP startup, Docker packaging, CI/CD structure, and deployment assets.

It is a companion to `MODEL_INPUT_PREPARATION_ARCHITECTURE.md`. That first document focuses on model-side tensor preparation. This document focuses on how requests reach the engine, how HTTP protocols are exposed, how runtime images are built, and how the repository supports deployment and validation.

## High-Level Summary

vLLM's online serving path is organized around a layered architecture:

1. CLI entrypoints parse `vllm serve` arguments.
2. The OpenAI-compatible API server creates an engine client.
3. A FastAPI app is built.
4. Protocol-specific routers are attached based on supported model tasks.
5. Middleware is installed for CORS, authentication, request IDs, scaling state, and optional custom behavior.
6. Uvicorn serves the FastAPI app over TCP or Unix domain sockets.
7. Route handlers validate protocol requests and call serving classes.
8. Serving classes render requests into engine inputs, call the async engine, and format full or streaming responses.
9. Dockerfiles package build and runtime variants.
10. Buildkite and GitHub Actions validate source, images, hardware targets, releases, and smoke tests.
11. Kubernetes, Helm, Ray, SageMaker, nginx, and integration examples provide deployment paths.

## Core Operational Files

### `vllm/entrypoints/cli/serve.py`

Defines the `vllm serve` CLI subcommand.

Responsibilities:

- register serve-specific CLI arguments
- validate parsed arguments
- set up logging and environment behavior
- call `run_server(args)` from the OpenAI API server

This is the public command-line entry into online serving.

### `vllm/entrypoints/openai/api_server.py`

Main online serving server.

Responsibilities:

- build an `AsyncLLM` engine client
- create and configure the FastAPI app
- attach route groups
- initialize app state
- bind TCP or Unix sockets
- start the Uvicorn server

Important functions:

- `build_async_engine_client`
- `build_async_engine_client_from_engine_args`
- `build_app`
- `init_app_state`
- `setup_server`
- `build_and_serve`
- `run_server`
- `run_server_worker`

### `vllm/entrypoints/launcher.py`

Generic Uvicorn launcher.

Responsibilities:

- log available routes
- configure h11 header limits
- create a Uvicorn server
- run the server with an optional pre-bound socket
- install shutdown handlers
- shut down the engine in drain or abort mode
- watch for engine failure and terminate the HTTP server when needed
- optionally refresh SSL certificates

Important functions:

- `serve_http`
- `watchdog_loop`
- `terminate_if_errored`

### `vllm/entrypoints/serve/__init__.py`

Registers shared vLLM service routers.

Production router groups:

- health and metrics instrumentation
- LoRA adapter management
- profiling
- tokenization and detokenization

Development-only router groups:

- cache
- RLHF
- RPC
- server info
- sleep

Development routes are gated by `VLLM_SERVER_DEV_MODE` and are explicitly warned as unsafe for production.

### `vllm/entrypoints/serve/utils/server_utils.py`

Shared API server utilities.

Important components:

- `AuthenticationMiddleware`
- `XRequestIdMiddleware`
- Uvicorn log config helpers
- SSE decoder helpers
- exception handlers
- FastAPI lifespan handling

Authentication is applied to guarded prefixes:

```python
GUARDED_PREFIX = ("/v1", "/v2", "/inference")
```

Health and metrics endpoints are intentionally left outside these guarded prefixes.

## Server Startup Flow

The usual online serving path is:

```text
vllm serve
  -> vllm.entrypoints.cli.serve.ServeSubcommand
  -> vllm.entrypoints.openai.api_server.run_server
  -> setup_server
  -> build_async_engine_client
  -> build_and_serve
  -> build_app
  -> init_app_state
  -> launcher.serve_http
  -> uvicorn.Server
```

### Socket Binding

`setup_server` binds the server socket before engine initialization.

Supported socket modes:

- TCP socket using host and port
- Unix domain socket using `--uds`

The socket is created before the engine is initialized to avoid race conditions in distributed or Ray-backed setups.

### App Construction

`build_app` creates a `FastAPI` instance and then conditionally registers routers based on supported tasks.

Task-dependent router registration:

- `generate`: completions, chat completions, responses, beam/generative scoring, disaggregated serving
- `render`: prompt rendering and derendering routes
- `transcription` or `realtime`: speech routes
- pooling tasks: embeddings, classify, score, rerank, pooling

Common routers are always registered first through `register_vllm_serve_api_routers`.

### App State Initialization

`init_app_state` stores shared runtime objects in `app.state`.

Important state entries:

- `engine_client`
- `vllm_config`
- `args`
- `openai_serving_models`
- `online_renderer`
- `online_derenderer`
- `serving_tokenization`
- `serving_render`
- task-specific serving handlers
- request logger
- server load tracking counters

This is how route handlers access the engine and shared serving objects.

### HTTP Runtime

`launcher.serve_http` wraps Uvicorn startup.

Important runtime behavior:

- logs all registered routes
- applies h11 header limits
- starts a watchdog task
- handles SIGINT and SIGTERM
- shuts down the engine before stopping the HTTP server
- supports SSL options and certificate refresh

## API Protocol Surface

vLLM exposes multiple protocol surfaces through FastAPI routers.

### OpenAI-Compatible Routes

The OpenAI-compatible server is the main production API surface.

Core route groups:

- models
- completions
- chat completions
- responses
- embeddings
- scoring and reranking
- transcription and translation
- realtime speech WebSocket

Representative files:

- `vllm/entrypoints/openai/models/api_router.py`
- `vllm/entrypoints/openai/completion/api_router.py`
- `vllm/entrypoints/openai/chat_completion/api_router.py`
- `vllm/entrypoints/openai/responses/api_router.py`
- `vllm/entrypoints/pooling/embed/api_router.py`
- `vllm/entrypoints/pooling/scoring/api_router.py`
- `vllm/entrypoints/speech_to_text/transcription/api_router.py`
- `vllm/entrypoints/speech_to_text/translation/api_router.py`
- `vllm/entrypoints/speech_to_text/realtime/api_router.py`

Examples of OpenAI-compatible paths:

```text
GET  /v1/models
POST /v1/completions
POST /v1/chat/completions
POST /v1/chat/completions/batch
POST /v1/responses
GET  /v1/responses/{response_id}
POST /v1/responses/{response_id}/cancel
POST /v1/embeddings
POST /v1/audio/transcriptions
POST /v1/audio/translations
WS   /v1/realtime
```

Request and response schemas are defined in protocol files beside each router:

```text
protocol.py
serving.py
api_router.py
```

This pattern appears throughout the entrypoints tree.

### Anthropic-Compatible Routes

Anthropic-compatible support lives under:

```text
vllm/entrypoints/anthropic/
```

Representative routes:

```text
POST /v1/messages
POST /v1/messages/count_tokens
```

The Anthropic serving implementation converts Anthropic message requests into OpenAI-style chat completion requests internally, then re-formats the result into Anthropic response shapes.

### Pooling Routes

Pooling endpoints support embedding, classification, scoring, reranking, and pooling-style tasks.

Representative files:

- `vllm/entrypoints/pooling/embed/api_router.py`
- `vllm/entrypoints/pooling/classify/api_router.py`
- `vllm/entrypoints/pooling/scoring/api_router.py`
- `vllm/entrypoints/pooling/pooling/api_router.py`

Representative paths:

```text
POST /v1/embeddings
POST /embed
POST /classify
POST /score
POST /v1/score
POST /rerank
POST /v1/rerank
```

Pooling routes are attached only when the loaded model advertises pooling tasks.

### Speech Routes

Speech-to-text and realtime routes live under:

```text
vllm/entrypoints/speech_to_text/
```

Representative paths:

```text
POST /v1/audio/transcriptions
POST /v1/audio/translations
WS   /v1/realtime
```

Realtime speech uses WebSocket routing and has dedicated metrics middleware.

### Tokenization and Rendering Routes

Shared service routes include:

```text
POST /tokenize
POST /detokenize
GET  /tokenizer_info
POST /v1/completions/render
POST /v1/chat/completions/render
POST /v1/completions/derender
POST /v1/chat/completions/derender
```

These routes are useful when clients need to inspect prompt formatting, tokenization, chat template application, or reverse-rendering behavior without running full generation.

### Health and Metrics Routes

Instrumentation routes live under:

```text
vllm/entrypoints/serve/instrumentator/
```

Core paths:

```text
GET /health
GET /metrics
```

`/health` is used by Docker, Kubernetes, Helm chart probes, smoke tests, and serving readiness checks.

`/metrics` mounts a Prometheus ASGI app.

## Request Handling Pattern

Most API route groups follow the same structure.

```text
api_router.py
  -> FastAPI route definitions
  -> pulls handler from app.state
  -> validates request content type
  -> calls serving class
  -> returns JSONResponse or StreamingResponse

protocol.py
  -> Pydantic request/response models
  -> OpenAI, Anthropic, pooling, or speech schema definitions

serving.py
  -> converts protocol objects into internal request parameters
  -> renders prompts
  -> calls engine_client
  -> formats full or streaming responses
```

This separation keeps protocol schema, route binding, and engine interaction distinct.

## Streaming Protocol

Streaming HTTP responses use Server-Sent Events.

Common response type:

```python
StreamingResponse
```

For chat and completion streams, serving classes emit chunks compatible with the relevant protocol:

- OpenAI chat completion chunks
- OpenAI completion chunks
- OpenAI responses stream events
- Anthropic message stream events

`SSEDecoder` in `server_utils.py` provides robust parsing support for streamed chunks.

## Middleware and Security

### CORS

`build_app` installs `CORSMiddleware` using CLI arguments:

- `allowed_origins`
- `allow_credentials`
- `allowed_methods`
- `allowed_headers`

### API Key Authentication

If `--api-key` or `VLLM_API_KEY` is set, `AuthenticationMiddleware` is installed.

Behavior:

- checks `Authorization: Bearer <token>`
- stores SHA-256 token hashes
- compares using constant-time comparison
- protects paths beginning with `/v1`, `/v2`, or `/inference`
- skips authentication for `OPTIONS`
- does not guard `/health` or `/metrics`

### Request IDs

If enabled, `XRequestIdMiddleware` appends `X-Request-Id` to responses. It preserves an incoming request ID if supplied, otherwise generates a UUID.

### Custom Middleware

`--middleware` allows importing custom middleware by module path. The imported object can be either:

- middleware class
- async HTTP middleware function

### SSL and UDS

The server supports:

- `ssl_keyfile`
- `ssl_certfile`
- `ssl_ca_certs`
- `ssl_cert_reqs`
- `ssl_ciphers`
- SSL refresh
- Unix domain sockets

## Docker Architecture

Docker support is split across multiple Dockerfiles and build targets.

### CUDA Dockerfile

Main file:

```text
docker/Dockerfile
```

Important targets:

- `base`
- `rust-build`
- `csrc-build`
- `extensions-build`
- `build`
- `dev`
- `vllm-base`
- `test`
- `vllm-openai-base`
- `vllm-sagemaker`
- `vllm-openai`
- `vllm-openai-nonroot`

The runtime OpenAI target ends with:

```dockerfile
ENTRYPOINT ["vllm", "serve"]
```

The non-root image uses:

```dockerfile
ENTRYPOINT ["/usr/local/bin/vllm-nonroot-entrypoint.sh"]
```

### CPU Dockerfile

Main file:

```text
docker/Dockerfile.cpu
```

Important targets:

- CPU build image
- CPU test image
- CPU OpenAI runtime image
- CPU Zen runtime variant

Runtime target also uses:

```dockerfile
ENTRYPOINT ["vllm", "serve"]
```

### ROCm Dockerfile

Main files:

```text
docker/Dockerfile.rocm
docker/Dockerfile.rocm_base
docker/docker-bake-rocm.hcl
docker/ci-rocm.hcl
```

The ROCm build is more specialized and includes:

- ROCm base setup
- csrc wheel building
- Rust component building
- release wheel export
- CI base image
- test targets
- optional network/backend components such as UCX, RIXL, rocSHMEM, DeepEP, and NIC-specific setup

### XPU Dockerfile

Main file:

```text
docker/Dockerfile.xpu
```

Important characteristics:

- Intel oneAPI and XPU dependencies
- Rust build stage
- vLLM OpenAI runtime target
- `VLLM_TARGET_DEVICE=xpu`
- `VLLM_WORKER_MULTIPROC_METHOD=spawn`

### TPU Dockerfile

Main file:

```text
docker/Dockerfile.tpu
```

Important characteristics:

- starts from a TPU/XLA base image
- installs TPU-specific dependencies
- sets `VLLM_TARGET_DEVICE=tpu`
- uses editable install for vLLM

### Docker Bake

Bake files define grouped builds for CI and release workflows:

```text
docker/docker-bake.hcl
docker/docker-bake-rocm.hcl
docker/ci-rocm.hcl
```

These files let CI build selected targets consistently, pass metadata, and export wheels or images.

### Non-Root Runtime Support

The CUDA Dockerfile creates a `vllm` user with UID `2000` and group `0`.

The non-root entrypoint:

```text
docker/entrypoints/vllm-nonroot-entrypoint.sh
```

sets safe defaults for environment variables such as `HOME`, `USER`, and `LOGNAME` when running under non-root or arbitrary UID environments.

## CI/CD Architecture

The repository uses both GitHub Actions and Buildkite.

### GitHub Actions

Workflow directory:

```text
.github/workflows/
```

Key workflows:

- `pre-commit.yml`
- `macos-smoke-test.yml`
- `new_pr_bot.yml`
- `issue_autolabel.yml`
- `add_label_automerge.yml`
- `stale.yml`

GitHub Actions primarily handles:

- pre-commit checks
- PR validation gates
- PR automation
- issue labeling
- stale issue management
- macOS Apple Silicon smoke testing

The macOS smoke test builds vLLM on macOS, starts `vllm serve`, waits for `/health`, and calls `/v1/completions`.

### Buildkite

Buildkite is the heavy CI system.

Main config:

```text
.buildkite/ci_config.yaml
```

Important directories:

```text
.buildkite/image_build/
.buildkite/test_areas/
.buildkite/hardware_tests/
.buildkite/performance-benchmarks/
.buildkite/lm-eval-harness/
.buildkite/scripts/
```

`ci_config.yaml` defines:

- job directories
- files that trigger full CI
- exclude patterns
- container registries
- premerge and postmerge repositories

The deprecated `.buildkite/test-pipeline.yaml` states that CI has moved into:

- `.buildkite/test_areas`
- `.buildkite/image_build`
- `.buildkite/hardware_tests`
- `.buildkite/ci_config.yaml`

### Buildkite Job Areas

The `test_areas` directory groups tests by functional ownership:

- attention
- basic correctness
- benchmarks
- compile
- CUDA
- disaggregated serving
- distributed
- Docker
- engine
- entrypoints
- expert parallelism
- kernels
- LoRA
- model executor
- model runner v2
- quantization
- Ray compatibility
- Rust frontend
- samplers
- speculative decoding
- weight loading

### Hardware CI

Hardware-specific pipelines live under:

```text
.buildkite/hardware_tests/
.buildkite/scripts/hardware_ci/
```

Supported hardware areas include:

- AMD/ROCm
- CPU
- GH200
- Intel
- Intel XPU
- Ascend NPU
- TPU
- other platform-specific runners

### Image Build CI

Image build scripts live under:

```text
.buildkite/image_build/
```

Representative scripts:

- `image_build.sh`
- `image_build_cpu.sh`
- `image_build_arm64.sh`
- `image_build_xpu.sh`
- `image_build_torch_nightly.sh`
- `image_build_hpu.sh`

These scripts build and publish CI/runtime images for different targets.

### Release and Nightly Jobs

Release and nightly scripts live under:

```text
.buildkite/release-pipeline.yaml
.buildkite/scripts/
```

Representative scripts:

- `upload-release-wheels-pypi.sh`
- `upload-nightly-wheels.sh`
- `push-nightly-builds.sh`
- `publish-release-images.sh`
- `generate-and-upload-nightly-index.sh`
- `cleanup-nightly-builds.sh`

## Deployment Architecture

Deployment docs and examples live mainly under:

```text
docs/deployment/
examples/deployment/
examples/ray_serving/
examples/disaggregated/
```

### Docker Deployment

Primary doc:

```text
docs/deployment/docker.md
```

Important runtime image:

```text
vllm/vllm-openai
```

The image runs `vllm serve` by default. Users pass the model name as the container command arguments.

Non-root deployment is supported by either:

- running the default image with `--user 2000:0`
- building the `vllm-openai-nonroot` target

### Kubernetes Deployment

Primary doc:

```text
docs/deployment/k8s.md
```

The Kubernetes examples use:

- Deployment
- Service
- PVC for model/cache storage
- Secret for Hugging Face token
- GPU resource requests
- `/health` readiness/liveness checks

### Helm Chart Example

Example chart:

```text
examples/deployment/chart-helm/
```

Important files:

- `Chart.yaml`
- `values.yaml`
- `templates/deployment.yaml`
- `templates/service.yaml`
- `templates/pvc.yaml`
- `templates/secrets.yaml`
- `templates/configmap.yaml`
- `templates/hpa.yaml`
- `templates/job.yaml`

The default Helm values use:

```text
image.repository: vllm/vllm-openai
image.tag: latest
containerPort: 8000
servicePort: 80
```

The default container command is a `vllm serve` invocation using `/data/` as the model path.

The chart supports:

- replicas
- resource requests and limits
- GPU model node affinity
- PVC-backed model storage
- S3 model download init flow
- extra containers
- custom init containers
- autoscaling
- probes on `/health`

### SageMaker

Relevant files:

- `vllm/entrypoints/serve/sagemaker/api_router.py`
- `examples/deployment/sagemaker-entrypoint.sh`
- Docker target `vllm-sagemaker`

SageMaker support includes:

- `/ping`
- inference route mapping
- SageMaker-specific entrypoint target

### Ray Serving

Examples:

```text
examples/ray_serving/
```

These demonstrate Ray cluster and Ray Serve patterns, including multi-node serving and elastic expert-parallel examples.

### Disaggregated Serving

Relevant files:

```text
vllm/entrypoints/serve/disagg/
examples/disaggregated/
docs/assets/features/disagg_prefill/
```

Disaggregated serving separates parts of the serving workflow, such as prefill/decode roles or proxy-driven request routing.

### nginx

Primary doc:

```text
docs/deployment/nginx.md
```

This covers reverse proxy deployment patterns in front of the vLLM HTTP server.

### Integrations

Deployment integration docs include:

```text
docs/deployment/integrations/
docs/deployment/frameworks/
```

Examples include:

- KServe
- KubeRay
- KAITO
- AIBrix
- production-stack
- llm-d
- llama-stack
- LiteLLM
- Open WebUI
- Streamlit
- BentoML
- Runpod
- Modal
- SkyPilot
- Triton

## Networking and Serving Data Flow

A typical OpenAI-compatible chat request flows like this:

```text
Client
  -> HTTP POST /v1/chat/completions
  -> FastAPI route in chat_completion/api_router.py
  -> handler from app.state
  -> OpenAIServingChat.create_chat_completion
  -> request validation and model validation
  -> OnlineRenderer renders chat messages into engine input
  -> engine_client.generate or related async engine call
  -> async output stream or final result
  -> OpenAI-compatible response object
  -> JSONResponse or StreamingResponse
```

A pooling request follows a similar pattern:

```text
Client
  -> /v1/embeddings or /score or /rerank
  -> pooling api_router.py
  -> pooling serving class
  -> pooling IO processor
  -> engine_client
  -> pooler output formatting
  -> JSONResponse
```

Realtime speech uses WebSockets:

```text
Client
  -> WebSocket /v1/realtime
  -> realtime api_router.py
  -> connection/session handling
  -> serving class
  -> engine interaction
  -> streamed WebSocket events
```

## Operational Boundaries

The serving layer does not directly format model tensors. Its responsibility is to:

- expose HTTP/WebSocket routes
- validate protocol schemas
- authenticate and log requests
- render prompts into engine-ready inputs
- call the async engine
- stream or serialize responses
- expose operational routes such as `/health` and `/metrics`

The model-side tensor preparation begins after the async engine scheduler selects work for model execution. That lower-level path is documented separately in `MODEL_INPUT_PREPARATION_ARCHITECTURE.md`.

## Practical File Map

Use this map when navigating the repository.

```text
vllm/entrypoints/cli/serve.py
  CLI entrypoint for vllm serve

vllm/entrypoints/openai/api_server.py
  Main FastAPI app and engine bootstrap

vllm/entrypoints/launcher.py
  Uvicorn server lifecycle

vllm/entrypoints/openai/*/api_router.py
  OpenAI-compatible route registration

vllm/entrypoints/openai/*/protocol.py
  OpenAI-compatible request/response schemas

vllm/entrypoints/openai/*/serving.py
  Protocol-to-engine serving logic

vllm/entrypoints/pooling/
  Embedding, classify, score, rerank, and pooling endpoints

vllm/entrypoints/speech_to_text/
  Transcription, translation, and realtime speech endpoints

vllm/entrypoints/anthropic/
  Anthropic-compatible messages API

vllm/entrypoints/serve/
  Shared service endpoints and utilities

docker/
  Dockerfiles, bake files, runtime entrypoints

.github/workflows/
  GitHub Actions automation and lightweight checks

.buildkite/
  Heavy CI, images, hardware tests, releases, benchmarks

docs/deployment/
  Deployment docs

examples/deployment/
  Deployment examples, including Helm and SageMaker
```

## Architectural Takeaway

vLLM's operational architecture is intentionally modular:

- CLI parsing is separate from server construction.
- Server construction is separate from Uvicorn lifecycle management.
- API protocol schemas are separate from serving logic.
- Shared operational routes are separated from model-task-specific routes.
- Docker build targets separate build, test, runtime, platform, and release concerns.
- CI is split between GitHub Actions for lightweight repository automation and Buildkite for heavyweight image, hardware, and release workflows.
- Deployment examples are kept as docs and manifests rather than hardwired into the runtime server.

The result is a serving stack that can expose many protocols while keeping the engine boundary consistent.

