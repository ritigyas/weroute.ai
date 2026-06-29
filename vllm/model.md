# vLLM Model-Side Input Preparation Architecture

## Purpose

This document explains how vLLM prepares model-side inputs before model execution. It focuses only on local model execution internals: prompt normalization, request state, tensor formatting, multimodal embedding handling, attention metadata, and the final model forward call.

This document intentionally excludes network serving, API protocols, Docker, CI/CD, deployment, and external orchestration.

## High-Level Summary

vLLM converts user-facing prompts into a compact execution format used by the model runner. The central idea is:

1. Normalize incoming prompts into typed engine inputs.
2. Store active requests in a persistent batch structure.
3. Use scheduler output to decide which tokens are executed in the current step.
4. Flatten request-major data into token-major tensors.
5. Build attention, KV-cache, pooling, sampling, and multimodal metadata.
6. Call the model with a small canonical interface:

```python
model_output = self._model_forward(
    input_ids=input_ids,
    positions=positions,
    intermediate_tensors=intermediate_tensors,
    inputs_embeds=inputs_embeds,
    **model_kwargs,
)
```

Most model families eventually receive some combination of:

- `input_ids`
- `positions`
- `inputs_embeds`
- `intermediate_tensors`
- model-specific keyword arguments such as `encoder_outputs`

## Model Families Using This Pipeline

The V1 worker/model-runner input preparation path is used by several categories of models.

### Text Generation Models

Decoder-only causal language models use the basic path. These models receive flattened token IDs and positions, then produce hidden states. The runner selects the required hidden states and calls `compute_logits`.

Relevant interface:

- `VllmModel`
- `VllmModelForTextGeneration`

Core file:

- `vllm/model_executor/models/interfaces_base.py`

### Multimodal Generation Models

Multimodal models accept text plus media-like inputs such as images, audio, video, or prompt embeddings. Their non-text inputs are encoded into embeddings and merged into the decoder input stream.

Relevant interface:

- `SupportsMultiModal`

Core file:

- `vllm/model_executor/models/interfaces.py`

Important behavior:

- Multimodal processors create placeholder token ranges in `prompt_token_ids`.
- Model-side encoder execution produces multimodal embeddings.
- The runner gathers embeddings that overlap the current scheduled token window.
- The final model usually receives `inputs_embeds` instead of plain `input_ids`.

### Encoder-Decoder Models

Encoder-decoder models split the input into encoder and decoder portions. The encoder side is processed separately, and the decoder receives `encoder_outputs` through model kwargs.

Core functions:

- `build_enc_dec_input`
- `split_enc_dec_input`

Core file:

- `vllm/inputs/engine.py`

### Pooling Models

Pooling models support embedding, classification, scoring, reranking, and late-interaction tasks. They share much of the same token preparation path but do not use sampling metadata in the same way as generation models.

Relevant interface:

- `VllmModelForPooling`

Core files:

- `vllm/model_executor/models/interfaces_base.py`
- `vllm/model_executor/models/adapters.py`
- `vllm/v1/worker/gpu_input_batch.py`

### Hybrid, Mamba, and Position-Specialized Models

The same preparation path also supports models with special state or positional requirements:

- Mamba and hybrid models
- M-RoPE models, such as Qwen-VL style models
- XD-RoPE models
- Attention-free models
- Speculative decoding paths

Core files:

- `vllm/v1/worker/gpu_model_runner.py`
- `vllm/v1/worker/mamba_utils.py`

## Core Files

### `vllm/inputs/engine.py`

Defines the typed input structures accepted by the engine:

- `TokensInput`
- `EmbedsInput`
- `MultiModalInput`
- `MultiModalEncDecInput`
- `EncoderDecoderInput`

It also contains helpers for encoder-decoder preparation:

- `build_enc_dec_input`
- `split_enc_dec_input`

This file describes the data shape before requests enter the model runner.

### `vllm/inputs/preprocess.py`

Contains `InputPreprocessor`, which converts raw prompt forms into the typed structures from `engine.py`.

Important methods:

- `_process_text`
- `_process_tokens`
- `_process_embeds`
- `_process_multimodal`
- `_process_encoder_decoder_prompt`
- `_process_decoder_only_prompt`
- `preprocess`

This is where raw text becomes token IDs, token prompts are truncated, prompt embeddings are accepted, and multimodal processors are invoked.

### `vllm/v1/worker/gpu_input_batch.py`

Defines persistent batch state for GPU execution.

Important classes:

- `CachedRequestState`
- `InputBatch`

This file stores active request data in CPU/GPU-friendly layouts. It is responsible for:

- prompt token IDs
- output token IDs
- prompt embeddings
- per-position token/embedding masks
- request-to-row mapping
- block table rows
- sampling parameters
- pooling parameters
- LoRA request mapping

### `vllm/v1/worker/gpu_model_runner.py`

This is the main execution preparation file.

Important methods:

- `_update_states`
- `_prepare_inputs`
- `_prepare_input_ids`
- `_build_attention_metadata`
- `_execute_mm_encoder`
- `_gather_mm_embeddings`
- `_preprocess`
- `execute_model`

This file converts persistent request state plus scheduler output into model-ready tensors.

### `vllm/v1/worker/block_table.py`

Defines block table data structures for KV-cache addressing.

Important classes and methods:

- `MultiGroupBlockTable`
- `add_row`
- `commit_block_table`
- `compute_slot_mapping`
- `get_device_tensor`

The block table maps logical request positions to KV-cache blocks and slots used by attention backends.

### `vllm/model_executor/models/interfaces_base.py`

Defines model capability interfaces:

- `VllmModel`
- `VllmModelForTextGeneration`
- `VllmModelForPooling`
- `is_vllm_model`
- `is_text_generation_model`
- `is_pooling_model`

These interfaces describe what the runner expects a model to implement.

### `vllm/model_executor/models/interfaces.py`

Defines multimodal and specialized model interfaces.

Important interface:

- `SupportsMultiModal`

Important methods and flags:

- `embed_multimodal`
- `get_language_model`
- `requires_raw_input_tokens`
- `supports_multimodal_raw_input_only`

## Input Normalization

Input normalization begins in `InputPreprocessor.preprocess`.

For decoder-only models:

```python
return self._process_decoder_only_prompt(
    parse_dec_only_prompt(prompt),
    tokenization_kwargs=tokenization_kwargs,
)
```

For encoder-decoder models:

```python
return self._process_encoder_decoder_prompt(
    parse_enc_dec_prompt(prompt),
    tokenization_kwargs,
)
```

The resulting object is an `EngineInput`, not yet a model tensor. It may contain token IDs, prompt embeddings, multimodal metadata, or encoder-decoder splits.

## Engine Input Types

### Token Input

`TokensInput` represents prompts that are already tokenized or were tokenized from text.

Important fields:

- `type: "token"`
- `prompt_token_ids: list[int]`
- optional `prompt`
- optional `cache_salt`

### Embedding Input

`EmbedsInput` represents prompts where embeddings are supplied directly.

Important fields:

- `type: "embeds"`
- `prompt_embeds: torch.Tensor`
- optional `prompt_token_ids`
- optional `is_token_ids`

The optional `prompt_token_ids` and `is_token_ids` fields support mixed-mode inputs where some positions are normal token IDs and other positions are precomputed embedding rows.

### Multimodal Input

`MultiModalInput` represents text plus processed multimodal data.

Important fields:

- `type: "multimodal"`
- `prompt_token_ids`
- `mm_kwargs`
- `mm_hashes`
- `mm_placeholders`

The `prompt_token_ids` include placeholder tokens where multimodal embeddings will later be inserted.

### Encoder-Decoder Input

`EncoderDecoderInput` separates the input into:

- `encoder_prompt`
- `decoder_prompt`

`build_enc_dec_input` ensures decoder input IDs start with the decoder start token unless configured otherwise.

## Persistent Request State

When the scheduler sends work to the GPU model runner, request data is stored in `CachedRequestState`.

Important fields:

```python
req_id: str
prompt_token_ids: list[int] | None
prompt_embeds: torch.Tensor | None
prompt_is_token_ids: list[bool] | None
mm_features: list[MultiModalFeatureSpec]
sampling_params: SamplingParams | None
pooling_params: PoolingParams | None
block_ids: tuple[list[int], ...]
num_computed_tokens: int
output_token_ids: list[int]
```

`CachedRequestState.num_tokens` is computed as:

```python
num_prompt_tokens + len(output_token_ids)
```

This state is request-oriented. It tracks the full logical sequence for each request.

## Persistent Batch State

`InputBatch` stores request data in dense rows. Each active request is assigned a row index.

Important tensors and arrays:

```python
token_ids_cpu_tensor: [max_num_reqs, max_model_len], int32
is_token_ids_tensor: [max_num_reqs, max_model_len], bool
num_tokens_no_spec_cpu_tensor: [max_num_reqs], int32
num_prompt_tokens_cpu_tensor: [max_num_reqs], int32
num_computed_tokens_cpu_tensor: [max_num_reqs], int32
```

The row-major CPU table is intentionally persistent. It allows vLLM to update only changed requests instead of rebuilding every batch from scratch.

When a request is added through `InputBatch.add_request`:

1. Prompt token IDs are copied into the request row.
2. Existing output token IDs are appended after prompt IDs.
3. `is_token_ids` is set to distinguish token-ID positions from embedding positions.
4. Prompt embeddings are stored separately in `req_prompt_embeds`.
5. KV-cache block IDs are inserted into the block table.
6. Sampling or pooling metadata is stored.
7. LoRA mapping is stored if present.

## Scheduler Output to Flat Model Tensors

The main conversion happens in `GPUModelRunner._prepare_inputs`.

The scheduler tells the runner how many tokens to run for each request:

```python
num_scheduled_tokens = [tokens_for_req_0, tokens_for_req_1, ...]
```

The runner converts this request-major information into token-major tensors.

Example:

```text
num_scheduled_tokens = [2, 5, 3]
```

The flattened request index tensor becomes:

```text
req_indices = [0, 0, 1, 1, 1, 1, 1, 2, 2, 2]
```

The cumulative query locations become:

```text
query_start_loc = [0, 2, 7, 10]
```

The per-request local positions become:

```text
query_pos = [0, 1, 0, 1, 2, 3, 4, 0, 1, 2]
```

The absolute position for each scheduled token is:

```python
positions = num_computed_tokens_for_request + query_pos
```

To gather token IDs from the row-major CPU table, vLLM computes:

```python
token_indices = positions_np + req_indices * max_model_len
```

Then it gathers into the flat input buffer:

```python
torch.index_select(
    self.input_batch.token_ids_cpu_tensor.flatten(),
    0,
    token_indices_tensor,
    out=self.input_ids.cpu[:total_num_scheduled_tokens],
)
```

The result is a flat `input_ids` tensor ordered exactly as the model will execute tokens.

## Main Model-Side Tensor Formats

### `input_ids`

Flat token IDs.

Typical shape:

```text
[num_input_tokens]
```

Typical dtype:

```text
torch.int32
```

Used by text-only models and by multimodal models that require raw token IDs.

### `positions`

Flat position IDs.

Typical shape:

```text
[num_input_tokens]
```

Typical dtype:

```text
torch.int64
```

For M-RoPE models:

```text
[3, num_input_tokens]
```

For XD-RoPE models:

```text
[xdrope_dim, num_input_tokens]
```

### `inputs_embeds`

Flat embedding tensor.

Typical shape:

```text
[num_input_tokens, hidden_size]
```

Typical dtype:

```text
model dtype, such as float16 or bfloat16
```

Used when:

- multimodal embeddings are merged into the input stream
- prompt embeddings are supplied directly
- token IDs must be converted to embeddings outside the model forward path

### `query_start_loc`

Cumulative token offsets per request.

Shape:

```text
[num_reqs + 1]
```

Example:

```text
[0, 2, 7, 10]
```

Used by attention kernels to identify each request's query token range inside the flat token tensor.

### `seq_lens`

Current sequence length per request.

Shape:

```text
[num_reqs]
```

Computed as:

```python
num_computed_tokens + num_scheduled_tokens
```

### `block_table`

Maps each request to KV-cache block IDs.

Shape depends on KV-cache group and maximum blocks per request.

Used by attention backends to locate cached keys and values.

### `slot_mapping`

Maps each scheduled token to the KV-cache slot where its key/value should be read or written.

Shape:

```text
[num_input_tokens]
```

Generated by:

```python
self.input_batch.block_table.compute_slot_mapping(...)
```

## Attention Metadata

`GPUModelRunner._build_attention_metadata` creates attention backend metadata from common tensors:

- `query_start_loc`
- `seq_lens`
- CPU sequence lengths
- number of requests
- number of tokens
- max query length
- max sequence length
- block table
- slot mapping
- positions
- encoder sequence lengths for cross-attention
- multimodal prefix bidirectional ranges when needed

This metadata is then registered in the forward context by `set_forward_context`.

The model layers do not manually receive all attention tensors as function arguments. Instead, attention layers access the metadata through the forward context.

## Multimodal Input Preparation

Multimodal preparation has two phases.

### Encoder Phase

`GPUModelRunner._execute_mm_encoder` batches multimodal inputs by modality and calls:

```python
model.embed_multimodal(**mm_kwargs_batch)
```

The returned embeddings may be:

- a 3D tensor with batch dimension
- a list of 2D tensors
- a tuple of 2D tensors

Each item corresponds to one multimodal input item.

Outputs are cached by multimodal hash:

```python
self.encoder_cache[mm_hash] = output
```

Prompt embeddings use a shortcut: if the modality is `prompt_embeds`, the tensor is already in model embedding space and no encoder is run.

### Gathering Phase

`GPUModelRunner._gather_mm_embeddings` selects only embeddings whose placeholder ranges overlap the scheduled token window.

It returns:

```python
mm_embeds: list[torch.Tensor]
is_mm_embed: torch.Tensor
```

`is_mm_embed` is a boolean mask over scheduled tokens. It marks positions where multimodal embeddings should replace or augment text token embeddings.

### Merging Phase

In `_preprocess`, multimodal inputs are merged through:

```python
inputs_embeds_scheduled = self.model.embed_input_ids(
    self.input_ids.gpu[:num_scheduled_tokens],
    multimodal_embeddings=mm_embeds,
    is_multimodal=is_mm_embed,
)
```

The result is copied into:

```python
self.inputs_embeds.gpu[:num_scheduled_tokens]
```

The model then receives `inputs_embeds` instead of plain token IDs, unless the model declares that it requires raw input tokens.

## Prompt Embedding Input Preparation

When prompt embeddings are enabled, `InputBatch` stores embeddings separately:

```python
req_prompt_embeds: dict[int, torch.Tensor]
```

During `_prepare_inputs`, the runner copies only the scheduled slice of each request's prompt embeddings into the flat `inputs_embeds` buffer.

Mixed-mode inputs use `is_token_ids`:

- `True`: this position is a normal token ID and should be embedded by the model.
- `False`: this position already has a precomputed embedding row.

In `_preprocess`, token-ID positions are embedded, while precomputed embedding positions are preserved.

## Encoder-Decoder Handling

Encoder-decoder models are detected through model config:

```python
self.model_config.is_encoder_decoder
```

Input preprocessing builds separate encoder and decoder prompts.

During model-side preprocessing:

1. Encoder multimodal or token inputs are run first.
2. Encoder outputs are collected.
3. Decoder model kwargs are updated:

```python
model_kwargs.update({"encoder_outputs": encoder_outputs})
```

Cross-attention metadata also receives encoder sequence lengths through `_get_encoder_seq_lens`.

## Sampling Metadata

Generation models use `SamplingMetadata`, built by `InputBatch._make_sampling_metadata`.

It contains:

- temperature
- top-p
- top-k
- penalties
- generators
- allowed token masks
- bad word token IDs
- prompt token IDs when penalties require them
- output token IDs when logits processors require them
- speculative decoding token IDs

Sampling metadata is not part of the model forward call itself. It is used after hidden states are converted to logits.

## Pooling Metadata

Pooling models use `PoolingMetadata`, built by:

```python
InputBatch.get_pooling_metadata()
```

It contains:

- prompt lengths
- optional prompt token IDs
- pooling parameters
- pooling states

After the model forward pass, pooling models call:

```python
model.pooler(
    hidden_states=hidden_states,
    pooling_metadata=pooling_metadata,
)
```

This replaces the generation path where logits and sampling would normally happen.

## Final Execution Path

The final execution happens in `GPUModelRunner.execute_model`.

The important sequence is:

1. `_update_states`
2. `_prepare_inputs`
3. `_build_attention_metadata`
4. `_preprocess`
5. `set_forward_context`
6. `_model_forward`
7. postprocess hidden states
8. either compute logits or pool hidden states

The actual model forward call is:

```python
model_output = self._model_forward(
    input_ids=input_ids,
    positions=positions,
    intermediate_tensors=intermediate_tensors,
    inputs_embeds=inputs_embeds,
    **model_kwargs,
)
```

For text generation:

```python
sample_hidden_states = hidden_states[logits_indices]
logits = self.model.compute_logits(sample_hidden_states)
```

For pooling:

```python
return self._pool(
    hidden_states,
    num_scheduled_tokens,
    num_scheduled_tokens_np,
    kv_connector_output,
)
```

## End-to-End Data Shape Example

Assume three active requests have scheduled token counts:

```text
request A: 2 tokens
request B: 5 tokens
request C: 3 tokens
```

The scheduler view is request-major:

```text
[2, 5, 3]
```

The model runner view becomes token-major:

```text
req_indices:
[0, 0, 1, 1, 1, 1, 1, 2, 2, 2]

query_pos:
[0, 1, 0, 1, 2, 3, 4, 0, 1, 2]

query_start_loc:
[0, 2, 7, 10]
```

If request B already has 100 computed tokens, its scheduled positions become:

```text
[100, 101, 102, 103, 104]
```

All scheduled tokens across all requests are then gathered into:

```text
input_ids: [10 flat token IDs]
positions: [10 flat positions]
```

If multimodal or prompt embeddings are involved:

```text
inputs_embeds: [10, hidden_size]
```

Attention metadata tells kernels how to split those 10 flat tokens back into their original request sequences.

## Architectural Takeaway

The model-side input preparation pipeline is built around a deliberate separation:

- User-facing inputs are flexible.
- Internal request state is persistent and row-major.
- Per-step execution tensors are flat and token-major.
- Attention metadata reconstructs request boundaries and KV-cache locations.
- Multimodal and embedding inputs are normalized into `inputs_embeds`.
- Text-only inputs keep `input_ids` for better performance.

This lets vLLM support many model types while keeping the final forward interface small and predictable.
