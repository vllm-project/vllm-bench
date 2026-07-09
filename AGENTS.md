# AGENTS.md

## Project Overview

Rust rewrite of `vllm bench serve` — a high-performance benchmark client for vLLM serving endpoints. Standalone binary, no Python dependency at runtime.

The crate ships both a `[lib]` target (`vllm_bench`) and the `[[bin]]` (`vllm-bench`). The library exposes the full run path so external crates can register custom backends without forking (see "Custom backends" below); the binary is a thin `vllm_bench::run_cli()` wrapper (the `#[global_allocator]` lives in the binary, not the lib).

## Build & Test

```bash
# Build release binary
cargo build --release

# Run all tests
cargo test

# Run ignored integration tests (requires network for tokenizer download)
cargo test -- --ignored
```

## Architecture

- `src/main.rs` — Thin binary entry point (mimalloc global allocator + `vllm_bench::run_cli()`)
- `src/lib.rs` — Library root: `pub` modules, public extension API re-exports (`CustomBackend`, `register_backend`, `RequestFuncInput/Output`, `build_headers`), `run_cli`/`run` (rlimit bump, tokio runtime, mode dispatch: compare/sweep/multi-run/multi-turn/single)
- `src/cli.rs` — clap derive CLI args (~50+ flags)
- `src/config.rs` — Validated config from CLI; `GoodputConfig`, `RampUpConfig`, sampling param merging
- `src/error.rs` — `BenchError` enum (Http, Json, Tokenizer, Config, EndpointTimeout, Backend, Io)
- `src/benchmark.rs` — Core benchmark orchestrator (spawn-per-request with tokio + Semaphore; fetches speculative decoding metrics from `/metrics`)
- `src/multi_turn.rs` — Multi-turn conversation orchestrator (channel-based worker pool, sequential turns per conversation)
- `src/sweep.rs` — Concurrency/rate parameter sweep (`--sweep-max-concurrency`, `--sweep-request-rate`)
- `src/multi_run.rs` — N-run aggregation with mean/std/min/max/CV (`--num-runs`)
- `src/compare.rs` — Side-by-side diff of two result JSON files (`--compare`)
- `src/tokenizer.rs` — `TokenizerKind` enum: Local(HuggingFace), Tiktoken, OR Server-side `/tokenize`+`/detokenize` fallback
- `src/tiktoken.rs` — Tiktoken BPE loader (`.tiktoken`/`.model` files; built-in encodings o200k_base/cl100k_base; pat_str extraction from Python source)
- `src/rate_control.rs` — Gamma/Poisson request scheduling + linear/exponential ramp-up
- `src/ready_checker.rs` — Endpoint readiness with retry
- `src/backends/` — Backend implementations (enum dispatch, not trait objects)
  - `mod.rs` — `Backend` enum (incl. `Custom(Arc<dyn CustomBackend>)`), `CustomBackend` trait (extension point), `RequestFuncInput`/`RequestFuncOutput` (includes `messages` field for multi-turn), `get_backend` factory
  - `registry.rs` — Process-global registry for external custom backends (`register_backend`, `lookup`, `set_active`/`active_custom`); keeps `get_backend(BackendKind)` signature unchanged
  - `streaming.rs` — SSE parser (`StreamedResponseHandler`) with speculative JSON parse for split TCP segments
  - `openai_completions.rs` — `/v1/completions` backend
  - `openai_chat.rs` — `/v1/chat/completions` backend (uses `input.messages` when set; zero-copy raw JSON payload for multimodal)
  - `pooling.rs` — Non-streaming pooling/embedding backends: `openai-embeddings`, `openai-embeddings-chat`, `vllm-pooling`, `vllm-rerank`
- `src/datasets/random.rs` — Random dataset generation with rayon parallelism
- `src/datasets/random_mm.rs` — Random multimodal dataset (synthetic JPEG images, bucket config sampling, pre-serialized JSON fragments); `--enable-multimodal-chat` pre-builds the chat `messages` array at dataset time (mirrors Python's `apply_multimodal_chat_transformation`)
- `src/datasets/sharegpt.rs` — ShareGPT JSON loader + HuggingFace Hub auto-download with caching
- `src/datasets/sonnet.rs` — Sonnet dataset (built-in Shakespeare sonnets via `include_str!("../../sonnet.txt")`; controllable token length + shared prefix; mirrors Python `SonnetDataset`)
- `src/datasets/speed_bench.rs` — NVIDIA SPEED-Bench loader (HF datasets-server API, 6 configs, 11 categories, local cache)
- `src/datasets/hf_dataset.rs` — Generic HuggingFace dataset loader (datasets-server API, column auto-detection)
- `src/datasets/custom.rs` — Custom JSONL dataset (`{"prompt": ..., "output_tokens": ...}` per line; `--custom-output-len -1` uses per-line output_tokens; prompts always sent raw — no client-side chat template)
- `src/datasets/prefix_repetition.rs` — Prefix repetition dataset (N shared prefixes × fresh random suffixes, standard prefix-cache stress; mirrors Python `PrefixRepetitionRandomDataset`)
- `src/datasets/random_rerank.rs` — Random rerank dataset (one query + batched documents per request for `vllm-rerank`; `--no-reranker` for embedding-based scoring; mirrors Python `RandomDatasetForReranking`)
- `src/datasets/multi_turn.rs` — Multi-turn synthetic generator + ShareGPT multi-turn loader (3-tier prefix sharing: global/conversation/unique-suffix; `per_turn_input_len`)
- `src/metrics/mod.rs` — `BenchmarkMetrics` and `MultiTurnMetrics` structs
- `src/metrics/calculator.rs` — TTFT/TPOT/ITL/E2EL/throughput stats, goodput SLO checking, peak concurrency, `calculate_multi_turn_metrics`
- `src/metrics/steady_state.rs` — Steady-state window detection (in-flight concurrency plateau via two-pointer start/end merge) + plateau throughput/TTFT/TPOT; gated on `--max-concurrency` set + `--request-rate inf` (closed-loop)
- `src/output/console.rs` — Terminal output matching Python format + multi-turn per-turn breakdown
- `src/output/json.rs` — JSON result file (compatible with Python schema) + multi-turn JSON with `per_turn_metrics`

## Key Design Decisions

- **Enum dispatch** for backends (avoids async trait object issues with `dyn`); the one exception is `Backend::Custom(Arc<dyn CustomBackend>)`, where externally-registered backends pay a small boxing/dynamic-dispatch cost via `async-trait`. Built-ins stay zero-cost.
- **Custom backends are compile-time, registered into a process-global registry**: external crates depend on the `vllm_bench` lib, `register_backend("name", Arc::new(...))` before `run_cli()`, and select with `--backend name`. The global registry (`backends/registry.rs`) lets `get_backend(BackendKind)` and every call site stay parameter-free. `--backend` is a `String` resolved in `config.rs` (built-in via `BackendKind::from_name`, else registry lookup, else error listing available names); backend metadata (`is_pooling`, OpenAI-compat, endpoint, display name) is read from the trait once and cached on `BenchConfig` (`is_pooling`, `backend_name`).
- **reqwest http1_only()** to match Python aiohttp behavior
- **rayon** for parallel dataset generation (key perf win over Python)
- **mimalloc** global allocator to reduce contention at 1400+ concurrency (page-agnostic; works on aarch64 64K-page kernels where jemalloc aborts with `LG_PAGE=12` builds)
- **Arc\<str\> prompts** zero-copy sharing across tokio tasks (~3GB savings at 100k prompts with 8k-token inputs)
- **Spawn-per-request** `tokio::spawn` + `Semaphore` (matches Python asyncio pattern)
- **Speculative JSON parse** in SSE handler — detects complete JSON before `\n\n` arrives, improving TTFT/ITL accuracy when TCP segments split
- **Tokenizer fallback chain**: Local HF → Tiktoken (`.tiktoken`/`.model` + built-in encodings) → Server-side `/tokenize`+`/detokenize`. Blocking HTTP in rayon threads for server fallback.
- **hf-hub** for downloading tokenizers and datasets from HuggingFace Hub
- **Pre-serialized mm fragments** (`Arc<str>`) for multimodal: image content stored as JSON strings, zero-copy concatenated into payload — avoids deep-cloning ~200KB+ base64 per request
- **Steady-state metrics** (default-on in closed-loop): measure throughput/TTFT/TPOT only over the saturated plateau to cut run-to-run variance at high concurrency; `steady_state` is an `Option` in JSON (`#[serde(default)]` for backward compat), null when the scope gate fails or `--no-steady-state`
- **`--prompt-token-ids`** (random dataset only): send token-ID arrays instead of text to skip server-side tokenization; also skips the token-length verification pass (counts exact by construction)
- **`--random-range-ratio`** follows Python semantics: lengths sampled uniformly from `[len*(1-r), len*(1+r)]`, default `0.0` = fixed; accepts a float in `[0,1)` or `'{"input": r1, "output": r2}'`. (The pre-2026-07 Rust-only form `[len*r, len]` with default 1.0 is rejected with a migration hint.)
- **`prompt_list`** (`Arc<[Arc<str>]>` on `SampleRequest`/`RequestFuncInput`): multiple inputs per request for pooling backends — embeddings batches (`--random-batch-size`) send `"input": [...]`, rerank sends `[0]` as query + `[1..]` as documents
- JSON output schema must match Python `vllm bench serve` exactly

## Common Issues

- **localhost vs 127.0.0.1**: Some systems resolve `localhost` to IPv6 `::1` while vLLM listens on IPv4 only. Use `127.0.0.1` or the actual hostname.
- **Models without tokenizer.json** (e.g., `nvidia/Kimi-K2.5-NVFP4`): Automatically falls back to server-side tokenization. Can also use `--tokenizer` to point to a model with `tokenizer.json`.
- **usage.completion_tokens parsing**: vLLM sends final usage chunk with `"choices":[]` (empty array). The usage `if` must be separate from the choices `if` (not `else if`).

## Custom Backends (external extensions)

Add a backend without forking by depending on the `vllm_bench` library and registering an
implementation of the `CustomBackend` trait. Examples:

- `examples/custom_backend.rs` — OpenAI-compatible streaming backend
  (`cargo run --example custom_backend -- --backend my-llm ...`).
- `examples/fake_backend.rs` + `examples/fake_server.py` — a **non-OpenAI** ("FakeGen")
  protocol end-to-end. Run `examples/run_demo.sh` to spin up the FastAPI server, build the
  custom backend, and benchmark it with `--backend fakegen` (proves non-OpenAI endpoints work).

```rust
use std::sync::Arc;
use vllm_bench::{build_headers, CustomBackend, RequestFuncInput, RequestFuncOutput, Result};

struct MyBackend;

#[async_trait::async_trait]
impl CustomBackend for MyBackend {
    fn default_endpoint(&self) -> String { "/v1/completions".into() }
    // Optional: fn is_pooling(&self) -> bool / fn is_openai_compatible(&self) -> bool
    async fn send_request(
        &self,
        input: &RequestFuncInput,
        client: &reqwest::Client,
    ) -> Result<RequestFuncOutput> {
        // Build payload, send via `client`, fill timing metrics (ttft/itl/latency/output_tokens).
        // Reuse `build_headers(...)` and `vllm_bench::StreamedResponseHandler`.
        Ok(RequestFuncOutput { success: true, ..Default::default() })
    }
}

fn main() -> anyhow::Result<()> {
    vllm_bench::register_backend("my-llm", Arc::new(MyBackend));
    vllm_bench::run_cli() // `--backend my-llm` now selects it; built-ins still work
}
```

## Typical Usage

```bash
# Embedding benchmark (openai-embeddings, 8 inputs batched per request)
./target/release/vllm-bench \
  --backend openai-embeddings \
  --base-url http://gb200-10:30000 \
  --model BAAI/bge-large-en-v1.5 \
  --dataset-name random \
  --random-input-len 512 \
  --random-batch-size 8 \
  --num-prompts 1000 \
  --save-result

# vLLM rerank benchmark (one query + 8 documents per request)
./target/release/vllm-bench \
  --backend vllm-rerank \
  --base-url http://gb200-10:30000 \
  --model BAAI/bge-reranker-v2-m3 \
  --dataset-name random-rerank \
  --random-input-len 512 \
  --random-batch-size 8 \
  --num-prompts 500 \
  --save-result

# Prefix-cache stress (10 shared prefixes, 256+256 tokens)
./target/release/vllm-bench \
  --backend vllm \
  --base-url http://gb200-10:30000 \
  --model nvidia/Kimi-K2.5-NVFP4 \
  --dataset-name prefix_repetition \
  --prefix-repetition-prefix-len 256 \
  --prefix-repetition-suffix-len 256 \
  --prefix-repetition-num-prefixes 10 \
  --num-prompts 1000

# Custom JSONL workload ({"prompt": ..., "output_tokens": ...} per line)
./target/release/vllm-bench \
  --backend openai-chat \
  --base-url http://gb200-10:30000 \
  --model nvidia/Kimi-K2.5-NVFP4 \
  --dataset-name custom \
  --dataset-path workload.jsonl \
  --custom-output-len -1 \
  --num-prompts 1000

# Random dataset
./target/release/vllm-bench \
  --backend vllm \
  --base-url http://gb200-10:30000 \
  --model nvidia/Kimi-K2.5-NVFP4 \
  --dataset-name random \
  --random-input-len 8192 \
  --random-output-len 1024 \
  --ignore-eos \
  --num-prompts 4096 \
  --percentile-metrics "ttft,tpot,itl,e2el" \
  --save-result \
  --max-concurrency 1400

# Random multimodal dataset (VLM benchmark)
./target/release/vllm-bench \
  --backend openai-chat \
  --base-url http://gb200-10:30000 \
  --model Qwen/Qwen2.5-VL-7B-Instruct \
  --dataset-name random-mm \
  --random-input-len 512 \
  --random-output-len 128 \
  --num-prompts 100 \
  --random-mm-base-items-per-request 1 \
  --random-mm-limit-mm-per-prompt '{"image": 1, "video": 0}' \
  --random-mm-bucket-config '{(1024, 800, 1): 1.0}'

# HuggingFace dataset (WildChat)
./target/release/vllm-bench \
  --backend openai-chat \
  --base-url http://gb200-10:30000 \
  --model nvidia/Kimi-K2.5-NVFP4 \
  --dataset-name hf \
  --dataset-path allenai/WildChat-4.8M \
  --hf-split train \
  --num-prompts 1000 \
  --save-result

# HuggingFace dataset (LongBench with subset)
./target/release/vllm-bench \
  --backend openai-chat \
  --base-url http://gb200-10:30000 \
  --model nvidia/Kimi-K2.5-NVFP4 \
  --dataset-name hf \
  --dataset-path THUDM/LongBench \
  --hf-subset narrativeqa \
  --hf-split test \
  --hf-output-len 512 \
  --num-prompts 200
```
