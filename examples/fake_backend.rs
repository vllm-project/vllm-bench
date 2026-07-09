// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright contributors to the vLLM project

//! A custom backend speaking the made-up "FakeGen" protocol from
//! `examples/fake_server.py` — a demonstration that vllm-bench can drive
//! endpoints that are *not* OpenAI-compatible.
//!
//! Unlike `examples/custom_backend.rs` (which speaks the OpenAI Completions
//! protocol), this backend sends `{"prompt_tokens", "max_tokens"}` and parses a
//! bespoke SSE stream of `{"text", "index"}` token messages.
//!
//! See `examples/run_demo.sh` for an end-to-end runner (starts the FastAPI
//! server, then benchmarks it through this backend).

use std::sync::Arc;
use std::time::Instant;

use futures::StreamExt;
use vllm_bench::{
    build_headers, trim_bytes, CustomBackend, RequestFuncInput, RequestFuncOutput, Result,
    StreamedResponseHandler,
};

struct FakeGenBackend;

#[async_trait::async_trait]
impl CustomBackend for FakeGenBackend {
    /// Endpoint appended to `--base-url` when `--endpoint` is not given.
    fn default_endpoint(&self) -> String {
        "/v1/fakegen".to_string()
    }

    async fn send_request(
        &self,
        input: &RequestFuncInput,
        client: &reqwest::Client,
    ) -> Result<RequestFuncOutput> {
        // The made-up protocol only needs the input size and how many tokens to
        // generate — it does not care about the prompt content.
        let payload = serde_json::json!({
            "prompt_tokens": input.prompt_len,
            "max_tokens": input.output_len,
        });

        let headers = build_headers(
            Some("application/json"),
            &input.extra_headers,
            &input.request_id,
        );

        let mut output = RequestFuncOutput {
            prompt_len: input.prompt_len,
            ..Default::default()
        };

        let st = Instant::now();
        let mut req = client.post(&input.api_url).json(&payload);
        for (k, v) in &headers {
            req = req.header(k, v);
        }

        let resp = match req.send().await {
            Ok(r) => r,
            Err(e) => {
                output.error = e.to_string();
                return Ok(output);
            }
        };
        if !resp.status().is_success() {
            output.error = format!("HTTP {}", resp.status());
            return Ok(output);
        }

        // Parse the FakeGen SSE stream, timing TTFT and inter-token latencies.
        let mut handler = StreamedResponseHandler::new();
        let mut stream = resp.bytes_stream();
        let mut most_recent = st;
        let mut first = false;
        let mut text = String::new();
        let mut n_tokens = 0usize;

        while let Some(chunk) = stream.next().await {
            let chunk = match chunk {
                Ok(c) => c,
                Err(e) => {
                    output.error = e.to_string();
                    return Ok(output);
                }
            };
            let now = Instant::now();
            for msg in handler.add_chunk(&chunk) {
                let data = match trim_bytes(msg.as_bytes()).strip_prefix(b"data: ") {
                    Some(d) => d,
                    None => continue,
                };
                if data == b"[DONE]" {
                    continue;
                }
                let v: serde_json::Value = match serde_json::from_slice(data) {
                    Ok(v) => v,
                    Err(_) => continue,
                };
                // A per-token message: {"text": "...", "index": i}
                if let Some(t) = v["text"].as_str() {
                    if !first {
                        first = true;
                        output.ttft = now.duration_since(st).as_secs_f64();
                    } else {
                        output
                            .itl
                            .push(now.duration_since(most_recent).as_secs_f64());
                    }
                    most_recent = now;
                    text.push_str(t);
                    n_tokens += 1;
                }
                // The final summary message: {"done": true, "generated_tokens": N}
                if let Some(g) = v["generated_tokens"].as_u64() {
                    n_tokens = g as usize;
                }
            }
        }

        output.latency = st.elapsed().as_secs_f64();
        output.output_tokens = n_tokens;
        output.generated_text = text;
        output.success = true;
        Ok(output)
    }
}

fn main() -> anyhow::Result<()> {
    vllm_bench::register_backend("fakegen", Arc::new(FakeGenBackend));
    vllm_bench::run_cli()
}
