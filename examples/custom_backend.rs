// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright contributors to the vLLM project

//! Minimal example of an externally-provided custom backend.
//!
//! This builds a standalone benchmark binary that depends on `vllm-bench` as a
//! library, registers a custom backend under the name `my-llm`, and hands off to
//! the normal CLI. The custom backend speaks the OpenAI streaming Completions
//! protocol — swap the payload/parse logic for your own wire format.
//!
//! Run it:
//! ```bash
//! cargo run --example custom_backend -- \
//!   --backend my-llm \
//!   --base-url http://127.0.0.1:8000 \
//!   --model my-model \
//!   --dataset-name random --random-input-len 32 --random-output-len 16 \
//!   --num-prompts 8
//! ```
//!
//! Built-in backends still work from the same binary (e.g. `--backend openai`).

use std::sync::Arc;
use std::time::Instant;

use futures::StreamExt;
use vllm_bench::{
    build_headers, trim_bytes, CustomBackend, RequestFuncInput, RequestFuncOutput, Result,
    StreamedResponseHandler,
};

/// A custom generative backend hitting an OpenAI-style `/v1/completions` endpoint.
struct MyBackend;

#[async_trait::async_trait]
impl CustomBackend for MyBackend {
    /// Endpoint appended to `--base-url` when `--endpoint` is not passed.
    fn default_endpoint(&self) -> String {
        "/v1/completions".to_string()
    }

    async fn send_request(
        &self,
        input: &RequestFuncInput,
        client: &reqwest::Client,
    ) -> Result<RequestFuncOutput> {
        let model = input.model_name.as_deref().unwrap_or(&input.model);

        let mut payload = serde_json::json!({
            "model": model,
            "prompt": input.prompt.as_ref(),
            "max_tokens": input.output_len,
            "stream": true,
            "stream_options": { "include_usage": true },
        });
        if input.ignore_eos {
            payload["ignore_eos"] = serde_json::json!(true);
        }
        // Merge user-supplied --extra-body fields.
        if let Some(serde_json::Value::Object(map)) = input.extra_body.as_ref() {
            for (k, v) in map {
                payload[k] = v.clone();
            }
        }

        // `build_headers` adds Content-Type, OPENAI_API_KEY bearer auth,
        // --headers extras, and the x-request-id trace header.
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

        // Parse the SSE stream, timing first-token (TTFT) and inter-token latencies (ITL).
        let mut handler = StreamedResponseHandler::new();
        let mut stream = resp.bytes_stream();
        let mut most_recent = st;
        let mut first = false;
        let mut text = String::new();

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
                let data = trim_bytes(msg.as_bytes());
                let data = match data.strip_prefix(b"data: ") {
                    Some(d) => d,
                    None => continue,
                };
                if data == b"[DONE]" {
                    continue;
                }
                let parsed: serde_json::Value = match serde_json::from_slice(data) {
                    Ok(v) => v,
                    Err(_) => continue,
                };
                if let Some(t) = parsed["choices"][0]["text"].as_str() {
                    if !t.is_empty() {
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
                    }
                }
                if let Some(ct) = parsed["usage"]["completion_tokens"].as_u64() {
                    output.output_tokens = ct as usize;
                }
            }
        }

        output.latency = st.elapsed().as_secs_f64();
        output.generated_text = text;
        output.success = true;
        Ok(output)
    }
}

fn main() -> anyhow::Result<()> {
    vllm_bench::register_backend("my-llm", Arc::new(MyBackend));
    vllm_bench::run_cli()
}
