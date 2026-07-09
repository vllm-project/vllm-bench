// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright contributors to the vLLM project

//! `vllm-bench` as a library.
//!
//! This exposes the benchmark machinery so that external crates can register
//! their own [`CustomBackend`] implementations and reuse the full CLI/run path
//! without forking the repo. A minimal external binary looks like:
//!
//! ```no_run
//! use std::sync::Arc;
//! # use vllm_bench::{CustomBackend, RequestFuncInput, RequestFuncOutput, Result};
//! # struct MyBackend;
//! # #[async_trait::async_trait]
//! # impl CustomBackend for MyBackend {
//! #     async fn send_request(&self, _i: &RequestFuncInput, _c: &reqwest::Client) -> Result<RequestFuncOutput> { Ok(Default::default()) }
//! #     fn default_endpoint(&self) -> String { "/v1/completions".into() }
//! # }
//! fn main() -> anyhow::Result<()> {
//!     vllm_bench::register_backend("my-llm", Arc::new(MyBackend));
//!     vllm_bench::run_cli() // `--backend my-llm` now selects it
//! }
//! ```

// Crate internals — NOT part of the public API. Only the items re-exported
// below are stable; everything else may change without notice.
mod backends;
mod benchmark;
mod compare;
mod config;
mod datasets;
mod error;
mod metrics;
mod multi_run;
mod multi_turn;
mod output;
mod rate_control;
mod ready_checker;
mod sweep;
mod tiktoken;
mod tokenizer;

// `Cli` appears in `run`'s signature, so this module (and its clap-derived field
// types) must stay reachable — but it is not a stable API surface, hence hidden.
#[doc(hidden)]
pub mod cli;

use anyhow::Context;
use clap::Parser;

use config::BenchConfig;

// --- Public extension API ---
// Custom-backend authors depend on exactly these items.
pub use backends::streaming::{trim_bytes, StreamedResponseHandler};
pub use backends::{
    build_headers, register_backend, CustomBackend, RequestFuncInput, RequestFuncOutput,
};
pub use cli::Cli;
pub use error::{BenchError, Result};

/// Parse CLI args from the process environment and run the benchmark.
///
/// Call any [`register_backend`] before this so custom `--backend <name>`
/// values resolve.
pub fn run_cli() -> anyhow::Result<()> {
    // Raise the open-file soft limit to the hard limit. High-concurrency
    // benchmarks (1024+ requests) easily exceed the default 1024 fd soft limit.
    if let Ok(new) = rlimit::increase_nofile_limit(u64::MAX) {
        if new > 1024 {
            eprintln!("Open-file limit: {new}");
        }
    }

    run(Cli::parse())
}

/// Run the benchmark from an already-parsed [`Cli`].
pub fn run(cli: Cli) -> anyhow::Result<()> {
    // --- Compare mode: no server needed, just diff two JSON files ---
    if let Some(ref files) = cli.compare {
        return compare::compare_results(&files[0], &files[1]).context("Comparison failed");
    }

    let config = BenchConfig::from_cli(&cli).context("Configuration error")?;

    let runtime = tokio::runtime::Builder::new_multi_thread()
        .enable_all()
        .build()
        .expect("Failed to build tokio runtime");

    runtime
        .block_on(async {
            if config.multi_turn {
                if let Some(ref sweep_mc) = cli.sweep_max_concurrency {
                    // --- Sweep over concurrency in multi-turn mode ---
                    let values = sweep::parse_concurrency_values(sweep_mc)
                        .context("Invalid --sweep-max-concurrency")?;
                    sweep::run_multi_turn_concurrency_sweep(
                        &config,
                        &values,
                        cli.sweep_num_prompts_factor,
                    )
                    .await?;
                } else {
                    // --- Single multi-turn conversation benchmark ---
                    multi_turn::run_multi_turn_benchmark(&config).await?;
                }
            } else if let Some(ref sweep_mc) = cli.sweep_max_concurrency {
                // --- Sweep over max-concurrency ---
                let values = sweep::parse_concurrency_values(sweep_mc)
                    .context("Invalid --sweep-max-concurrency")?;
                sweep::run_concurrency_sweep(&config, &values, cli.sweep_num_prompts_factor)
                    .await?;
            } else if let Some(ref sweep_rate) = cli.sweep_request_rate {
                // --- Sweep over request-rate ---
                let values =
                    sweep::parse_rate_values(sweep_rate).context("Invalid --sweep-request-rate")?;
                sweep::run_rate_sweep(&config, &values).await?;
            } else if cli.num_runs > 1 {
                // --- Multi-run with statistical aggregation ---
                multi_run::run_multi(&config, cli.num_runs).await?;
            } else {
                // --- Normal single benchmark ---
                benchmark::run_benchmark(&config).await?;
            }
            anyhow::Ok(())
        })
        .context("Benchmark failed")
}
