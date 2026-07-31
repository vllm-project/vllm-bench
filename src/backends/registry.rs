// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright contributors to the vLLM project

//! Process-global registry for externally-provided custom backends.
//!
//! A `vllm-bench` process runs a single benchmark, so the selected backend is a
//! process-wide singleton. Keeping the registry global lets `get_backend()` keep
//! its `BackendKind`-only signature — every existing call site (benchmark, ready
//! check, multi-turn, sweeps) works unchanged.

use std::collections::HashMap;
use std::sync::{Arc, Mutex, OnceLock};

use super::CustomBackend;

/// Name -> backend, populated by `register_backend` before `run_cli`/`run`.
static REGISTRY: OnceLock<Mutex<HashMap<String, Arc<dyn CustomBackend>>>> = OnceLock::new();

/// The backend selected by `--backend` for the current run (set during config
/// resolution). Overwritten on each `set_active` so that a library caller which
/// invokes `run()` more than once in a process — each time selecting a different
/// custom backend — always sees the most recent selection, not the first.
static ACTIVE_CUSTOM: Mutex<Option<Arc<dyn CustomBackend>>> = Mutex::new(None);

fn registry() -> &'static Mutex<HashMap<String, Arc<dyn CustomBackend>>> {
    REGISTRY.get_or_init(|| Mutex::new(HashMap::new()))
}

/// Register a custom backend under `name`, selectable via `--backend <name>`.
///
/// Re-registering the same name overwrites the previous entry. Call this before
/// [`crate::run_cli`] / [`crate::run`].
pub fn register_backend(name: &str, backend: Arc<dyn CustomBackend>) {
    registry().lock().unwrap().insert(name.to_string(), backend);
}

/// Look up a registered backend by name.
pub fn lookup(name: &str) -> Option<Arc<dyn CustomBackend>> {
    registry().lock().unwrap().get(name).cloned()
}

/// Sorted list of registered custom backend names (for error messages / help).
pub fn registered_backend_names() -> Vec<String> {
    let mut names: Vec<String> = registry().lock().unwrap().keys().cloned().collect();
    names.sort();
    names
}

/// Mark `backend` as the active selection for this run, replacing any previous one.
pub fn set_active(backend: Arc<dyn CustomBackend>) {
    *ACTIVE_CUSTOM.lock().unwrap() = Some(backend);
}

/// The active custom backend, if `--backend` resolved to one.
pub fn active_custom() -> Option<Arc<dyn CustomBackend>> {
    ACTIVE_CUSTOM.lock().unwrap().clone()
}
