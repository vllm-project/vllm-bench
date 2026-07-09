#!/usr/bin/env bash
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
#
# End-to-end demo: benchmark a non-OpenAI ("FakeGen") server through a custom
# vllm-bench backend, proving the binary supports externally-provided backends.
#
#   1. spins up examples/fake_server.py (FastAPI, made-up /v1/fakegen protocol)
#   2. builds examples/fake_backend.rs (registers a "fakegen" CustomBackend)
#   3. runs the benchmark with `--backend fakegen` against the fake server
#
# Usage:  examples/run_demo.sh            # debug build (fast to compile)
#         PROFILE=release examples/run_demo.sh   # release build (realistic perf)
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(cd "$HERE/.." && pwd)"
PORT="${PORT:-8000}"
PROFILE="${PROFILE:-debug}"
VENV="$HERE/.demo-venv"

# --- Toolchain -------------------------------------------------------------
if ! command -v cargo >/dev/null 2>&1; then
  [ -f "$HOME/.cargo/env" ] && . "$HOME/.cargo/env"
fi
command -v cargo >/dev/null 2>&1 || { echo "error: cargo not found on PATH"; exit 1; }
command -v python3 >/dev/null 2>&1 || { echo "error: python3 not found on PATH"; exit 1; }

# --- Python env (FastAPI + uvicorn) ---------------------------------------
if [ ! -x "$VENV/bin/python" ]; then
  echo "[demo] creating venv + installing fastapi/uvicorn into $VENV ..."
  python3 -m venv "$VENV"
  "$VENV/bin/pip" install -q --upgrade pip
  "$VENV/bin/pip" install -q fastapi uvicorn
fi

# --- Start the fake server -------------------------------------------------
echo "[demo] starting FakeGen server on 127.0.0.1:$PORT ..."
"$VENV/bin/python" "$HERE/fake_server.py" --port "$PORT" &
SERVER_PID=$!
cleanup() { kill "$SERVER_PID" 2>/dev/null || true; }
trap cleanup EXIT

echo "[demo] waiting for server readiness ..."
for _ in $(seq 1 40); do
  if curl -sf "http://127.0.0.1:$PORT/health" >/dev/null 2>&1; then
    READY=1; break
  fi
  sleep 0.25
done
[ "${READY:-}" = "1" ] || { echo "error: server did not become ready"; exit 1; }

# --- Build the custom-backend example -------------------------------------
echo "[demo] building fake_backend example ($PROFILE) ..."
if [ "$PROFILE" = "release" ]; then
  ( cd "$ROOT" && cargo build --release --example fake_backend )
  BIN="$ROOT/target/release/examples/fake_backend"
else
  ( cd "$ROOT" && cargo build --example fake_backend )
  BIN="$ROOT/target/debug/examples/fake_backend"
fi

# --- Run the benchmark through the custom backend --------------------------
# gpt2 is a built-in tiktoken encoding (no download); --prompt-token-ids skips
# the server-side /tokenize verification pass the fake server doesn't implement.
echo "[demo] benchmarking via --backend fakegen ..."
echo
"$BIN" \
  --backend fakegen \
  --base-url "http://127.0.0.1:$PORT" \
  --model fake-model \
  --tokenizer gpt2 --prompt-token-ids \
  --dataset-name random \
  --random-input-len 32 --random-output-len 16 \
  --num-prompts 20 \
  --max-concurrency 4

echo
echo "[demo] done. (server pid $SERVER_PID will be stopped on exit)"
