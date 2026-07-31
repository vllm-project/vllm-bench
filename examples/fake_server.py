#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""A made-up (non-OpenAI) generation server, to showcase vllm-bench custom backends.

This deliberately does NOT speak the OpenAI protocol — it has its own request and
streaming response shape ("FakeGen") that a custom `CustomBackend` in
`examples/fake_backend.rs` knows how to talk to.

Protocol
--------
POST /v1/fakegen
    request:  {"prompt_tokens": <int>, "max_tokens": <int>}
    response: text/event-stream, one SSE message per generated token:
                  data: {"text": "<tok>", "index": <i>}

              followed by a final summary and a sentinel:
                  data: {"done": true, "generated_tokens": <N>}
                  data: [DONE]

GET /health -> {"status": "ok"}   (used by the demo script's readiness wait)

Run: python fake_server.py --port 8000
"""
import argparse
import asyncio
import json

import uvicorn
from fastapi import FastAPI, Request
from fastapi.responses import StreamingResponse

app = FastAPI()

# A pool of made-up "tokens" the server streams back.
FAKE_TOKENS = (
    "lorem ipsum dolor sit amet consectetur adipiscing elit sed do eiusmod".split()
)

# Simulated per-token generation latency (seconds) so TTFT/ITL are non-trivial.
TOKEN_DELAY_S = 0.005


@app.get("/health")
async def health():
    return {"status": "ok"}


@app.post("/v1/fakegen")
async def fakegen(req: Request):
    body = await req.json()
    max_tokens = int(body.get("max_tokens", 16))

    async def stream():
        for i in range(max_tokens):
            tok = FAKE_TOKENS[i % len(FAKE_TOKENS)]
            yield f'data: {json.dumps({"text": tok, "index": i})}\n\n'
            await asyncio.sleep(TOKEN_DELAY_S)
        yield f'data: {json.dumps({"done": True, "generated_tokens": max_tokens})}\n\n'
        yield "data: [DONE]\n\n"

    return StreamingResponse(stream(), media_type="text/event-stream")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8000)
    args = parser.parse_args()
    uvicorn.run(app, host=args.host, port=args.port, log_level="warning")
