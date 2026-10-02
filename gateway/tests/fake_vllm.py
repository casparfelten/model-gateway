"""A stand-in for a vLLM node, for testing the gateway and SMG without a GPU: /health, /v1/models, /metrics,
/v1/chat/completions (streamed or not) and /v1/responses (not streamed). Needs Bearer $VLLM_API_KEY.

    python tests/fake_vllm.py [port]
"""
import asyncio
import json
import os
import sys
import time

import uvicorn
from starlette.applications import Starlette
from starlette.responses import JSONResponse, PlainTextResponse, StreamingResponse
from starlette.routing import Route

MODEL = "Qwen/Qwen3.8-27B"
KEY = os.environ.get("VLLM_API_KEY", "test")
counts = {"requests": 0, "running": 0, "prompt": 0}


def authorized(request):
    return request.headers.get("authorization") == f"Bearer {KEY}"


async def health(request):
    return PlainTextResponse("")


async def models(request):
    if not authorized(request):
        return JSONResponse({"error": "Unauthorized"}, 401)
    return JSONResponse({"object": "list", "data": [{"id": MODEL, "object": "model", "max_model_len": 600000}]})


async def metrics(request):
    lines = [f'vllm:num_requests_running{{model_name="{MODEL}"}} {counts["running"]}',
             f'vllm:num_requests_waiting{{model_name="{MODEL}"}} 0',
             f'vllm:kv_cache_usage_perc{{model_name="{MODEL}"}} 0.1',
             f'vllm:prompt_tokens_total{{model_name="{MODEL}"}} {counts["prompt"]}',
             f'vllm:prefix_cache_hits_total{{model_name="{MODEL}"}} 0',
             f'vllm:inter_token_latency_seconds_sum{{model_name="{MODEL}"}} {counts["requests"] * 0.01}',
             f'vllm:inter_token_latency_seconds_count{{model_name="{MODEL}"}} {counts["requests"]}']
    return PlainTextResponse("\n".join(lines) + "\n")


async def chat(request):
    if not authorized(request):
        return JSONResponse({"error": "Unauthorized"}, 401)
    body = await request.json()
    counts["requests"] += 1
    prompt = len(json.dumps(body.get("messages"))) // 3
    counts["prompt"] += prompt
    words = ["fake", "node", "says", "hi"]
    usage = {"prompt_tokens": prompt, "completion_tokens": len(words), "total_tokens": prompt + len(words),
             "prompt_tokens_details": {"cached_tokens": 0}}
    base = {"id": f"chatcmpl-{time.time()}", "model": body.get("model"), "created": int(time.time())}
    if not body.get("stream"):
        return JSONResponse({**base, "object": "chat.completion", "usage": usage, "choices": [
            {"index": 0, "finish_reason": "stop", "message": {"role": "assistant", "content": " ".join(words)}}]})

    async def events():
        counts["running"] += 1
        try:
            for w in words:
                await asyncio.sleep(0.05)
                yield "data: " + json.dumps({**base, "object": "chat.completion.chunk", "choices": [
                    {"index": 0, "delta": {"content": w + " "}, "finish_reason": None}]}) + "\n\n"
            yield "data: " + json.dumps({**base, "object": "chat.completion.chunk", "choices": [
                {"index": 0, "delta": {}, "finish_reason": "stop"}]}) + "\n\n"
            if (body.get("stream_options") or {}).get("include_usage"):
                yield "data: " + json.dumps({**base, "object": "chat.completion.chunk", "choices": [],
                                             "usage": usage}) + "\n\n"
            yield "data: [DONE]\n\n"
        finally:
            counts["running"] -= 1
    return StreamingResponse(events(), media_type="text/event-stream")


async def responses(request):
    if not authorized(request):
        return JSONResponse({"error": "Unauthorized"}, 401)
    body = await request.json()
    return JSONResponse({"id": f"resp_{time.time()}", "object": "response", "status": "completed",
                         "model": body.get("model"), "output": [{"type": "message", "role": "assistant", "content": [
                             {"type": "output_text", "text": "fake node says hi"}]}],
                         "usage": {"input_tokens": 10, "input_tokens_details": {"cached_tokens": 0},
                                   "output_tokens": 4, "total_tokens": 14}})


app = Starlette(routes=[Route("/health", health), Route("/v1/models", models), Route("/metrics", metrics),
                        Route("/v1/chat/completions", chat, methods=["POST"]),
                        Route("/v1/responses", responses, methods=["POST"])])

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=int(sys.argv[1]) if len(sys.argv) > 1 else 8000, log_level="warning")
