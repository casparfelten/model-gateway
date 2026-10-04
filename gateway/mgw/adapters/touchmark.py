"""Touchmark's router (https://router.touchmark.ai/v1).

- Chat Completions only: since 2026-10-04 it answers /v1/responses with 404 endpoint_not_supported (the gateway then
  stops asking it for that path). Earlier its Responses API wrote invalid JSON (`"metadata":{}"background"`) when the
  request had no metadata; prepare() and repair() still guard against that.
- Prompt cache: on 2026-10-02 (first key) a `prompt_cache_key` gave 97% cached tokens; on 2026-10-04 (current key)
  every request reported cached_tokens 0 with created_cache_tokens ~ the prompt, with or without a key. On this block
  cached and uncached input cost the same, so it only costs time to first token.
"""


def prepare(body: dict, headers: dict, ctx: dict) -> None:
    body.setdefault("prompt_cache_key", ctx["session"])
    if ctx["path"].endswith("/responses") and not body.get("metadata"):
        body["metadata"] = {"client": "model-gateway"}


def repair(data: bytes) -> bytes:
    return data.replace(b'"metadata":{}"', b'"metadata":{},"')
