"""Touchmark's router (https://router.touchmark.ai/v1).

- Its prompt cache only works with a `prompt_cache_key` (tested: 0 cached tokens without one, 97% with one).
- Its Responses API writes `"metadata":{}"background"...` (no comma: invalid JSON) when the request has no metadata.
  A non-empty metadata avoids it; repair() fixes the bytes anyway.
"""


def prepare(body: dict, headers: dict, ctx: dict) -> None:
    body.setdefault("prompt_cache_key", ctx["session"])
    if ctx["path"].endswith("/responses") and not body.get("metadata"):
        body["metadata"] = {"client": "model-gateway"}


def repair(data: bytes) -> bytes:
    return data.replace(b'"metadata":{}"', b'"metadata":{},"')
