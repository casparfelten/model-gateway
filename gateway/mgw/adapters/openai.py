"""Any OpenAI-compatible backend: sends the session as the prompt cache key, so the backend can keep the session on
the replica that holds its cache."""


def prepare(body: dict, headers: dict, ctx: dict) -> None:
    body.setdefault("prompt_cache_key", ctx["session"])
