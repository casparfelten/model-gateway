"""Vercel AI Gateway (https://ai-gateway.vercel.sh/v1). Which providers serve the model, and its caching, are set in
the backend's extra_body (providerOptions.gateway)."""


def prepare(body: dict, headers: dict, ctx: dict) -> None:
    body.setdefault("prompt_cache_key", ctx["session"])
