"""The gateway's HTTP server: OpenAI-compatible /v1/chat/completions, /v1/responses and /v1/completions in front of
the backends in gateway.yaml.

A request goes to the best-scoring backend (score.py). If that backend fails before sending anything (no connection,
5xx, 429, no first token in time), the request goes to the next one; when every backend has failed it waits and
tries again, up to routing.max_wait_s. Once an answer has started streaming it cannot move.
"""
import asyncio
import contextlib
import copy
import hashlib
import json
import logging
import os
import time
from pathlib import Path

import httpx
from starlette.applications import Starlette
from starlette.requests import Request
from starlette.responses import JSONResponse, Response, StreamingResponse
from starlette.routing import Route

from . import pool
from .adapters import Adapters
from .config import Backend, Watched, contract, norm
from .score import Ask, Option, rank
from .sessions import Clients, Sessions, session_key
from .state import State

log = logging.getLogger("mgw")
PATHS = {"/v1/chat/completions": "/chat/completions", "/v1/responses": "/responses", "/v1/completions": "/completions"}
CONTEXT_ERRORS = ("context length", "context_length", "maximum context", "too long", "max_model_len", "too many tokens")


def merged(*bodies: dict) -> dict:
    """Bodies merged, nested dicts key by key (providerOptions.gateway...)."""
    out: dict = {}
    for body in bodies:
        for k, v in body.items():
            out[k] = merged(out[k], v) if isinstance(v, dict) and isinstance(out.get(k), dict) else copy.deepcopy(v)
    return out


def error(status: int, message: str, kind: str = "gateway_error") -> JSONResponse:
    return JSONResponse({"error": {"message": message, "type": kind}}, status_code=status)


def usage_of(d: dict) -> dict | None:
    """Prompt, cached and output tokens from a chat or responses answer (or stream event)."""
    u = d.get("usage") or (d.get("response") or {}).get("usage")
    if not u:
        return None
    details = u.get("prompt_tokens_details") or u.get("input_tokens_details") or {}
    return {"prompt": u.get("prompt_tokens", u.get("input_tokens")) or 0, "cached": details.get("cached_tokens") or 0,
            "output": u.get("completion_tokens", u.get("output_tokens")) or 0}


class Gateway:
    def __init__(self, config_path: str, adapters_dir: str | None = None):
        self.watched = Watched(config_path)
        self.adapters = Adapters(adapters_dir or Path(config_path).parent / "adapters")
        self.http = httpx.AsyncClient(timeout=httpx.Timeout(connect=10, read=None, write=120, pool=None),
                                      limits=httpx.Limits(max_connections=4000, max_keepalive_connections=500))
        self.states: dict[str, State] = {}
        self.backends: dict[str, Backend] = {}
        self.listed: dict[str, list[str]] = {}       # touchmark entry -> the model ids its key lists
        self.sessions, self.clients = Sessions(), Clients()
        self.chars_per_token = 3.0
        self.expand()

    @property
    def config(self):
        return self.watched.config

    # backends
    def expand(self) -> None:
        """The backends in use: the config's, with each touchmark entry turned into one backend per model it lists."""
        backends = {}
        for b in self.config.backends.values():
            if b.kind != "touchmark":
                backends[b.name] = b
                continue
            terms = {norm(k): v for k, v in b.contracts.items()}
            for model_id in self.listed.get(b.name, []):
                ours = self.config.model_for(model_id)
                if ours is None:
                    continue
                c = contract(b, model_id, terms.get(norm(model_id), {}))
                c.models = {ours: model_id}
                backends[c.name] = c
        for name, b in backends.items():
            if name not in self.states:
                self.states[name] = State(b)
            elif b.max_inflight and not self.states[name].learned_limit:
                self.states[name].limit = b.max_inflight
        self.backends = backends

    async def list_touchmark(self) -> None:
        changed = False
        for b in self.config.backends.values():
            if b.kind != "touchmark" or not b.api_key:
                continue
            try:
                r = await self.http.get(f"{b.url}/models", headers={"Authorization": f"Bearer {b.api_key}"},
                                        timeout=20)
                r.raise_for_status()
                ids = sorted(m["id"] for m in r.json().get("data", []))
            except (httpx.HTTPError, ValueError, KeyError) as ex:
                log.warning("touchmark %s: model list failed (%s); keeping %s", b.name, ex, self.listed.get(b.name))
                continue
            if ids != self.listed.get(b.name):
                log.info("touchmark %s lists %s", b.name, ids)
                self.listed[b.name], changed = ids, True
        if changed:
            self.expand()

    async def loop(self) -> None:
        """Background work: config reloads, Touchmark model lists, the GPU pool's nodes and load."""
        last = {"touchmark": 0.0, "discover": 0.0, "purge": 0.0}
        while True:
            now = time.time()
            try:
                if self.watched.check():
                    self.expand()
                    last["touchmark"] = last["discover"] = 0.0
                if now - last["touchmark"] > 60:
                    last["touchmark"] = now
                    await self.list_touchmark()
                for b in self.backends.values():
                    if b.kind != "pool":
                        continue
                    try:
                        if now - last["discover"] > 15:
                            await pool.discover(self.http, b, self.states[b.name])
                        else:
                            await pool.scrape(self.http, b, self.states[b.name])
                    except httpx.HTTPError as ex:
                        log.warning("pool %s: %s: %s", b.name, type(ex).__name__, ex)
                if now - last["discover"] > 15:
                    last["discover"] = now
                if now - last["purge"] > 60:
                    last["purge"] = now
                    self.sessions.purge(now, self.config.routing["session_ttl_s"])
            except Exception:
                log.exception("background loop")
            await asyncio.sleep(2)

    # requests
    def client_of(self, request: Request) -> str:
        auth = request.headers.get("authorization", "")
        if auth:
            return hashlib.sha256(auth.encode()).hexdigest()[:12]
        return request.client.host if request.client else "unknown"

    def authorized(self, request: Request) -> bool:
        keys = [k.strip() for k in os.environ.get("GATEWAY_API_KEY", "").split(",") if k.strip()]
        if not keys:
            return True
        auth = request.headers.get("authorization", "")
        return auth.removeprefix("Bearer ").strip() in keys

    async def handle(self, request: Request) -> Response:
        if not self.authorized(request):
            return error(401, "invalid api key", "authentication_error")
        path = PATHS[request.url.path]
        try:
            body = await request.json()
        except ValueError:
            return error(400, "body is not JSON", "invalid_request_error")
        model = self.config.model_for(body.get("model", ""))
        if model is None:
            return error(404, f"unknown model {body.get('model')!r}; known: {sorted(self.config.models)}",
                         "invalid_request_error")
        now = time.time()
        client = self.client_of(request)
        key = session_key(body, request.headers, client, model)
        pinned = None
        if body.get("previous_response_id"):
            known = self.sessions.responses.get(body["previous_response_id"])
            if known:
                key, pinned = known[0], known[1]
        chars = len(json.dumps({k: body.get(k) for k in ("messages", "input", "instructions", "tools", "prompt")}))
        prompt = chars / self.chars_per_token
        limit = body.get("max_tokens") or body.get("max_completion_tokens") or body.get("max_output_tokens")
        default = self.config.models[model].default_output_tokens
        output = min(limit or 1e12, self.clients.expected(client, model, default))
        session = self.sessions.get(key)
        ttl = {name: b.cache_ttl_s for name, b in self.backends.items()}
        ask = Ask(model, prompt, output, self.sessions.cached(session, ttl, now), session.current, pinned)
        ctx = {"session": key.split(":", 1)[1], "client": client, "chars": chars, "path": path, "model": model}

        deadline = now + self.config.routing["max_wait_s"]
        tried: set[str] = set()
        wait = 1.0
        while True:
            t = time.time()
            # backends that answered "no such endpoint" for this path lately are not tried for it
            lacking = {n for n, st in self.states.items() if st.unsupported.get(path, 0) > t}
            options = rank(self.backends, self.states, ask, t, self.config.routing, exclude=tried | lacking)
            if not options:
                # every backend that could take it now lacks this endpoint: answer at once, don't wait
                able = [o.backend for o in rank(self.backends, self.states, ask, t, self.config.routing)]
                if lacking and able and all(n in lacking for n in able):
                    return ctx.get("refusal") or error(404, f"no backend serves {request.url.path} for {model}",
                                                       "invalid_request_error")
                if await request.is_disconnected():
                    return error(499, "client went away")
                if time.time() > deadline:
                    return error(503, f"no backend took the request in {self.config.routing['max_wait_s']} s")
                if pinned and pinned not in self.backends:
                    return error(404, f"previous_response_id is on backend {pinned}, which is gone",
                                 "invalid_request_error")
                await asyncio.sleep(wait)
                wait = min(wait * 2, 10.0)
                tried.clear()   # everything was tried: start over
                continue
            choice = options[0]
            if tried and await request.is_disconnected():
                return error(499, "client went away")
            answer = await self.attempt(self.backends[choice.backend], choice, body, ask, ctx)
            if answer is not None:
                return answer
            tried.add(choice.backend)

    async def attempt(self, b: Backend, choice: Option, body: dict, ask: Ask, ctx: dict) -> Response | None:
        """Send to one backend. None: it failed before answering (try another)."""
        st = self.states[b.name]
        now = time.time()
        if st.open_until and now >= st.open_until:
            st.testing = True   # this request tests a backend that was taken out of use
        stream = bool(body.get("stream"))
        up = merged(body, b.extra_body)
        up["model"] = b.models[ask.model]
        # identity: the answer is relayed byte for byte, so it must not arrive compressed
        headers = {"content-type": "application/json", "accept-encoding": "identity", **b.headers}
        if b.kind == "pool":
            headers["x-smg-routing-key"] = ctx["session"]   # SMG's own auth to the nodes is the worker's key
        elif b.api_key:
            headers["authorization"] = f"Bearer {b.api_key}"
        strip_usage = False
        if stream and ctx["path"] in ("/chat/completions", "/completions"):
            opts = up.setdefault("stream_options", {})
            if not opts.get("include_usage"):
                opts["include_usage"], strip_usage = True, True   # to learn the usage; not passed on
        adapter = self.adapters.get(b.adapter)
        if hasattr(adapter, "prepare"):
            adapter.prepare(up, headers, {"backend": b, "session": ctx["session"], "path": ctx["path"],
                                          "stream": stream})
        repair = getattr(adapter, "repair", None)

        fresh = choice.parts["fresh"]
        w = b.weights
        reserve = (ask.output_tokens * w["output"] + fresh * w["input"] + choice.parts["cached"] * w["cached"]
                   if b.tokens_per_minute else 0.0)
        st.inflight += 1
        st.pending_prefill += fresh
        st.reserved += reserve
        held = {"prefill": fresh}

        def release():
            st.inflight -= 1
            st.pending_prefill -= held["prefill"]
            st.reserved -= reserve
            held["prefill"] = 0

        first_timeout = 60 + 3 * fresh / max(1.0, b.prefill_tokens_per_s if b.kind != "pool"
                                             else b.capacity["prefill_tokens_per_s"])
        started = time.time()
        resp = None
        try:
            req = self.http.build_request("POST", b.url + ctx["path"], json=up, headers=headers)
            resp = await asyncio.wait_for(self.http.send(req, stream=True),
                                          timeout=first_timeout if stream else 1800)
            if resp.status_code != 200:
                text = (await resp.aread())[:2000]
                await resp.aclose()
                release()
                return self.failed(b, st, resp.status_code, text, resp.headers, ctx)
            out_headers = {"content-type": resp.headers.get("content-type", "application/json"),
                           "x-gateway-backend": b.name}
            if resp.headers.get("x-request-id"):
                out_headers["x-request-id"] = resp.headers["x-request-id"]
            if not stream:
                data = await asyncio.wait_for(resp.aread(), timeout=1800)
                await resp.aclose()
                if repair:
                    data = repair(data)
                held_prefill = held["prefill"]
                release()
                self.finished(b, st, ask, ctx, choice, data_events=[data], seconds=time.time() - started,
                              first=time.time() - started, fresh=held_prefill)
                return Response(data, status_code=200, headers=out_headers)
            chunks = resp.aiter_raw()
            first = await asyncio.wait_for(chunks.__anext__(), timeout=first_timeout)
        except (httpx.HTTPError, asyncio.TimeoutError, StopAsyncIteration) as ex:
            if resp is not None:
                await resp.aclose()
            release()
            st.record_failure(time.time(), f"{type(ex).__name__}: {ex}"[:300])
            log.warning("backend %s failed before answering: %r", b.name, ex)
            return None
        except BaseException:
            if resp is not None:
                await resp.aclose()
            release()
            raise
        ttft = time.time() - started
        st.observe_ttft(ttft, held["prefill"], b.prefill_tokens_per_s if b.kind != "pool"
                        else b.capacity["prefill_tokens_per_s"])
        fresh_tokens = held["prefill"]
        st.pending_prefill -= held["prefill"]
        held["prefill"] = 0
        return StreamingResponse(self.relay(b, st, resp, first, chunks, repair, strip_usage, release, ask, ctx,
                                            choice, started, ttft, fresh_tokens),
                                 status_code=200, headers=out_headers)

    async def relay(self, b, st, resp, first, chunks, repair, strip_usage, release, ask, ctx, choice, started, ttft,
                    fresh):
        """Pass a stream through line by line, reading its usage (and its response id) on the way."""
        events, buffer, ok = [], b"", False
        try:
            async for chunk in self._chain(first, chunks):
                buffer += chunk
                *lines, buffer = buffer.split(b"\n")
                out = []
                for line in lines:
                    if repair:
                        line = repair(line)
                    if line.startswith(b"data:") and (b'"usage"' in line or b'"response.created"' in line):
                        events.append(line[5:])
                        if strip_usage and b'"choices":[]' in line.replace(b" ", b""):
                            continue
                    out.append(line + b"\n")
                if out:
                    yield b"".join(out)
            if buffer:
                yield repair(buffer) if repair else buffer
            ok = True
        except httpx.HTTPError as ex:
            st.record_failure(time.time(), f"stream broke: {ex}"[:300])
            log.warning("backend %s: stream broke after %.1f s: %r", b.name, time.time() - started, ex)
        finally:
            await resp.aclose()
            release()
            if ok:
                self.finished(b, st, ask, ctx, choice, data_events=events, seconds=time.time() - started,
                              first=ttft, fresh=fresh)

    @staticmethod
    async def _chain(first, rest):
        yield first
        async for chunk in rest:
            yield chunk

    def failed(self, b: Backend, st: State, status: int, text: bytes, headers, ctx: dict) -> Response | None:
        now = time.time()
        message = text.decode(errors="replace")
        if status in (404, 405):
            # the backend does not offer this endpoint (e.g. Touchmark dropped /v1/responses): not a failure of the
            # backend; skip it for this path for 10 minutes, and keep its answer for the client if no one else has it
            st.unsupported[ctx["path"]] = now + 600
            ctx["refusal"] = Response(text, status_code=status, headers={"content-type": "application/json",
                                                                         "x-gateway-backend": b.name})
            log.warning("backend %s does not serve %s: HTTP %s %s", b.name, ctx["path"], status, message[:200])
            return None
        if status == 429:
            retry = headers.get("retry-after")
            st.record_busy(now, float(retry) if retry and retry.replace(".", "", 1).isdigit() else None)
            log.warning("backend %s: 429 (limit now %.1f in flight)", b.name, st.limit or -1)
            return None
        if status in (400, 413, 422):
            if any(e in message.lower() for e in CONTEXT_ERRORS):
                log.warning("backend %s: request too long for it: %s", b.name, message[:200])
                return None
            return Response(text, status_code=status, headers={"content-type": "application/json",
                                                                "x-gateway-backend": b.name})
        st.record_failure(now, f"HTTP {status}: {message[:200]}")
        log.warning("backend %s: HTTP %s %s", b.name, status, message[:300])
        return None

    def finished(self, b, st, ask, ctx, choice, data_events, seconds, first, fresh) -> None:
        now = time.time()
        usage, response_id = None, None
        for raw in data_events:
            try:
                d = json.loads(raw)
            except ValueError:
                continue
            usage = usage_of(d) or usage
            rid = (d.get("response") or {}).get("id") if "response" in d else d.get("id")
            if ctx["path"] == "/responses" and rid and not response_id:
                response_id = rid
        st.record_success()
        st.record_ok_under_load()
        if usage:
            if usage["prompt"]:
                self.chars_per_token = 0.95 * self.chars_per_token + 0.05 * (ctx["chars"] / usage["prompt"])
            self.clients.observe(ctx["client"], ask.model, usage["output"], now)
            if b.tokens_per_minute:
                w = b.weights
                st.window.append((now, usage["output"] * w["output"] + usage["cached"] * w["cached"]
                                  + (usage["prompt"] - usage["cached"]) * w["input"]))
        tokens = (usage["prompt"] + usage["output"]) if usage else ask.prompt_tokens + ask.output_tokens
        key = f"{ctx['client']}:{ctx['session']}"
        self.sessions.record(key, b.name, tokens, now, response_id)
        log.info(json.dumps({"backend": b.name, "model": ask.model, "session": ctx["session"][:16],
                             "usage": usage, "ttft": round(first, 2), "seconds": round(seconds, 2),
                             "score": round(choice.score, 6), "parts": choice.parts}))

    # status
    def status(self) -> dict:
        now = time.time()
        from .score import fill
        out = {}
        for name, b in self.backends.items():
            st = self.states[name]
            u, full = fill(b, st, now) if b.kind != "pool" or st.live_nodes(now) else (1.0, True)
            out[name] = {
                "active": b.active(now), "usable": st.usable(now), "fill": round(u, 3), "full": full,
                "inflight": st.inflight, "limit": st.limit, "served": st.served, "failed": st.failed,
                "last_error": st.last_error, "ttft_base_s": round(st.ttft_base, 2),
                "tokens_last_minute": round(st.tokens_last_minute(now)) if b.tokens_per_minute else None,
                "nodes": {u_: {"ok": n.ok, "running": n.running, "waiting": n.waiting, "kv": round(n.kv_usage, 3),
                               "itl_s": n.itl_s, "prefill_tps": n.prefill_tps} for u_, n in st.nodes.items()} or None,
            }
        return {"backends": out, "config_error": self.watched.error, "sessions": len(self.sessions.sessions),
                "touchmark_models": self.listed, "chars_per_token": round(self.chars_per_token, 2)}


def build(config_path: str) -> Starlette:
    gw = Gateway(config_path)

    async def models(request: Request) -> Response:
        return JSONResponse({"object": "list", "data": [{"id": m, "object": "model", "owned_by": "model-gateway"}
                                                        for m in gw.config.models]})

    async def health(request: Request) -> Response:
        return Response("ok")

    async def status(request: Request) -> Response:
        if not gw.authorized(request):
            return error(401, "invalid api key", "authentication_error")
        return JSONResponse(gw.status())

    @contextlib.asynccontextmanager
    async def lifespan(app):
        task = asyncio.create_task(gw.loop())
        yield
        task.cancel()

    routes = [Route(p, gw.handle, methods=["POST"]) for p in PATHS]
    routes += [Route("/v1/models", models), Route("/health", health), Route("/gateway/status", status)]
    app = Starlette(routes=routes, lifespan=lifespan)
    app.state.gateway = gw
    return app
