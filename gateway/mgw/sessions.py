"""Sessions (conversations) and clients: where each session's prompt is cached, and how long each client's answers are.

A session is named by the client (body `prompt_cache_key`, or header `session-id` / `x-session-id`), or else by a
hash of the client's key and the start of the conversation (the first two messages), which stays the same as turns
are appended.
"""
import hashlib
import json
import math
import time
from dataclasses import dataclass, field

HALF_LIFE_S = 300   # a client's expected answer length follows its last ~5 minutes


@dataclass
class Session:
    current: str | None = None
    cache: dict = field(default_factory=dict)   # backend -> (tokens it likely holds, when they were written)
    seen: float = 0.0


class Sessions:
    def __init__(self):
        self.sessions: dict[str, Session] = {}
        self.responses: dict[str, tuple[str, str, float]] = {}   # response id -> (session, backend, when)

    def get(self, key: str) -> Session:
        s = self.sessions.get(key)
        if s is None:
            s = self.sessions[key] = Session()
        s.seen = time.time()
        return s

    def cached(self, s: Session, ttl: dict[str, float], now: float) -> dict[str, float]:
        """Per backend, the tokens of this session it likely still holds in its prompt cache."""
        return {b: tokens for b, (tokens, at) in s.cache.items() if b in ttl and now - at < ttl[b]}

    def record(self, key: str, backend: str, tokens: float, now: float, response_id: str | None = None) -> None:
        s = self.get(key)
        s.current = backend
        s.cache[backend] = (tokens, now)
        if response_id:
            self.responses[response_id] = (key, backend, now)

    def purge(self, now: float, ttl: float) -> None:
        for k in [k for k, s in self.sessions.items() if now - s.seen > ttl]:
            del self.sessions[k]
        for k in [k for k, r in self.responses.items() if now - r[2] > ttl]:
            del self.responses[k]


class Clients:
    """Per client and model: a time-weighted average of answer lengths (output tokens)."""

    def __init__(self):
        self.avg: dict[tuple, tuple[float, float]] = {}   # (client, model) -> (average, when last updated)

    def expected(self, client: str, model: str, default: float) -> float:
        known = self.avg.get((client, model))
        return known[0] if known else default

    def observe(self, client: str, model: str, output_tokens: float, now: float) -> None:
        known = self.avg.get((client, model))
        if not known:
            self.avg[(client, model)] = (output_tokens, now)
            return
        keep = math.exp(-math.log(2) * (now - known[1]) / HALF_LIFE_S)
        keep = max(keep, 0.8)  # one answer never replaces the history entirely
        self.avg[(client, model)] = (keep * known[0] + (1 - keep) * output_tokens, now)


def digest(*parts) -> str:
    return hashlib.sha256(json.dumps(parts, sort_keys=True, default=str).encode()).hexdigest()[:24]


def session_key(body: dict, headers, client: str, model: str) -> str:
    named = body.get("prompt_cache_key") or headers.get("session-id") or headers.get("x-session-id")
    if named:
        return f"{client}:{named}"
    if "messages" in body:
        start = body["messages"][:2]
    else:
        given = body.get("input")
        start = [body.get("instructions"), given[:2] if isinstance(given, list) else str(given)[:4000]]
    return f"{client}:{digest(model, start)}"
