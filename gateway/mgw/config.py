"""The config file (gateway.yaml): loading, checking, and reloading it when it changes."""
import datetime as dt
import logging
import os
import re
from dataclasses import dataclass, field
from pathlib import Path

import yaml

log = logging.getLogger("mgw")
M = 1e6


def norm(name: str) -> str:
    """A model name without vendor prefix, case or punctuation: `alibaba/qwen3.8-27b` and `qwen-3.8-27b` -> qwen3827b."""
    return re.sub(r"[^a-z0-9]", "", name.lower().rsplit("/", 1)[-1])


@dataclass
class Model:
    name: str
    aliases: list[str]
    default_output_tokens: int = 1500


@dataclass
class Backend:
    name: str
    kind: str                         # openai | pool | touchmark (a touchmark entry expands into openai backends)
    url: str
    api_key_env: str = ""
    adapter: str = "openai"
    models: dict = field(default_factory=dict)        # our model name -> the backend's model id
    extra_body: dict = field(default_factory=dict)
    headers: dict = field(default_factory=dict)
    prices: dict = field(default_factory=dict)        # dollars per token: input, cached, output
    prepaid: bool = False
    token_price: float = 0.0          # dollars per weighted Token (rate-capped backends)
    weights: dict = field(default_factory=lambda: {"output": 1.0, "input": 1.0, "cached": 1.0})
    tokens_per_minute: float | None = None
    max_inflight: int | None = None
    max_context: int = 131072
    cache_ttl_s: float = 300
    ttft_s: float = 1.0
    prefill_tokens_per_s: float = 20000   # an external backend's guessed prefill speed (for the TTFT estimate)
    penalty_dollars: float = 0.0
    active_hours_utc: list | None = None
    expires: float | None = None      # unix time
    enabled: bool = True
    # pool only
    runpod_pool: str | None = None
    workers: list = field(default_factory=list)
    capacity: dict = field(default_factory=dict)
    # touchmark only: terms by Touchmark model id
    contracts: dict = field(default_factory=dict)

    @property
    def api_key(self) -> str | None:
        return os.environ.get(self.api_key_env) if self.api_key_env else None

    def active(self, now: float) -> bool:
        if not self.enabled or (self.expires and now >= self.expires):
            return False
        if self.active_hours_utc:
            start, end = self.active_hours_utc
            hour = dt.datetime.fromtimestamp(now, dt.timezone.utc).hour
            return start <= hour < end if start <= end else hour >= start or hour < end
        return True


@dataclass
class Config:
    models: dict[str, Model]
    routing: dict
    backends: dict[str, Backend]      # as written; touchmark entries are expanded at run time
    by_alias: dict[str, str]          # normalized name -> our model name

    def model_for(self, requested: str) -> str | None:
        return self.by_alias.get(norm(requested or ""))


ROUTING = {"latency_dollars_per_s": 0.0002, "congestion_starts_at": 0.5, "switch_margin_dollars": 0.0005, "switch_margin_fraction": 0.1,
           "max_wait_s": 600, "session_ttl_s": 7200}
CAPACITY = {"dollars_per_hour": 18.36, "max_running": 160, "prefill_tokens_per_s": 20000, "ttft_target_s": 3,
            "decode_tokens_per_s_min": 30}
BACKEND_FIELDS = set(Backend.__dataclass_fields__) - {"name"}


def when(value) -> float | None:
    if value is None:
        return None
    if isinstance(value, (int, float)):
        return float(value)
    if isinstance(value, dt.datetime):
        return (value if value.tzinfo else value.replace(tzinfo=dt.timezone.utc)).timestamp()
    if isinstance(value, dt.date):
        return dt.datetime(value.year, value.month, value.day, tzinfo=dt.timezone.utc).timestamp()
    return dt.datetime.fromisoformat(str(value).replace("Z", "+00:00")).timestamp()


def per_token(prices: dict | None) -> dict:
    """Prices in the file are per million tokens; in code, per token."""
    prices = prices or {}
    unknown = set(prices) - {"input", "cached", "output"}
    if unknown:
        raise ValueError(f"unknown price fields {sorted(unknown)}")
    return {k: float(prices.get(k, 0)) / M for k in ("input", "cached", "output")}


def backend(name: str, raw: dict) -> Backend:
    unknown = set(raw) - BACKEND_FIELDS
    if unknown:
        raise ValueError(f"backend {name}: unknown fields {sorted(unknown)}")
    if raw.get("kind") not in ("openai", "pool", "touchmark"):
        raise ValueError(f"backend {name}: kind must be openai, pool or touchmark")
    if not raw.get("url"):
        raise ValueError(f"backend {name}: url missing")
    b = Backend(name=name, **{k: v for k, v in raw.items() if k not in ("prices", "expires", "capacity")})
    if b.kind == "pool" and "adapter" not in raw:
        b.adapter = "vllm"
    b.url = b.url.rstrip("/")
    b.prices = per_token(raw.get("prices"))
    b.token_price = float(raw.get("token_price", 0)) / M
    b.expires = when(raw.get("expires"))
    b.capacity = {**CAPACITY, **(raw.get("capacity") or {})}
    b.weights = {"output": 1.0, "input": 1.0, "cached": 1.0, **(raw.get("weights") or {})}
    if b.kind == "touchmark":
        for model_id, terms in b.contracts.items():
            contract(b, model_id, terms)  # checked now, so a bad entry is reported at load
    return b


CONTRACT_FIELDS = {"prepaid", "token_price", "weights", "tokens_per_minute", "max_inflight", "expires", "ttft_s",
                   "prices", "max_context", "cache_ttl_s", "penalty_dollars", "enabled", "active_hours_utc"}


def contract(parent: Backend, model_id: str, terms: dict | None) -> Backend:
    """The backend for one Touchmark model: the touchmark entry's settings, overridden by the contract's terms."""
    terms = dict(terms or {})
    unknown = set(terms) - CONTRACT_FIELDS
    if unknown:
        raise ValueError(f"backend {parent.name}, contract {model_id}: unknown fields {sorted(unknown)}")
    b = Backend(name=f"{parent.name}/{model_id}", kind="openai", url=parent.url, api_key_env=parent.api_key_env,
                adapter="touchmark", extra_body=parent.extra_body, headers=parent.headers,
                max_context=parent.max_context, cache_ttl_s=parent.cache_ttl_s, ttft_s=parent.ttft_s,
                penalty_dollars=parent.penalty_dollars)
    for k, v in terms.items():
        if k == "prices":
            b.prices = per_token(v)
        elif k == "expires":
            b.expires = when(v)
        elif k == "weights":
            b.weights = {**b.weights, **v}
        elif k == "token_price":
            b.token_price = float(v) / M
        else:
            setattr(b, k, v)
    return b


def parse(text: str) -> Config:
    raw = yaml.safe_load(text) or {}
    unknown = set(raw) - {"models", "routing", "backends"}
    if unknown:
        raise ValueError(f"unknown top-level fields {sorted(unknown)}")
    models, by_alias = {}, {}
    for name, m in (raw.get("models") or {}).items():
        m = m or {}
        models[name] = Model(name=name, aliases=list(m.get("aliases") or []),
                             default_output_tokens=int(m.get("default_output_tokens", 1500)))
        for alias in [name, *models[name].aliases]:
            by_alias[norm(alias)] = name
    routing = {**ROUTING, **(raw.get("routing") or {})}
    if set(routing) - set(ROUTING):
        raise ValueError(f"routing: unknown fields {sorted(set(routing) - set(ROUTING))}")
    backends = {name: backend(name, b or {}) for name, b in (raw.get("backends") or {}).items()}
    for b in backends.values():
        for model in b.models:
            if model not in models:
                raise ValueError(f"backend {b.name}: model {model!r} is not in models")
    return Config(models=models, routing=routing, backends=backends, by_alias=by_alias)


class Watched:
    """The config file, re-read when its modification time changes. A file that fails to load leaves the last good
    config in place (`error` says why)."""

    def __init__(self, path: str | Path):
        self.path = Path(path)
        self.mtime = None
        self.error: str | None = None
        self.config: Config = self._load()

    def _load(self) -> Config:
        self.mtime = self.path.stat().st_mtime
        return parse(self.path.read_text())

    def check(self) -> bool:
        """Reload if the file changed. True when a new config is in use."""
        try:
            if self.path.stat().st_mtime == self.mtime:
                return False
            self.config, self.error = self._load(), None
            log.info("config reloaded from %s", self.path)
            return True
        except Exception as ex:  # keep the last good config
            self.error = f"{type(ex).__name__}: {ex}"
            log.error("config %s not loaded, keeping the previous one: %s", self.path, self.error)
            return False
