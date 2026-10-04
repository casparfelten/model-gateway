"""What the gateway knows about each backend while it runs: failures, requests in flight, measured speed, rate-cap use,
and for the GPU pool the nodes' load from vLLM's /metrics."""
import collections
import math
import re
import time
from dataclasses import dataclass, field

from .config import Backend

FAILS_TO_OPEN = 5        # failures in a row (no success between them) that take a backend out of use...
FAIL_WINDOW_S = 30       # ... when they span at least this long (many requests at once can fail in one moment)
OPEN_S = 60              # how long it stays out; then one request tests it


@dataclass
class Node:
    """One vLLM server of the GPU pool, as of its last /metrics."""
    url: str
    ok: bool = False
    seen: float = 0.0
    running: float = 0.0
    waiting: float = 0.0
    kv_usage: float = 0.0
    itl_s: float | None = None          # mean time between output tokens over the last scrape interval
    prefill_tps: float | None = None    # tokens prefilled per second over the last interval (not cache hits)
    counters: dict = field(default_factory=dict)


METRIC = re.compile(r"^(vllm:[a-z_]+)(?:\{[^}]*\})? ([0-9.eE+-]+|NaN)$", re.M)


def parse_metrics(text: str) -> dict[str, float]:
    """Sum of each vllm: metric over its label sets."""
    out: dict[str, float] = collections.defaultdict(float)
    for name, value in METRIC.findall(text):
        if value != "NaN":
            out[name] += float(value)
    return out


def update_node(node: Node, text: str, now: float) -> None:
    m = parse_metrics(text)
    before, dt = node.counters, now - node.seen if node.seen else 0
    node.running = m.get("vllm:num_requests_running", 0.0)
    node.waiting = m.get("vllm:num_requests_waiting", 0.0)
    node.kv_usage = m.get("vllm:kv_cache_usage_perc", m.get("vllm:gpu_cache_usage_perc", 0.0))
    itl = "vllm:inter_token_latency_seconds" if "vllm:inter_token_latency_seconds_count" in m \
        else "vllm:time_per_output_token_seconds"
    counters = {"itl_sum": m.get(f"{itl}_sum", 0.0), "itl_count": m.get(f"{itl}_count", 0.0),
                "prompt": m.get("vllm:prompt_tokens_total", 0.0),
                "hits": m.get("vllm:prefix_cache_hits_total", 0.0)}
    if before and dt > 0:
        d_count = counters["itl_count"] - before["itl_count"]
        node.itl_s = (counters["itl_sum"] - before["itl_sum"]) / d_count if d_count > 0 else None
        prefilled = (counters["prompt"] - before["prompt"]) - (counters["hits"] - before["hits"])
        node.prefill_tps = max(0.0, prefilled) / dt
    node.counters, node.seen, node.ok = counters, now, True


class State:
    def __init__(self, b: Backend):
        self.name = b.name
        self.inflight = 0
        self.pending_prefill = 0.0              # uncached prompt tokens of requests with no first token yet
        self.ttft_base = b.ttft_s               # measured time to first token, less the prefill estimate
        self.fails, self.first_fail = 0, 0.0
        self.open_until, self.testing = 0.0, False
        self.window: collections.deque = collections.deque()   # (time, Tokens) used in the last minute
        self.reserved = 0.0                     # Tokens of requests in flight (estimated)
        self.limit: float | None = b.max_inflight   # concurrency limit: configured, or learned from 429s
        self.learned_limit = False
        self.busy_until = 0.0                   # after a 429: no new requests until then
        self.nodes: dict[str, Node] = {}        # pool only
        self.prefill_tps: float | None = None   # pool only: best per-node prefill speed seen
        self.unsupported: dict[str, float] = {}  # path -> until when this backend is not asked for it
        self.served = self.failed = 0
        self.last_error = ""

    # failures
    def usable(self, now: float) -> bool:
        """Closed, or open long enough that one request may test it."""
        if now < self.open_until:
            return False
        return not (self.open_until and self.testing)

    def record_failure(self, now: float, why: str) -> None:
        self.failed += 1
        self.last_error = why
        if not self.fails:
            self.first_fail = now
        self.fails += 1
        if self.open_until or (self.fails >= FAILS_TO_OPEN and now - self.first_fail >= FAIL_WINDOW_S):
            self.open_until, self.testing = now + OPEN_S, False

    def record_success(self) -> None:
        self.served += 1
        self.fails, self.open_until, self.testing = 0, 0.0, False

    def record_busy(self, now: float, retry_after: float | None) -> None:
        """A 429: the backend is at its limit. Lower the concurrency limit to below what was in flight."""
        self.busy_until = now + (retry_after or 2.0)
        self.limit = max(1.0, min(self.limit or math.inf, self.inflight * 0.7))
        self.learned_limit = True

    def record_ok_under_load(self) -> None:
        """Successes raise a learned limit slowly back up."""
        if self.learned_limit and self.limit is not None and self.inflight >= self.limit - 1:
            self.limit += 0.2

    # rate cap
    def tokens_last_minute(self, now: float) -> float:
        while self.window and self.window[0][0] < now - 60:
            self.window.popleft()
        return sum(t for _, t in self.window)

    # speed
    def observe_ttft(self, seconds: float, fresh_tokens: float, prefill_tps: float) -> None:
        base = max(0.05, seconds - fresh_tokens / prefill_tps)
        self.ttft_base = 0.8 * self.ttft_base + 0.2 * base

    def live_nodes(self, now: float) -> list[Node]:
        return [n for n in self.nodes.values() if n.ok and now - n.seen < 30]
