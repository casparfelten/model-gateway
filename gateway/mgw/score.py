"""Where a request should go: each backend's estimated cost for it, in dollars (see gateway.yaml for the idea)."""
from dataclasses import dataclass

from .config import Backend
from .state import State


@dataclass
class Ask:
    """What the scoring needs to know about a request."""
    model: str
    prompt_tokens: float            # estimated
    output_tokens: float            # expected
    cached: dict                    # backend -> tokens of this prompt it likely still holds in its cache
    current: str | None = None      # the backend that served this session last
    pinned: str | None = None       # a backend the request must go to (it continues a response stored there)


@dataclass
class Option:
    backend: str
    score: float
    full: bool
    parts: dict


def fill(b: Backend, st: State, now: float) -> tuple[float, bool]:
    """How full the backend is, 0 to 1, and whether it should take new work only as a last resort."""
    if now < st.busy_until:
        return 1.0, True
    u = 0.0
    if b.tokens_per_minute:
        u = max(u, (st.tokens_last_minute(now) + st.reserved) / b.tokens_per_minute)
    if st.limit:
        u = max(u, st.inflight / st.limit)
    if b.kind == "pool":
        nodes = st.live_nodes(now)
        cap = b.capacity
        itl_target = 1 / cap["decode_tokens_per_s_min"]
        per_node = []
        for n in nodes:
            parts = [n.running / cap["max_running"]]
            if n.itl_s:
                parts.append(n.itl_s / itl_target)
            per_node.append(1.0 if n.waiting > 0 else min(1.0, max(parts)))
        prefill = prefill_speed(b, st) * len(nodes)
        queued = st.pending_prefill / (prefill * cap["ttft_target_s"])
        u = max(u, sum(per_node) / len(per_node), queued)
        return min(u, 1.0), u >= 1.0 or all(n.waiting > 0 for n in nodes)
    return min(u, 1.0), u >= 1.0


def prefill_speed(b: Backend, st: State) -> float:
    if b.kind == "pool":
        return max(b.capacity["prefill_tokens_per_s"], st.prefill_tps or 0)
    return b.prefill_tokens_per_s


def curve(u: float, knee: float) -> float:
    """What capacity is worth as the backend fills, as a multiple of its price: nothing below `knee` (batched
    inference slows nobody down while the batch has room), then rising steeply: 1 halfway from knee to full, 99 at
    full."""
    if u <= knee:
        return 0.0
    return min(99.0, (u - knee) / (1 - u)) if u < 1 else 99.0


def option(b: Backend, st: State, ask: Ask, now: float, routing: dict) -> Option:
    cached = min(ask.cached.get(b.name, 0.0), ask.prompt_tokens)
    fresh = ask.prompt_tokens - cached
    out = ask.output_tokens
    price = 0.0 if b.prepaid else fresh * b.prices["input"] + cached * b.prices["cached"] + out * b.prices["output"]
    u, full = fill(b, st, now)
    knee = routing["congestion_starts_at"]
    load = 0.0
    if b.kind == "pool":
        cap = b.capacity
        node_seconds = fresh / prefill_speed(b, st) + out / (cap["decode_tokens_per_s_min"] * cap["max_running"])
        load = cap["dollars_per_hour"] / 3600 * node_seconds * curve(u, knee)
    elif b.tokens_per_minute:
        w = b.weights
        tokens = out * w["output"] + fresh * w["input"] + cached * w["cached"]
        load = tokens * b.token_price * curve(u, knee)
    if b.kind == "pool":
        nodes = max(1, len(st.live_nodes(now)))
        ttft = 0.3 + (st.pending_prefill / nodes + fresh) / prefill_speed(b, st)
    else:
        ttft = st.ttft_base + fresh / prefill_speed(b, st)
    latency = ttft * routing["latency_dollars_per_s"]
    total = price + load + latency + b.penalty_dollars
    return Option(b.name, total, full, {"price": price, "load": load, "latency": latency, "fill": round(u, 3),
                                        "cached": round(cached), "fresh": round(fresh), "ttft": round(ttft, 2)})


def rank(backends: dict[str, Backend], states: dict[str, State], ask: Ask, now: float, routing: dict,
         exclude: set = frozenset()) -> list[Option]:
    """The backends that can take the request, best first: not full before full, then by score, except that the
    session's current backend is kept unless another beats it by the switch margin."""
    options = []
    for b in backends.values():
        st = states.get(b.name)
        if st is None or b.name in exclude or ask.model not in b.models or not b.active(now) or not st.usable(now):
            continue
        if ask.pinned and b.name != ask.pinned:
            continue
        if ask.prompt_tokens > b.max_context:
            continue
        if b.kind == "pool" and not st.live_nodes(now):
            continue
        options.append(option(b, st, ask, now, routing))
    options.sort(key=lambda o: (o.full, o.score))
    if options and ask.current and options[0].backend != ask.current:
        best = options[0]
        here = next((o for o in options if o.backend == ask.current and o.full == best.full), None)
        margin = max(routing["switch_margin_dollars"], routing["switch_margin_fraction"] * best.score)
        if here and here.score - best.score <= margin:
            options.remove(here)
            options.insert(0, here)
    return options
