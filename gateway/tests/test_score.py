"""Where requests go as the GPU pool and the Touchmark rate cap fill, and when sessions move back."""
import time
from pathlib import Path

import pytest

from mgw.config import contract, parse
from mgw.score import Ask, rank
from mgw.state import Node, State

CONFIG = parse((Path(__file__).resolve().parent.parent / "config" / "gateway.yaml").read_text())
ROUTING = CONFIG.routing
NOW = time.time()


def backends():
    gpu, vercel, tm = CONFIG.backends["gpu"], CONFIG.backends["vercel"], CONFIG.backends["touchmark"]
    block = contract(tm, "qwen-3.8-27b", tm.contracts["qwen-3.8-27b"])
    block.models = {"qwen3.8-27b": "qwen-3.8-27b"}
    return {"gpu": gpu, block.name: block, "vercel": vercel}


def states(bs, running=20, waiting=0, itl=0.012, tokens_used=0.0):
    out = {name: State(b) for name, b in bs.items()}
    out["gpu"].nodes["http://n1"] = Node("http://n1", ok=True, seen=NOW, running=running, waiting=waiting, itl_s=itl)
    if tokens_used:
        out["touchmark/qwen-3.8-27b"].window.append((NOW, tokens_used))
    return out


def ask(prompt=20_000, output=1500, cached=None, current=None):
    return Ask("qwen3.8-27b", prompt, output, cached or {}, current)


def first(bs, sts, a):
    return rank(bs, sts, a, NOW, ROUTING)[0].backend


def test_idle_gpu_takes_new_sessions():
    bs = backends()
    assert first(bs, states(bs, running=20), ask()) == "gpu"


def test_full_gpu_spills_to_the_touchmark_block_then_vercel_when_the_rate_cap_is_near():
    bs = backends()
    assert first(bs, states(bs, waiting=5), ask()) == "touchmark/qwen-3.8-27b"
    assert first(bs, states(bs, waiting=5, tokens_used=1_990_000), ask()) == "vercel"


def test_busy_gpu_spills_uncached_sessions_first():
    """GPU 85% full. A new session (cached nowhere) costs the GPU far more than one whose 100k-token prompt it holds,
    so new sessions are the first to go. With Vercel (paid) the only other choice, the cached session stays."""
    bs = backends()
    sts = states(bs, running=136)
    new, cached = ask(prompt=100_000), ask(prompt=100_000, cached={"gpu": 99_000}, current="gpu")
    gpu_score = lambda a: next(o for o in rank(bs, sts, a, NOW, ROUTING) if o.backend == "gpu").score
    assert gpu_score(new) > 10 * gpu_score(cached)
    assert first(bs, sts, new) == "touchmark/qwen-3.8-27b"   # the block is free while its rate cap has room
    del bs["touchmark/qwen-3.8-27b"]
    assert first(bs, sts, cached) == "gpu"


def test_sessions_on_vercel_come_back_when_the_gpu_empties():
    """The evaluation case: sessions spilled to Vercel while the GPU was full. Once it is mostly idle, their next turn
    goes back to the GPU even though that re-prefills their prompt there; while it is busy they stay."""
    bs = backends()
    bs.pop("touchmark/qwen-3.8-27b")     # only GPU and Vercel
    on_vercel = ask(prompt=40_000, cached={"vercel": 38_000}, current="vercel")
    assert first(bs, states(bs, running=30), on_vercel) == "gpu"
    assert first(bs, states(bs, running=150), on_vercel) == "vercel"


def test_slow_decoding_counts_as_full():
    bs = backends()
    bs.pop("touchmark/qwen-3.8-27b")
    sts = states(bs, running=40, itl=1 / 20)   # 20 tokens/s per stream, below the 30 floor
    assert first(bs, sts, ask()) == "vercel"


def test_a_session_does_not_move_for_a_small_difference():
    bs = backends()
    bs.pop("vercel")
    sts = states(bs, running=60)
    a = ask(prompt=2_000, cached={"touchmark/qwen-3.8-27b": 1_900}, current="touchmark/qwen-3.8-27b")
    assert first(bs, sts, a) == "touchmark/qwen-3.8-27b"


def test_too_long_for_the_gpu_goes_elsewhere():
    bs = backends()
    assert first(bs, states(bs), ask(prompt=700_000)) == "touchmark/qwen-3.8-27b"


def test_failing_backend_is_skipped_then_tested_again():
    bs = backends()
    sts = states(bs, waiting=5)
    tm = sts["touchmark/qwen-3.8-27b"]
    for i in range(5):
        tm.record_failure(NOW - 40 + i * 10, "HTTP 502")
    assert first(bs, sts, ask()) == "vercel"   # the GPU is full: a full backend is the last resort
    assert not tm.usable(NOW)
    assert tm.usable(NOW + 61)


def test_contract_expiry_and_hours():
    bs = backends()
    block = bs["touchmark/qwen-3.8-27b"]
    assert block.active(NOW) == (NOW < block.expires)
    block.active_hours_utc = [8, 20]
    noon = time.mktime((2026, 10, 2, 12, 0, 0, 0, 0, 0)) - time.timezone
    assert block.active(noon) and not block.active(noon + 10 * 3600)
