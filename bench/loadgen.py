"""Load test that replays our real eval traffic against an OpenAI-compatible server (vLLM, SGLang, the gateway).

Each virtual session replays one recorded session from trace.json: its first prompt, then per turn the recorded number
of new prompt tokens (tool results) and of output tokens, with the recorded pause before the call. The conversation
really grows: each answer is appended, so the server's prefix cache behaves as in production. `--sessions` sessions
run at once; when one ends, the next starts (closed loop).

Output length is forced to the recorded length (`max_tokens` with `ignore_eos`, which vLLM and SGLang accept), so every
configuration does exactly the same work.

    python bench/loadgen.py --url http://127.0.0.1:8000/v1 --key $VLLM_API_KEY --model Qwen/Qwen3.8-27B \
        --sessions 200 --warmup 120 --duration 600 --out results/run.json

`--url` may list several servers separated by commas (e.g. two TP2 copies on one 4-GPU pod): each session stays on one
of them (session i uses server i mod n), as a cache-aware router would keep it.

Reported over the measured window: requests and tokens per second, time to first token (TTFT), output speed per
stream, whole-call latency, the server's prefix-cache hit rate, errors.
"""
import argparse
import asyncio
import json
import random
import re
import statistics
import time
from pathlib import Path

import httpx

WORDS = ("time year people way day man thing woman life child world school state family student group country problem "
         "hand part place case week company system program question work government number night point home water room "
         "mother area money story fact month lot right study book eye job word business issue side kind head house "
         "service friend father power hour game line end member law car city community name president team minute idea "
         "kid body information back parent face others level office door health person art war history party result "
         "change morning reason research girl guy moment air teacher force education foot boy age policy process music "
         "market sense nation plan college interest death experience effect use class control care field development role "
         "effort rate heart drug show leader light voice wife police mind price report decision son view relationship town "
         "road arm difference value building action model season society tax director position player record paper space "
         "ground form event official matter center couple site project activity star table need court oil situation cost "
         "industry figure street image phone data picture practice piece land product doctor wall patient worker news test "
         "movie north love support technology step baby computer type attention film tree source organization hair window "
         "evidence population site invoice order vendor account ledger payment customer email record status amount total").split()


class Filler:
    """Text of about n tokens. Word lists like this come out near 1 token per word; the ratio is corrected from the
    prompt token counts the server reports."""

    def __init__(self, seed: int):
        self.rng = random.Random(seed)
        self.tokens_per_word = 1.1

    def text(self, tokens: int) -> str:
        n = max(1, int(tokens / self.tokens_per_word))
        return " ".join(self.rng.choice(WORDS) for _ in range(n))


class Stats:
    def __init__(self):
        self.calls: list[dict] = []
        self.errors: list[str] = []
        self.window = (0.0, float("inf"))

    def add(self, c: dict):
        self.calls.append(c)


async def call(client, args, url, messages, out_tokens, stats, filler, measure):
    body = {"model": args.model, "messages": messages, "max_tokens": max(1, out_tokens), "stream": True,
            "stream_options": {"include_usage": True}, "temperature": 0.7}
    if args.force_length:
        body["ignore_eos"] = True
        body["min_tokens"] = max(1, out_tokens)
    if args.extra:
        body.update(json.loads(args.extra))
    t0 = time.time()
    first = None
    text, usage = [], None
    try:
        async with client.stream("POST", f"{url}/chat/completions", json=body,
                                 headers={"Authorization": f"Bearer {args.key}"}, timeout=args.timeout) as r:
            if r.status_code != 200:
                stats.errors.append(f"HTTP {r.status_code}: {(await r.aread())[:200]!r}")
                return None
            async for line in r.aiter_lines():
                if not line.startswith("data:") or line.strip() == "data: [DONE]":
                    continue
                d = json.loads(line[5:])
                for ch in d.get("choices") or []:
                    delta = ch.get("delta") or {}
                    piece = (delta.get("content") or "") + (delta.get("reasoning") or delta.get("reasoning_content") or "")
                    if piece:
                        if first is None:
                            first = time.time()
                        text.append(delta.get("content") or "")
                if d.get("usage"):
                    usage = d["usage"]
    except (httpx.HTTPError, ValueError) as ex:
        stats.errors.append(f"{type(ex).__name__}: {ex}"[:200])
        return None
    t1 = time.time()
    u = usage or {}
    prompt = u.get("prompt_tokens", 0)
    cached = (u.get("prompt_tokens_details") or {}).get("cached_tokens") or 0
    out = u.get("completion_tokens", 0)
    if measure(t0):
        stats.add({"start": t0, "end": t1, "ttft": (first or t1) - t0, "prompt": prompt, "cached": cached, "out": out,
                   "decode_tps": (out - 1) / (t1 - first) if first and out > 1 and t1 > first else None})
    return "".join(text) or "ok"


async def session(client, args, trace, stats, filler, stop_at, measure, sid):
    urls = args.url.split(",")
    url = urls[sid % len(urls)]
    while time.time() < stop_at:
        turns = random.choice(trace)
        messages = [{"role": "system", "content": f"[session {sid}-{random.random():.6f}] " + filler.text(turns[0][0])}]
        messages.append({"role": "user", "content": filler.text(50)})
        for i, (grow, out, gap) in enumerate(turns):
            if time.time() >= stop_at:
                return
            if i > 0:
                await asyncio.sleep(gap)
                messages.append({"role": "user", "content": filler.text(grow)})
            answer = await call(client, args, url, messages, min(out, args.max_output), stats, filler, measure)
            if answer is None:
                await asyncio.sleep(1)
                break   # a failed call ends this session; a new one starts
            messages.append({"role": "assistant", "content": answer})


def pct(a, p):
    a = sorted(x for x in a if x is not None)
    return round(a[min(len(a) - 1, int(p * len(a)))], 3) if a else None


async def server_metrics(client, args) -> dict:
    """vLLM/SGLang counters, summed over all servers."""
    out = {}
    for url in args.url.split(","):
        try:
            r = await client.get(url.removesuffix("/v1") + "/metrics", headers={"Authorization": f"Bearer {args.key}"},
                                 timeout=10)
        except httpx.HTTPError:
            continue
        for name, value in re.findall(r"^(vllm:[a-z_]+|sglang:[a-z_]+)(?:\{[^}]*\})? ([0-9.eE+-]+)$", r.text, re.M):
            out[name] = out.get(name, 0.0) + float(value)
    return out


async def main(args):
    trace = json.loads(Path(args.trace).read_text())["sessions"]
    if args.max_context:
        trace = [t for t in trace if sum(g + o for g, o, _ in t) < args.max_context]
    random.seed(args.seed)
    filler = Filler(args.seed)
    stats = Stats()
    start = time.time()
    begin, end = start + args.warmup, start + args.warmup + args.duration
    measure = lambda t0: begin <= t0 < end
    limits = httpx.Limits(max_connections=args.sessions + 50, max_keepalive_connections=args.sessions + 50)
    async with httpx.AsyncClient(limits=limits) as client:
        before = None

        async def snapshot():
            nonlocal before
            await asyncio.sleep(args.warmup)
            before = await server_metrics(client, args)

        snap = asyncio.create_task(snapshot())
        await asyncio.gather(*(session(client, args, trace, stats, filler, end, measure, i)
                               for i in range(args.sessions)))
        await snap
        after = await server_metrics(client, args)
    c = stats.calls
    window = args.duration
    finished = [x for x in c if x["end"] <= end]
    m = lambda k: (after.get(k, 0) - (before or {}).get(k, 0))
    hits, queries = m("vllm:prefix_cache_hits_total"), m("vllm:prefix_cache_queries_total")
    result = {
        "config": vars(args),
        "calls": len(c), "errors": len(stats.errors), "error_samples": stats.errors[:5],
        "calls_per_min": round(len(finished) / window * 60, 1),
        "output_tokens_per_s": round(sum(x["out"] for x in finished) / window, 1),
        "prompt_tokens_per_s": round(sum(x["prompt"] for x in finished) / window, 1),
        "uncached_prompt_tokens_per_s": round(sum(x["prompt"] - x["cached"] for x in finished) / window, 1),
        "ttft_s": {"p50": pct([x["ttft"] for x in c], .5), "p90": pct([x["ttft"] for x in c], .9),
                   "p99": pct([x["ttft"] for x in c], .99)},
        "decode_tokens_per_s_per_stream": {"p50": pct([x["decode_tps"] for x in c], .5),
                                           "p10": pct([x["decode_tps"] for x in c], .1)},
        "call_latency_s": {"p50": pct([x["end"] - x["start"] for x in c], .5),
                           "p90": pct([x["end"] - x["start"] for x in c], .9)},
        "prompt_tokens": {"p50": pct([x["prompt"] for x in c], .5), "p90": pct([x["prompt"] for x in c], .9)},
        "client_cached_share": round(sum(x["cached"] for x in c) / max(1, sum(x["prompt"] for x in c)), 3),
        "server_prefix_hit_rate": round(hits / queries, 3) if queries else None,
        "server_preemptions": m("vllm:num_preemptions_total") or None,
    }
    print(json.dumps(result, indent=1))
    if args.out:
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        Path(args.out).write_text(json.dumps(result, indent=1))


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--url", required=True)
    p.add_argument("--key", default="none")
    p.add_argument("--model", required=True)
    p.add_argument("--sessions", type=int, default=200)
    p.add_argument("--warmup", type=float, default=120)
    p.add_argument("--duration", type=float, default=600)
    p.add_argument("--trace", default=str(Path(__file__).with_name("trace.json")))
    p.add_argument("--max-output", type=int, default=8192, help="cap on one call's output tokens")
    p.add_argument("--max-context", type=int, default=0, help="skip recorded sessions longer than this")
    p.add_argument("--force-length", type=int, default=1, help="1: exact output length (ignore_eos)")
    p.add_argument("--extra", default="", help="JSON merged into every request body")
    p.add_argument("--timeout", type=float, default=1800)
    p.add_argument("--seed", type=int, default=1)
    p.add_argument("--out", default="")
    asyncio.run(main(p.parse_args()))
