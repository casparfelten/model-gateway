"""The GPU pool: which nodes exist (Runpod pods with env GATEWAY_POOL, plus fixed URLs), keeping SMG's worker list equal
to them, and reading each node's load from its /metrics."""
import asyncio
import logging
import os
import time

import httpx

from .config import Backend
from .state import Node, State, update_node

log = logging.getLogger("mgw")
RUNPOD = "https://api.runpod.io/v2"


async def runpod_nodes(http: httpx.AsyncClient, pool: str, port: int = 8000) -> list[str] | None:
    """URLs (http://ip:port) of running pods whose env GATEWAY_POOL is `pool`. None: the API did not answer."""
    key = os.environ.get("GATEWAY_RUNPOD_API_KEY") or os.environ.get("RUNPOD_API_KEY")
    if not key:
        return None
    urls, cursor = [], None
    try:
        while True:
            r = await http.get(f"{RUNPOD}/pods", params={"cursor": cursor} if cursor else None,
                               headers={"Authorization": f"Bearer {key}"}, timeout=20)
            r.raise_for_status()
            body = r.json()
            pods = body.get("pods", body.get("data", [])) if isinstance(body, dict) else body
            for p in pods:
                if (p.get("env") or {}).get("GATEWAY_POOL") != pool or p.get("status") not in ("RUNNING", None):
                    continue
                for m in (p.get("runtime") or {}).get("ports") or []:
                    if m.get("private") == port and m.get("type") == "tcp" and m.get("ip") and m.get("public"):
                        urls.append(f"http://{m['ip']}:{m['public']}")
            page = (body.get("pagination") or {}) if isinstance(body, dict) else {}
            cursor = page.get("nextCursor") if page.get("hasNextPage", True) else None
            if not cursor:
                return urls
    except (httpx.HTTPError, ValueError) as ex:
        log.warning("runpod pod list failed: %s", ex)
        return None


async def sync_smg(http: httpx.AsyncClient, b: Backend, ready: list[str], known: set[str]) -> None:
    """Add the ready nodes SMG lacks, and remove the workers that are no longer nodes at all. (A node that fails for a
    moment stays: SMG's own health checks route around it.)"""
    smg = b.url.removesuffix("/v1")
    r = await http.get(f"{smg}/workers", timeout=10)
    r.raise_for_status()
    have = {w["url"]: w["id"] for w in r.json().get("workers", [])}
    for url in set(ready) - set(have):
        spec = {"url": url, "api_key": b.api_key, "labels": {"source": "model-gateway"}}
        added = await http.post(f"{smg}/workers", json=spec, timeout=30)
        log.info("SMG: added worker %s -> %s %s", url, added.status_code, added.text[:200])
    for url in set(have) - known:
        removed = await http.delete(f"{smg}/workers/{have[url]}", timeout=30)
        log.info("SMG: removed worker %s -> %s", url, removed.status_code)


async def scrape(http: httpx.AsyncClient, b: Backend, st: State) -> None:
    """Every node's /metrics, now."""
    async def one(node: Node):
        try:
            r = await http.get(f"{node.url}/metrics", headers={"Authorization": f"Bearer {b.api_key}"}, timeout=5)
            r.raise_for_status()
            update_node(node, r.text, time.time())
            if node.prefill_tps and node.waiting > 0:   # busy: its prefill speed is near its capacity
                st.prefill_tps = max(st.prefill_tps or 0, node.prefill_tps)
        except httpx.HTTPError:
            node.ok = False
    await asyncio.gather(*(one(n) for n in list(st.nodes.values())))


async def discover(http: httpx.AsyncClient, b: Backend, st: State) -> None:
    """Refresh the node list (Runpod + fixed workers) and SMG's workers."""
    urls = [w.rstrip("/") for w in b.workers]
    if b.runpod_pool:
        found = await runpod_nodes(http, b.runpod_pool)
        if found is None:   # Runpod did not answer: keep the nodes we know
            urls += [u for u in st.nodes if u not in urls]
        else:
            urls += found
    for url in urls:
        st.nodes.setdefault(url, Node(url))
    for url in [u for u in st.nodes if u not in urls]:
        del st.nodes[url]
    # Register only nodes whose vLLM answers (a booting node would fail SMG's checks and be dropped)
    await scrape(http, b, st)
    await sync_smg(http, b, [u for u, n in st.nodes.items() if n.ok], set(st.nodes))
