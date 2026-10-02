# model-gateway

One OpenAI-compatible endpoint for Qwen3.8-27B in front of every place we can run it:

- **our GPU nodes**: vLLM on Runpod pods;
- **Touchmark contracts**: token blocks bought on Touchmark;
- **Vercel AI Gateway**: pay per token, unlimited capacity.

```
clients ──► router (gateway/mgw, ports 8080 TCP / 8090 HTTPS proxy)
              ├─► SMG on 127.0.0.1:30000 ──► vLLM nodes (Runpod pods, port 8000/tcp)
              ├─► touchmark/<model>   one backend per model the Touchmark key lists
              └─► vercel
```

- **SMG** ([Shepherd Model Gateway](https://github.com/lightseekorg/smg)) spreads requests over our nodes.
  - It uses its `cache_aware` policy: it remembers which node has seen which prompt text, and sends a request to the
    node whose cache already holds the longest start of that prompt, unless that node is much busier than the rest.
  - It also retries, runs health checks, and stops sending to a node that keeps failing.
- **The router** (our code) decides, per request, between the GPU pool, the Touchmark blocks and Vercel. SMG has no
  priorities between providers, and does not read contract prices or rate caps.

## How the router places a request

Each backend that can take the request gets a cost estimate in dollars, and the cheapest wins. The cost has four
parts:

1. **Price**: what the backend charges for the tokens.
   - Tokens it likely still holds in its prompt cache from this conversation's earlier turns count at its cached
     price. The rest have to be processed anew ("prefilled").
   - Prepaid capacity (our GPUs, a Touchmark block) costs nothing here.
2. **Load**: what the extra work costs in capacity.
   - This applies only where capacity is limited: our GPU nodes, and a contract's rate cap.
   - It is zero while the backend is less than half full. Past that it rises steeply.
   - So traffic spills over as a backend fills, and comes back when it empties.
3. **Latency**: the expected seconds until the first token, times a configured dollar value per second.
4. **Penalty**: a fixed amount per request, for example for a backend that fails more often.

Some consequences:

- **Uncached sessions spill first.** A new conversation is cached nowhere, so moving it re-processes nothing; a long
  conversation cached on the GPU costs much more to move.
- **Spilled sessions come back.** Say an evaluation overloads the GPU and some conversations go to Vercel. When the
  GPU empties, their next turn comes back to it, if processing their prompt again on the idle GPU costs less than
  Vercel's cached price.
- **A conversation does not move for a small difference.** It moves only when another backend is cheaper by
  `switch_margin`.

**How "full" is measured:**

- **GPU nodes**, from vLLM's `/metrics` every 2 s:
  - requests running versus `--max-num-seqs`;
  - any requests waiting, which means full;
  - prompt tokens queued for processing, versus what a node processes in `ttft_target_s`;
  - output speed per stream, versus `decode_tokens_per_s_min`.
  - KV-cache use is not counted. The nodes copy their cache to CPU memory (DRAM offloading), so a full GPU cache
    does not mean the node is full.
- **Rate-capped contracts:** Tokens used in the last minute, versus the cap.
- **Any backend that answers 429** (too many requests): its concurrency limit is lowered to 70% of what was in flight,
  and it creeps back up while requests succeed.

**Failures:**

- A request that fails before its first token (no connection, 5xx, 429, no first token in time) goes to the next
  backend. When all have failed, it waits and retries, up to `routing.max_wait_s` (10 min).
- A backend that fails 5 times in a row over at least 30 s is skipped for 60 s, then one request tests it.
- An answer that breaks mid-stream cannot move: the client sees it end early.

## Files

| Path | What |
|---|---|
| `gateway/config/gateway.yaml` | The config: backends, prices, contract terms, routing settings. Commented. |
| `gateway/mgw/` | The router. `score.py` holds the cost estimate, `app.py` the request handling, `pool.py` the GPU nodes. |
| `gateway/mgw/adapters/` | Per-provider request fixes. A file of the same name in `/workspace/config/adapters/` replaces one. |
| `gateway/Dockerfile`, `entrypoint.sh` | The image: `lightseekorg/smg:1.11.0` plus the router and SSH. |
| `node/start.sh` | A GPU node's start script: vLLM with the first node's exact settings, port 8000/tcp, restarts itself. |
| `deploy/runpod.py` | Runpod secrets, templates, volumes, pods. |
| `deploy/ecr.py` | Builds the gateway image and pushes it to AWS ECR, where Runpod pulls it from. |

## Running it

On the gateway pod:

- `/workspace/config/gateway.yaml` is the live config. Edit it over SSH. It is re-read within 2 s, and a broken edit
  is ignored (`GET /gateway/status` shows the error).
- SMG and the router restart by themselves if they exit. Logs are in `/workspace/logs/`.

Endpoints:

- `POST /v1/chat/completions`, `/v1/responses`, `/v1/completions`. Use model `qwen3.8-27b`, or any alias in the
  config. The response header `x-gateway-backend` says which backend answered.
- `GET /gateway/status`: every backend's fill, failures, measured speed, and nodes.
- Auth: `Authorization: Bearer $GATEWAY_API_KEY`.

**GPU nodes** come from the Runpod template `qwen3.8-27b-node`:

- `python deploy/runpod.py pod node <name> [gpus]`
- The gateway finds them by their env `GATEWAY_POOL=qwen3.8-27b`. They can also be used directly at
  `http://<ip>:<port mapped to 8000>/v1`, with the same vLLM key.
- 600K context needs at least one H200 (141 GB). On 80 GB cards the cache fits only about 230K tokens.

**Touchmark:**

- The key's `/v1/models` is polled every minute, and each model listed becomes a backend.
- Touchmark's API does not expose contract terms, so they go in `contracts:` in the config, by model id.
- Tested quirks (handled in `adapters/touchmark.py`):
  - Its prompt cache only works with a `prompt_cache_key`. Without one: 0 of 23K tokens cached. With one: 22.7K.
  - Its Responses API writes invalid JSON (`"metadata":{}"background"`) unless the request sets `metadata`.

## Tests

`cd gateway && python -m pytest tests` covers the routing decisions. `tests/fake_vllm.py` is a stand-in vLLM node for
running the whole image without a GPU.
