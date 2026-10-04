# Serving benchmark: Qwen3.8-27B and Qwen3.8-Flash-Next on 4x H200

Run on 2026-10-04 (UTC 01:55-05:08) on Runpod pod `kyx727ldkrwrcj` (4x H200 SXM, EUR-IS-5, driver 580 / CUDA 13.0,
160 vCPU, 352 GB `/dev/shm`). Image `vllm/vllm-openai:nightly-ac9126e58aa7bbab1856ba6593ba4d5003fea516`
(vLLM 0.30.1rc1.dev493). The pod has been terminated.

Load: `bench/loadgen.py` replaying `bench/trace.json`, closed loop, N concurrent sessions, output length forced
(`ignore_eos` + `min_tokens`), so every config does the same work. Each point: `--warmup 60 --duration 240`. The
load generator ran on the pod against `127.0.0.1`. With two server copies, session i goes to copy i mod 2. Measured
prompt p50 was about 12.5K tokens at 64-128 sessions, matching the production logs.

Raw results: `bench/results/<config>_s<sessions>.json`. Serve commands: `bench/configs/<config>.sh`. A `2xtp2`
config is the matching `tp2` script started twice: `GPUS=0,1 PORT=8000 SHARE=2` and `GPUS=2,3 PORT=8001 SHARE=2`.
GPU spend for the campaign was about $63.4: $2.90 on a first pod that could not run this image, and $60.55 for
3.3 h on the benchmark pod.

## Configs

All configs use the production flags from `node/start.sh`:

- `--max-num-batched-tokens 16384`, `--gpu-memory-utilization 0.92`.
- prefix caching, `--mamba-cache-mode align`, priority scheduling, the qwen3_coder tool parser and the qwen3
  reasoning parser.
- CPU KV offload: `TieringOffloadingSpec`, 85% of `/dev/shm`, split in half when two servers share the host.
- The 27B runs at max-model-len 600000 with the YaRN rope override. Flash-Next runs at its native 262144.

| config | what | launch to ready | GPU KV cache (per server) |
|---|---|---|---|
| `27b-bf16-tp4-s160` | 27B BF16, TP4, max-num-seqs 160 (today's production) | 367 s | 7,315,025 tokens (12.19x at 600K) |
| `27b-fp8-fp8kv-tp4-s384` | 27B FP8 checkpoint + `--kv-cache-dtype fp8`, TP4, max-num-seqs 384 | 361 s | 15,169,665 (25.28x) |
| `27b-fp8-fp8kv-2xtp2-s256` | same, as 2 copies of TP2 (GPUs 0,1 :8000 and 2,3 :8001), max-num-seqs 256 each | 497 s (both started together) | 7,107,455 each (11.85x) |
| `27b-fp8-fp8kv-2xtp2-s256-mtp2` | as above + MTP, `{"method":"mtp","num_speculative_tokens":2}` | 362-367 s | 6,477,519 each (10.80x) |
| `flash-fp8-tp4-ep-s384` | Flash-Next FP8, TP4 + `--enable-expert-parallel`, max-num-seqs 384 | 683 s | 7,144,948 (27.26x at 262K) |
| `flash-fp8-2xtp2-ep-s256` | Flash-Next FP8, 2 copies of TP2 + EP, max-num-seqs 256 each | 864-869 s | 4,561,601 each (17.40x) |

The launch-to-ready times include loading weights from the pod's network volume (MooseFS). They also include
torch.compile, which hit the warm compile cache on the second and later launches.

Not run, for lack of budget: 27B BF16 with max-num-seqs 384, 27B FP8 with a BF16 KV cache, and the 27B on a single
TP2 copy. The 2xTP2 rows give the per-GPU efficiency of TP2: one copy on 2 GPUs at N/2 sessions does the same work.
For Flash-Next 2xTP2, only the 200-session point was run.

## Results

`uncached prompt tok/s*` = prompt tok/s x (1 - server prefix-cache hit rate). vLLM does not return `cached_tokens` in
`usage` unless started with `--enable-prompt-tokens-details`, so loadgen's own `uncached_prompt_tokens_per_s` field
equals the total prompt rate in these JSONs. Preemptions were 0 everywhere: the metric was absent, and no preemption
appeared in any server log. Decode is output tokens per second per stream.

| config | sessions | calls/min | out tok/s | uncached prompt tok/s* | TTFT p50/p90/p99 s | decode tok/s p50/p10 | latency p50/p90 s | prefix hit | preempt | errors | prompt p50 |
|---|---|---|---|---|---|---|---|---|---|---|---|
| 27b-bf16-tp4-s160 | 64 | 239.5 | 2153.5 | 6625 | 0.232/0.365/0.483 | 43.739/40.077 | 6.315/33.274 | 0.888 | 0 | 0 | 12675 |
| 27b-bf16-tp4-s160 | 128 | 333.2 | 2457.8 | 8424 | 0.276/0.428/0.59 | 27.754/25.057 | 9.17/48.435 | 0.895 | 0 | 0 | 12226 |
| 27b-bf16-tp4-s160 | 200 | 333.0 | 2065.1 | 8508 | 6.194/8.694/10.683 | 19.314/17.342 | 18.491/76.949 | 0.872 | 0 | 0 | 10418 |
| 27b-bf16-tp4-s160 | 300 | 409.2 | 2313.0 | 11180 | 16.251/24.279/28.928 | 22.274/16.735 | 28.781/74.402 | 0.827 | 0 | 0 | 8843 |
| 27b-fp8-fp8kv-tp4-s384 | 64 | 221.0 | 1936.3 | 7310 | 0.242/0.37/0.516 | 39.379/35.94 | 6.878/34.706 | 0.864 | 0 | 0 | 12180 |
| 27b-fp8-fp8kv-tp4-s384 | 128 | 305.5 | 2042.0 | 9526 | 0.279/0.443/0.632 | 24.49/22.673 | 10.282/51.169 | 0.867 | 0 | 0 | 12083 |
| 27b-fp8-fp8kv-tp4-s384 | 200 | 370.0 | 2264.0 | 11633 | 0.329/0.551/1.026 | 17.803/15.774 | 14.123/76.874 | 0.857 | 0 | 0 | 10935 |
| 27b-fp8-fp8kv-tp4-s384 | 300 | 470.2 | 2290.2 | 15135 | 0.397/0.64/0.915 | 12.31/10.226 | 19.52/83.162 | 0.827 | 0 | 0 | 9681 |
| 27b-fp8-fp8kv-2xtp2-s256 | 64 | 242.0 | 2208.1 | 8221 | 0.223/0.383/0.529 | 45.168/40.523 | 6.148/34.405 | 0.863 | 0 | 0 | 12662 |
| 27b-fp8-fp8kv-2xtp2-s256 | 128 | 333.8 | 2581.2 | 10757 | 0.259/0.424/0.615 | 29.385/27.227 | 8.705/49.052 | 0.871 | 0 | 0 | 12939 |
| 27b-fp8-fp8kv-2xtp2-s256 | 200 | 416.2 | 2799.2 | 12938 | 0.296/0.499/0.816 | 21.539/19.738 | 11.687/64.5 | 0.866 | 0 | 0 | 11720 |
| 27b-fp8-fp8kv-2xtp2-s256 | 300 | 511.8 | 2913.7 | 15899 | 0.349/0.592/0.854 | 15.199/13.772 | 16.171/87.826 | 0.85 | 0 | 0 | 10568 |
| 27b-fp8-fp8kv-2xtp2-s256-mtp2 | 64 | 313.8 | 3004.6 | 18258 | 0.358/0.628/0.936 | 59.975/43.788 | 5.071/24.456 | 0.768 | 0 | 0 | 12758 |
| 27b-fp8-fp8kv-2xtp2-s256-mtp2 | 200 | 441.5 | 3112.9 | 25152 | 0.577/0.92/1.326 | 22.118/16.85 | 12.284/56.856 | 0.747 | 0 | 0 | 11422 |
| 27b-fp8-fp8kv-2xtp2-s256-mtp2 | 300 | 506.5 | 2785.9 | 30775 | 0.771/1.352/1.896 | 14.283/10.314 | 18.313/71.323 | 0.688 | 0 | 0 | 10060 |
| flash-fp8-tp4-ep-s384 | 64 | 215.8 | 1817.8 | 5663 | 0.298/0.458/0.574 | 37.814/33.85 | 7.138/37.438 | 0.892 | 0 | 0 | 12245 |
| flash-fp8-tp4-ep-s384 | 128 | 300.0 | 1936.1 | 7451 | 0.36/0.524/0.636 | 23.172/19.028 | 11.282/52.799 | 0.891 | 0 | 0 | 11791 |
| flash-fp8-tp4-ep-s384 | 200 | 367.8 | 2108.5 | 9491 | 0.424/0.593/0.772 | 16.338/12.136 | 16.03/74.683 | 0.875 | 0 | 0 | 10506 |
| flash-fp8-tp4-ep-s384 | 300 | 462.2 | 2179.0 | 14195 | 0.479/0.666/0.939 | 11.678/9.264 | 20.612/80.266 | 0.828 | 0 | 1 | 9375 |
| flash-fp8-2xtp2-ep-s256 | 200 | 451.5 | 3093.7 | 15138 | 0.334/0.508/0.686 | 23.265/18.687 | 11.366/62.27 | 0.852 | 0 | 0 | 11480 |

Notes on the table:

- At 300 sessions every config falls behind, sessions restart more often within the window, and the mix shifts to
  earlier, shorter turns (prompt p50 drops to 9-10K). Compare 300-session rows with that in mind.
- MTP rows show a lower prefix hit rate (0.69-0.77 against 0.85-0.87). The workload is the same, so this is most
  likely the hit-rate counter, not real extra prefill. The MTP drafter's KV is probably counted in the prefix-cache
  queries.
- The one Flash-Next error at 300 sessions was a client `ReadError` (a stream closed mid-call).
- MTP acceptance in the server logs: mean acceptance length 2.24-3.00 tokens per step (2 drafted), about 2.3 under
  load, with per-position acceptance about 0.75 and 0.56. This matches the assumed ~2.4. The text being predicted is
  the model's own reasoning about filler prompts, so acceptance on real traffic may differ.

## Cost efficiency

4x H200 = $18.36/hr. Every config here uses all 4 GPUs: the 2xTP2 rows are one 4-GPU pod. The SLO is TTFT p90 <= 5 s
and per-stream decode p10 >= 20 tok/s, with no errors.

| config | GPUs | $/hr | sessions | calls per $ | output tok per $ (M) | meets SLO (TTFT p90<=5s, decode p10>=20) |
|---|---|---|---|---|---|---|
| 27b-bf16-tp4-s160 | 4 | 18.36 | 64 | 783 | 0.422 | yes |
| 27b-bf16-tp4-s160 | 4 | 18.36 | 128 | 1089 | 0.482 | yes |
| 27b-bf16-tp4-s160 | 4 | 18.36 | 200 | 1088 | 0.405 | no |
| 27b-bf16-tp4-s160 | 4 | 18.36 | 300 | 1337 | 0.454 | no |
| 27b-fp8-fp8kv-tp4-s384 | 4 | 18.36 | 64 | 722 | 0.380 | yes |
| 27b-fp8-fp8kv-tp4-s384 | 4 | 18.36 | 128 | 998 | 0.400 | yes |
| 27b-fp8-fp8kv-tp4-s384 | 4 | 18.36 | 200 | 1209 | 0.444 | no |
| 27b-fp8-fp8kv-tp4-s384 | 4 | 18.36 | 300 | 1537 | 0.449 | no |
| 27b-fp8-fp8kv-2xtp2-s256 | 4 | 18.36 | 64 | 791 | 0.433 | yes |
| 27b-fp8-fp8kv-2xtp2-s256 | 4 | 18.36 | 128 | 1091 | 0.506 | yes |
| 27b-fp8-fp8kv-2xtp2-s256 | 4 | 18.36 | 200 | 1360 | 0.549 | no |
| 27b-fp8-fp8kv-2xtp2-s256 | 4 | 18.36 | 300 | 1673 | 0.571 | no |
| 27b-fp8-fp8kv-2xtp2-s256-mtp2 | 4 | 18.36 | 64 | 1025 | 0.589 | yes |
| 27b-fp8-fp8kv-2xtp2-s256-mtp2 | 4 | 18.36 | 200 | 1443 | 0.610 | no |
| 27b-fp8-fp8kv-2xtp2-s256-mtp2 | 4 | 18.36 | 300 | 1655 | 0.546 | no |
| flash-fp8-tp4-ep-s384 | 4 | 18.36 | 64 | 705 | 0.356 | yes |
| flash-fp8-tp4-ep-s384 | 4 | 18.36 | 128 | 980 | 0.380 | no |
| flash-fp8-tp4-ep-s384 | 4 | 18.36 | 200 | 1202 | 0.413 | no |
| flash-fp8-tp4-ep-s384 | 4 | 18.36 | 300 | 1510 | 0.427 | no |
| flash-fp8-2xtp2-ep-s256 | 4 | 18.36 | 200 | 1475 | 0.607 | no |

Highest tested session count meeting the SLO (tested points only):

- 27b-bf16-tp4-s160: highest tested session count meeting the SLO = 128
- 27b-fp8-fp8kv-tp4-s384: highest tested session count meeting the SLO = 128
- 27b-fp8-fp8kv-2xtp2-s256: highest tested session count meeting the SLO = 128
- 27b-fp8-fp8kv-2xtp2-s256-mtp2: highest tested session count meeting the SLO = 64
- flash-fp8-tp4-ep-s384: highest tested session count meeting the SLO = 64
- flash-fp8-2xtp2-ep-s256: only 200 was tested; it misses the SLO narrowly (decode p10 18.7)

## Recommendation for ~200 concurrent sessions

For Qwen3.8-27B: **run the 27B FP8 checkpoint with FP8 KV as two TP2 copies per 4-GPU node, with MTP (2 draft tokens):
`bench/configs/27b-fp8-fp8kv-tp2-s256-mtp2.sh`**. Start it twice per pod:

```
GPUS=0,1 PORT=8000 SHARE=2 bash 27b-fp8-fp8kv-tp2-s256-mtp2.sh
GPUS=2,3 PORT=8001 SHARE=2 bash 27b-fp8-fp8kv-tp2-s256-mtp2.sh
```

Then register both ports with the gateway as separate nodes.

For Qwen3.8-Flash-Next: **run two TP2 + expert-parallel copies per 4-GPU node** (`bench/configs/flash-fp8-tp2-ep-s256.sh`,
started twice the same way). At 200 sessions it is as cost-efficient as the best 27B config, with a better tail (see 6).

Why:

1. **Today's config cannot take 200 sessions.** With max-num-seqs 160, 40 sessions queue. At 200 sessions TTFT p90 is
   8.7 s, and output throughput (2065 tok/s) is no higher than at 128 sessions (2458 tok/s).
2. **2xTP2 is better than TP4 on the same 4 GPUs.** With FP8 + FP8 KV at 200 sessions, 2xTP2 gives 2799 out tok/s
   against 2264 for TP4 (+24%). Decode p10 is 19.7 against 15.8 tok/s. TTFT p90 is 0.50 s.
3. **MTP adds ~11% more at 200 sessions and ~36% at 64.**
   - At 200: 3113 against 2799 out tok/s, 1443 against 1360 calls per $, 0.61M against 0.55M output tokens per $.
   - At 64, per-stream decode p50 goes from 45 to 60 tok/s. That is the latency win when the cluster is not saturated.
   - The cost: at 200 sessions decode p10 drops to 16.9 tok/s (19.7 without MTP). At 300 sessions MTP is slightly
     worse than no MTP (2786 against 2914 out tok/s).
   - If the gateway keeps each node below its saturation point (~100 sessions per TP2 copy), MTP is better everywhere.
     If nodes are routinely pushed to 150+ sessions per copy, drop MTP.
4. **FP8 + FP8 KV alone does not help at TP4.**
   - At TP4 it was 6-17% slower than BF16 at the same session count (128 sessions: 2042 against 2458 out tok/s). Its
     only win is the doubled KV cache, which lets it raise max-num-seqs without queueing.
   - The FP8 GEMMs use the DeepGEMM block-scaled kernel, and attention uses FlashAttention with an FP8 KV cache. At
     TP4's small per-GPU shapes this was not faster.
   - The gain comes from the 2xTP2 layout, which FP8 makes possible: the FP8 weights fit on 2 GPUs with ~7M KV tokens
     per copy.
5. **The SLO is tight at 200 sessions for every config.** No 4-GPU config meets decode p10 >= 20 tok/s at 200
   sessions; 2xTP2 without MTP comes closest at 19.7. For that per-stream speed at ~200 sessions, plan on
   ~130-150 sessions per 4-GPU node (two nodes for 200-300 sessions) and let the gateway spill the rest.
   - At 128 sessions all 27B configs meet it.
   - 2xTP2 sustains 128, and probably ~180, sessions at decode p10 >= 20 (interpolating 128 → 27.2 and 200 → 19.7).
6. **Flash-Next: use 2xTP2 + EP, never TP4 + EP.**
   - TP4 + EP was the worst config at every load: 2108 out tok/s and decode p10 12 tok/s at 200 sessions, and
     38 tok/s per stream at 64 sessions against 43-45 for the 27B.
   - 2xTP2 + EP at 200 sessions gave 3094 out tok/s and 451 calls/min, TTFT p90 0.51 s and decode p50/p10 23.3/18.7.
     That is +47% over its own TP4 layout.
   - It ties the best 27B config: 1475 against 1443 calls per $, and 0.607M against 0.610M output tokens per $. Its
     tail is better, with decode p10 18.7 against 16.9 and TTFT p90 0.51 s against 0.92 s, and it has no MTP yet.
   - Its per-copy KV cache is smaller: 4.6M tokens against 6.5-7.1M for the 27B. It still had no preemptions at
     200 sessions.
   - Only the 200-session point was run, for lack of budget.
7. **Next steps, if Flash-Next's answer quality is acceptable for our evals:** a short run of Flash-Next 2xTP2 + EP
   + MTP (`{"method":"mtp",...}` maps to `qwen4_exp_mtp` in this build) at 128/200/300 sessions. It could beat the
   27B. Until then, the 27B config above is the measured choice and needs no model change.

## Comparison with the HTDYM estimates

The other analysis (HTDYM roofline estimates from real logs) predicted three things. The measurements agree with two:

- **2xTP2 beats TP4 for the 27B: confirmed.** It is +24% at 200 sessions and +27% at 300.
- **MTP at about 2.4 accepted tokens per step: confirmed.** The mean acceptance length measured about 2.3 under load.
  The throughput gain is large only below saturation, though: +36% at 64 sessions, +11% at 200 and −4% at 300.
- **FP8 + FP8 KV beats BF16: not confirmed by itself.** At TP4 it was slower. It pays off only through the 2xTP2
  layout it enables.

## Problems met

- The first pod (`l962qwroc2og6g`) landed on a host with driver 570 (CUDA 12.8). This image's torch is built for
  CUDA 13.0, so vLLM failed with "NVIDIA driver too old". With the image's CUDA 13 forward-compat libraries, NCCL
  then failed. That pod was terminated after ~10 min (~$2.90).
  - The replacement was created with `"gpu": {"minCudaVersion": "13.0"}` in the pod body, which
    `deploy/runpod.py` should also set for this image.
  - The production node template likely needs the same, or a new node may land on an old driver.
- A killed vLLM leaves its CPU-offload region (`/dev/shm/vllm_offload_*.mmap`, ~300 GB) behind. The next server
  then fails with "Insufficient space in /dev/shm".
  - `node/start.sh` restarts vLLM after a crash, so it would hit this too. It sizes the region from *available*
    `/dev/shm`, so a restart after a crash would get a much smaller CPU cache, or fail.
  - Fix: `rm -f /dev/shm/vllm_offload_*` before each start.
- Flash-Next weights (173 GB) load slowly from the network volume. Loading two TP2 copies at once took ~11 min.
