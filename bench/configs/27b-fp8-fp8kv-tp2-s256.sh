#!/bin/bash
# Benchmark config: 27b-fp8-fp8kv-tp2-s256
# One vLLM server. Env: GPUS (CUDA_VISIBLE_DEVICES, default 0,1,2,3; TP = number of GPUs), PORT (default 8000),
# SHARE (servers sharing this host's /dev/shm for the CPU KV copy, default 1).
export PYTHONHASHSEED=0 VLLM_ALLOW_LONG_MAX_MODEL_LEN=1 CUDA_VISIBLE_DEVICES=${GPUS:-0,1,2,3}
PORT=${PORT:-8000}; SHARE=${SHARE:-1}
TP=$(echo $CUDA_VISIBLE_DEVICES | tr , '\n' | wc -l)
CPU=$(( $(df -B1 --output=size /dev/shm | tail -1) * 85 / 100 / SHARE ))  # 85% of /dev/shm, split between servers
OFF='{"kv_connector":"OffloadingConnector","kv_role":"kv_both","kv_connector_extra_config":{"spec_name":"TieringOffloadingSpec","cpu_bytes_to_use":'$CPU',"blocks_per_chunk":1,"eviction_policy":"lru"}}'
ROPE='{"text_config":{"rope_parameters":{"mrope_interleaved":true,"mrope_section":[11,11,10],"rope_type":"yarn","rope_theta":10000000,"partial_rotary_factor":0.25,"factor":4.0,"original_max_position_embeddings":262144}}}'
exec vllm serve /workspace/models/Qwen3.8-27B-FP8 --served-model-name Qwen/Qwen3.8-27B \
  --tensor-parallel-size $TP --max-model-len 600000 --max-num-batched-tokens 16384 --gpu-memory-utilization 0.92 \
  --enable-auto-tool-choice --enable-prefix-caching --scheduling-policy priority \
  --mamba-cache-mode align --max-num-seqs 256 --tool-call-parser qwen3_coder --reasoning-parser qwen3 \
  --kv-transfer-config "$OFF" --hf-overrides "$ROPE" --kv-cache-dtype fp8 \
  --api-key "$VLLM_API_KEY" --host 0.0.0.0 --port $PORT
