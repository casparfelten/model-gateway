#!/bin/bash
# GPU node: vLLM serving Qwen/Qwen3.8-27B with the same settings as the first node (zc6m1kj9cpweg8).
# Runs as the pod's start command (deploy/runpod.py puts this file into the template).
#
# Ports (exposed as TCP by the template, so no 100 s proxy limit):
#   8000  vLLM (OpenAI API, /metrics). Auth: Bearer $VLLM_API_KEY.
#   8001, 8002  spare (nothing listens until you start something there).
#   22    SSH (key from the Runpod account, $PUBLIC_KEY).
# The gateway finds this pod through the Runpod API (env GATEWAY_POOL) and reads its public IP and port mapping from
# there; nothing here has to know the gateway.
# vLLM restarts by itself if it dies. To change flags: edit /workspace/node-serve.sh over SSH, then
# `kill $(cat /workspace/vllm.pid)` (it comes back with the new flags in about 10 s plus load time).

# SSH
apt-get update -qq && apt-get install -y -qq openssh-server curl >/dev/null 2>&1
mkdir -p /root/.ssh /run/sshd
[ -n "$PUBLIC_KEY" ] && echo "$PUBLIC_KEY" >> /root/.ssh/authorized_keys
service ssh start

# Model weights on the volume (kept across restarts)
M=Qwen/Qwen3.8-27B
mkdir -p /workspace/models
pip install -q huggingface_hub
[ -f /workspace/models/${M#*/}/config.json ] || hf download $M --local-dir /workspace/models/${M#*/}

# The serve command, editable in place. Same flags as the first node, port 8000.
[ -f /workspace/node-serve.sh ] || cat > /workspace/node-serve.sh <<'EOF'
#!/bin/bash
export PYTHONHASHSEED=0 VLLM_ALLOW_LONG_MAX_MODEL_LEN=1
TP=$(nvidia-smi -L | wc -l)
CPU=$(( $(df -B1 --output=avail /dev/shm | tail -1) * 85 / 100 ))  # 85% of /dev/shm for the CPU copy of the KV cache
OFF='{"kv_connector":"OffloadingConnector","kv_role":"kv_both","kv_connector_extra_config":{"spec_name":"TieringOffloadingSpec","cpu_bytes_to_use":'$CPU',"blocks_per_chunk":1,"eviction_policy":"lru"}}'
ROPE='{"text_config":{"rope_parameters":{"mrope_interleaved":true,"mrope_section":[11,11,10],"rope_type":"yarn","rope_theta":10000000,"partial_rotary_factor":0.25,"factor":4.0,"original_max_position_embeddings":262144}}}'
exec vllm serve /workspace/models/Qwen3.8-27B --served-model-name Qwen/Qwen3.8-27B \
  --tensor-parallel-size $TP --max-model-len 600000 --max-num-batched-tokens 16384 --gpu-memory-utilization 0.92 \
  --enable-auto-tool-choice --enable-prefix-caching --scheduling-policy priority \
  --mamba-cache-mode align --max-num-seqs 160 --tool-call-parser qwen3_coder --reasoning-parser qwen3 \
  --hf-overrides "$ROPE" --kv-transfer-config "$OFF" \
  --api-key "$VLLM_API_KEY" --host 0.0.0.0 --port 8000
EOF

# Run vLLM forever: restart it 10 s after it exits
while true; do
  pkill -9 -f '^VLLM::' 2>/dev/null  # engine/worker processes left by a crash still hold GPU memory
  rm -f /dev/shm/vllm_offload_*      # and its CPU copy of the KV cache (~85% of /dev/shm) would leave no room
  echo "$(date -Is) starting vLLM" >> /workspace/vllm.log
  bash /workspace/node-serve.sh >> /workspace/vllm.log 2>&1 &
  echo $! > /workspace/vllm.pid
  wait $!
  echo "$(date -Is) vLLM exited ($?)" >> /workspace/vllm.log
  sleep 10
done
