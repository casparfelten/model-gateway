#!/bin/bash
# Gateway container start: SSH, the config folder, then SMG and the router, each restarted if it exits.
#
#   /workspace/config/gateway.yaml   the config; edit it in place (re-read within 2 s, no restart)
#   /workspace/config/adapters/      adapter files that replace the built-in ones (re-read when changed)
#   /workspace/logs/                 smg.log, gateway.log
#
# Ports: 8080 (TCP) and 8090 (Runpod's HTTPS proxy) both serve the router; SMG listens on 127.0.0.1:30000 only.

mkdir -p /root/.ssh /workspace/config/adapters /workspace/logs
if [ -n "$PUBLIC_KEY" ]; then echo "$PUBLIC_KEY" >> /root/.ssh/authorized_keys; chmod 600 /root/.ssh/authorized_keys; fi
# SSH logins do not see the container's environment; give them the keys and paths too
env | grep -E '^(GATEWAY_|VLLM_API_KEY|RUNPOD_API_KEY|TOUCHMARK_API_KEY|AI_GATEWAY_API_KEY|PATH|PYTHONPATH)=' \
    | sed 's/^/export /; s/=\(.*\)$/="\1"/' > /etc/profile.d/gateway-env.sh
/usr/sbin/sshd

[ -f "$GATEWAY_CONFIG" ] || cp /opt/model-gateway/config/gateway.yaml "$GATEWAY_CONFIG"

forever() {  # forever <name> <command...>: run, and restart 3 s after any exit; log to /workspace/logs/<name>.log
  local name=$1; shift
  while true; do
    log=/workspace/logs/$name.log
    [ -f "$log" ] && [ "$(stat -c %s "$log")" -gt 500000000 ] && mv "$log" "$log.1"
    echo "$(date -Is) starting $name" >> "$log"
    "$@" >> "$log" 2>&1
    echo "$(date -Is) $name exited with $?" >> "$log"
    sleep 3
  done
}

forever smg python3 -m smg.launch_router --host 127.0.0.1 --port 30000 --policy cache_aware \
  --health-check-interval-secs 10 --load-monitor-interval 2 --prometheus-port 29000 --log-level warn &
forever gateway python3 -m mgw "$GATEWAY_CONFIG" &
wait
