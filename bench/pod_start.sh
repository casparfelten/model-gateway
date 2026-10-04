#!/bin/bash
# Benchmark pod start: SSH, and the command endpoint (bench/ctl.py, embedded below by deploy/runpod.py) on port 8888,
# reached through Runpod's HTTPS proxy. Nothing else starts: the benchmark driver starts vLLM/SGLang configs itself.
apt-get update -qq && apt-get install -y -qq openssh-server curl >/dev/null 2>&1
mkdir -p /root/.ssh /run/sshd /workspace/jobs /workspace/models
[ -n "$PUBLIC_KEY" ] && echo "$PUBLIC_KEY" >> /root/.ssh/authorized_keys
service ssh start
cat > /ctl.py <<'CTL_EOF'
__CTL_PY__
CTL_EOF
while true; do python3 /ctl.py >> /workspace/jobs/ctl.log 2>&1; sleep 2; done
