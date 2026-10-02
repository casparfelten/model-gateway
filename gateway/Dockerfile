# The gateway: SMG (Shepherd Model Gateway, the published image) plus our router in front of it, and SSH.
FROM lightseekorg/smg:1.11.0

USER root
RUN apt-get update && apt-get install -y --no-install-recommends openssh-server curl less nano \
    && rm -rf /var/lib/apt/lists/* && mkdir -p /run/sshd
COPY requirements.txt /opt/model-gateway/requirements.txt
# The optional `ca` build secret is for building behind a TLS-inspecting proxy; it is not kept in the image.
RUN --mount=type=secret,id=ca,required=false \
    if [ -f /run/secrets/ca ]; then export PIP_CERT=/run/secrets/ca; fi; \
    /opt/venv/bin/pip install --no-cache-dir -r /opt/model-gateway/requirements.txt
COPY mgw /opt/model-gateway/mgw
COPY config /opt/model-gateway/config
COPY entrypoint.sh /opt/model-gateway/entrypoint.sh

ENV PYTHONPATH=/opt/model-gateway \
    GATEWAY_CONFIG=/workspace/config/gateway.yaml \
    GATEWAY_PORTS=8080,8090
EXPOSE 8080 8090 22
ENTRYPOINT ["bash", "/opt/model-gateway/entrypoint.sh"]
