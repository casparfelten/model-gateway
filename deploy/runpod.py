"""Runpod templates and pods for the GPU nodes and the gateway (Runpod REST API v2).

    python deploy/runpod.py secret <name> <ENV_VAR>   # store an env var's value as a Runpod secret
    python deploy/runpod.py template node|gateway     # create or update the template
    python deploy/runpod.py volume <name> <dc> <GB>   # a network volume (the gateway keeps /workspace on one)
    python deploy/runpod.py pod node <name> [gpus]    # a GPU node (default 4 GPUs)
    python deploy/runpod.py pod gateway <name> [volume-id]
    python deploy/runpod.py show <pod-id>             # the pod's public IP and port mapping
    python deploy/runpod.py terminate <pod-id>        # delete the pod (a node's own disk goes with it)

Needs RUNPOD_API_KEY. Templates are matched by name, so running `template` again updates the same one. A pod copies its
template once: editing a template later does not change running pods.
"""
import json
import os
import sys
from pathlib import Path

import httpx

API = "https://api.runpod.io/v2"
ROOT = Path(__file__).resolve().parent.parent

NODE_IMAGE = "vllm/vllm-openai:nightly-ac9126e58aa7bbab1856ba6593ba4d5003fea516"  # the first node's vLLM build
GATEWAY_IMAGE = os.environ.get("GATEWAY_IMAGE", "412341941636.dkr.ecr.us-east-1.amazonaws.com/model-gateway:20261004-0209")
GATEWAY_DC = "EU-RO-1"   # on 2026-10-04 only 2-vCPU CPU pods could be placed, here and in EUR-IS-1


def script_cmd(path: Path) -> dict:
    """A start command that writes the script to a file and execs it, so the running process is `bash /start.sh`
    (not a `bash -c` whose command line holds the whole script)."""
    return {"entrypoint": ["bash", "-c"],
            "cmd": [f"cat > /start.sh <<'START_EOF'\n{path.read_text()}\nSTART_EOF\nexec bash /start.sh"]}


def bench_cmd() -> dict:
    script = (ROOT / "bench" / "pod_start.sh").read_text().replace("__CTL_PY__", (ROOT / "bench" / "ctl.py").read_text())
    return {"entrypoint": ["bash", "-c"],
            "cmd": [f"cat > /start.sh <<'START_EOF'\n{script}\nSTART_EOF\nexec bash /start.sh"]}


TEMPLATES = {
    "node": {
        "name": "qwen3.8-27b-node",
        "image": NODE_IMAGE,
        **script_cmd(ROOT / "node" / "start.sh"),
        "ports": ["8000/tcp", "8001/tcp", "8002/tcp", "22/tcp"],
        "disk": 50,
        "mounts": {"persistent": {"size": 300, "path": "/workspace"}},
        "env": {
            "VLLM_API_KEY": "{{ RUNPOD_SECRET_vllm_api_key }}",
            "GATEWAY_POOL": "qwen3.8-27b",  # the gateway adds pods with this env var to its GPU pool
            "PYTHONHASHSEED": "0",
            "VLLM_ENGINE_READY_TIMEOUT_S": "3600",
            "VLLM_CACHE_ROOT": "/workspace/.vllm_cache",
            "HF_XET_HIGH_PERFORMANCE": "1",
        },
        "startSsh": True,
        "startJupyter": False,
    },
    "bench": {   # benchmark pods: vLLM image, nothing started; driven through bench/ctl.py on 8888 (HTTPS proxy)
        "name": "qwen-bench-node",
        "image": NODE_IMAGE,
        **bench_cmd(),
        "ports": ["8000/tcp", "8888/http", "22/tcp"],
        "disk": 50,
        "mounts": {"persistent": {"size": 600, "path": "/workspace"}},
        "env": {
            "VLLM_API_KEY": "{{ RUNPOD_SECRET_vllm_api_key }}",
            "CTL_TOKEN": "{{ RUNPOD_SECRET_ctl_token }}",
            "HF_XET_HIGH_PERFORMANCE": "1",
            "VLLM_CACHE_ROOT": "/workspace/.vllm_cache",
        },
        "startSsh": True,
        "startJupyter": False,
    },
    "gateway": {
        "name": "model-gateway",
        "category": "CPU",
        "image": GATEWAY_IMAGE,
        "ports": ["8080/tcp", "8090/http", "8081/tcp", "22/tcp"],
        "disk": 20,
        "env": {
            "GATEWAY_API_KEY": "{{ RUNPOD_SECRET_gateway_api_key }}",
            "VLLM_API_KEY": "{{ RUNPOD_SECRET_vllm_api_key }}",
            "GATEWAY_RUNPOD_API_KEY": "{{ RUNPOD_SECRET_rp_api_key }}",
            "TOUCHMARK_API_KEY": "{{ RUNPOD_SECRET_touchmark_api_key }}",
            "AI_GATEWAY_API_KEY": "{{ RUNPOD_SECRET_ai_gateway_api_key }}",
        },
        "startSsh": True,
        "startJupyter": False,
    },
}

PODS = {
    # the vLLM image needs a CUDA 13 host driver: on an older one multi-GPU communication fails
    "node": {"gpu": {"id": "NVIDIA H200", "count": 4, "minCudaVersion": "13.0"}, "cloud": "SECURE"},
    "bench": {"gpu": {"id": "NVIDIA H200", "count": 4, "minCudaVersion": "13.0"}, "cloud": "SECURE"},
    "gateway": {"cpu": {"id": "cpu5c", "vcpuCount": 2}, "cloud": "SECURE", "dataCenterIds": [GATEWAY_DC]},
}


def client() -> httpx.Client:
    return httpx.Client(base_url=API, timeout=60, headers={"Authorization": f"Bearer {os.environ['RUNPOD_API_KEY']}"})


def check(r: httpx.Response) -> dict:
    if r.status_code >= 400:
        sys.exit(f"{r.request.method} {r.request.url.path}: {r.status_code} {r.text}")
    return r.json() if r.content else {}


def items(body, key: str) -> list:
    return body.get(key, []) if isinstance(body, dict) else body


def secret(name: str, env_var: str) -> None:
    with client() as c:
        existing = {s["name"]: s for s in items(check(c.get("/account/secrets")), "secrets")}
        if name in existing:
            check(c.patch(f"/account/secrets/{existing[name]['id']}", json={"value": os.environ[env_var]}))
            print(f"updated secret {name}")
        else:
            check(c.post("/account/secrets", json={"name": name, "value": os.environ[env_var]}))
            print(f"created secret {name}")


def template(kind: str) -> str:
    spec = TEMPLATES[kind]
    with client() as c:
        existing = {t["name"]: t for t in items(check(c.get("/templates")), "templates")}
        if spec["name"] in existing:
            tid = existing[spec["name"]]["id"]
            check(c.patch(f"/templates/{tid}", json={k: v for k, v in spec.items() if k not in ("name", "category")}))
            print(f"updated template {spec['name']} ({tid})")
        else:
            tid = check(c.post("/templates", json=spec))["id"]
            print(f"created template {spec['name']} ({tid})")
        return tid


def volume(name: str, dc: str, gb: str) -> None:
    with client() as c:
        v = check(c.post("/network-volumes", json={"name": name, "dataCenter": dc, "size": int(gb)}))
    print(json.dumps(v))


def pod(kind: str, name: str, extra: str = "") -> None:
    with client() as c:
        templates = {t["name"]: t["id"] for t in items(check(c.get("/templates")), "templates")}
        body = {"name": name, "templateId": templates[TEMPLATES[kind]["name"]], **PODS[kind]}
        if kind in ("node", "bench") and extra:
            body["gpu"] = {**body["gpu"], "count": int(extra)}
        if kind == "gateway" and extra:   # a network volume keeps the edited config across pod restarts
            body["mounts"] = {"network": [{"volumeId": extra, "path": "/workspace"}]}
        created = check(c.post("/pods", json=body))
        print(json.dumps({k: created.get(k) for k in ("id", "name", "status", "desiredStatus", "cost")}))


def terminate(pod_id: str) -> None:
    with client() as c:
        check(c.delete(f"/pods/{pod_id}"))
    print(f"terminated {pod_id}")


def show(pod_id: str) -> None:
    with client() as c:
        p = check(c.get(f"/pods/{pod_id}"))
    print(json.dumps({k: p.get(k) for k in ("id", "name", "desiredStatus", "publicIp", "runtime", "ports")}, indent=1))


if __name__ == "__main__":
    cmd, *args = sys.argv[1:] or ["help"]
    {"secret": secret, "template": template, "volume": volume, "pod": pod, "show": show,
     "terminate": terminate}.get(cmd, lambda *a: print(__doc__))(*args)
