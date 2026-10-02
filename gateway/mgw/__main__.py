"""Run the gateway: python -m mgw [config.yaml]. Ports: env GATEWAY_PORTS (default 8080,8090), all serving the same."""
import asyncio
import logging
import os
import sys

import uvicorn

from .app import build


async def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s %(message)s")
    logging.getLogger("httpx").setLevel(logging.WARNING)
    config = sys.argv[1] if len(sys.argv) > 1 else os.environ.get("GATEWAY_CONFIG", "/workspace/config/gateway.yaml")
    app = build(config)
    ports = [int(p) for p in os.environ.get("GATEWAY_PORTS", "8080,8090").split(",")]
    servers = [uvicorn.Server(uvicorn.Config(app, host="0.0.0.0", port=p, lifespan="on" if i == 0 else "off",
                                             timeout_keep_alive=75, log_level="warning"))
               for i, p in enumerate(ports)]
    await asyncio.gather(*(s.serve() for s in servers))


if __name__ == "__main__":
    asyncio.run(main())
