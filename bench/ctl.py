"""A command endpoint for benchmark pods, reached through Runpod's HTTPS proxy (we cannot SSH from where the benchmarks
are driven). Only for test pods: anyone with the token can run commands as root.

    POST /run   {"cmd": "...", "background": false, "timeout": 80}   header X-Token: $CTL_TOKEN
        foreground: returns {"code", "out"} (keep under the proxy's 100 s limit)
        background: starts it with nohup, output to /workspace/jobs/<id>.log; returns {"id", "log"}
    GET  /tail?path=/workspace/jobs/<id>.log&bytes=20000   the end of a file
    GET  /health
"""
import json
import os
import subprocess
import time
import uuid
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from urllib.parse import parse_qs, urlparse

TOKEN = os.environ["CTL_TOKEN"]
JOBS = "/workspace/jobs"


class Handler(BaseHTTPRequestHandler):
    def reply(self, code, body):
        data = json.dumps(body).encode()
        self.send_response(code)
        self.send_header("content-type", "application/json")
        self.send_header("content-length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def authorized(self):
        if self.headers.get("x-token") == TOKEN:
            return True
        self.reply(401, {"error": "bad token"})
        return False

    def do_GET(self):
        url = urlparse(self.path)
        if url.path == "/health":
            return self.reply(200, {"ok": True, "time": time.time()})
        if not self.authorized():
            return
        if url.path == "/tail":
            q = parse_qs(url.query)
            path, n = q["path"][0], int(q.get("bytes", ["20000"])[0])
            try:
                with open(path, "rb") as f:
                    f.seek(0, 2)
                    size = f.tell()
                    f.seek(max(0, size - n))
                    return self.reply(200, {"size": size, "text": f.read().decode(errors="replace")})
            except OSError as ex:
                return self.reply(404, {"error": str(ex)})
        self.reply(404, {"error": "not found"})

    def do_POST(self):
        if not self.authorized():
            return
        body = json.loads(self.rfile.read(int(self.headers.get("content-length", 0))) or b"{}")
        if urlparse(self.path).path != "/run":
            return self.reply(404, {"error": "not found"})
        cmd = body["cmd"]
        if body.get("background"):
            os.makedirs(JOBS, exist_ok=True)
            job = uuid.uuid4().hex[:8]
            log = f"{JOBS}/{job}.log"
            p = subprocess.Popen(["bash", "-c", cmd], stdout=open(log, "wb"), stderr=subprocess.STDOUT,
                                 start_new_session=True)
            return self.reply(200, {"id": job, "pid": p.pid, "log": log})
        try:
            r = subprocess.run(["bash", "-c", cmd], capture_output=True, timeout=body.get("timeout", 80))
            out = (r.stdout + r.stderr).decode(errors="replace")
            self.reply(200, {"code": r.returncode, "out": out[-60000:]})
        except subprocess.TimeoutExpired as ex:
            self.reply(200, {"code": None, "out": f"timed out after {ex.timeout} s (use background)"})

    def log_message(self, *a):
        pass


if __name__ == "__main__":
    ThreadingHTTPServer(("0.0.0.0", int(os.environ.get("CTL_PORT", "8888"))), Handler).serve_forever()
