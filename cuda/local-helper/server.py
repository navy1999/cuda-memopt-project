#!/usr/bin/env python3
"""
Local CUDA helper: HTTP server that runs build/matmul_* executables.
Usage: from project root, run:  python local-helper/server.py
"""
import json
import os
import subprocess
import sys
from http.server import HTTPServer, BaseHTTPRequestHandler
from urllib.parse import parse_qs

# Project root = parent of local-helper
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
BUILD = os.path.join(ROOT, "build")
PORT = 9000

VARIANTS = ["naive", "tiled", "vec", "unroll", "tuned"]


def exe_path(variant: str) -> str:
    p = os.path.join(BUILD, f"matmul_{variant}")
    if sys.platform == "win32":
        p += ".exe"
    return p


def run_benchmark(variant: str, N: int) -> dict:
    if variant not in VARIANTS:
        return {"time_ms": 0, "validation_ok": False, "error": f"Unknown variant: {variant}"}
    exe = exe_path(variant)
    if not os.path.isfile(exe):
        return {"time_ms": 0, "validation_ok": False, "error": f"Executable not found: {exe}"}
    try:
        out = subprocess.run(
            [exe, str(N)],
            capture_output=True,
            text=True,
            timeout=60,
            cwd=ROOT,
        )
    except subprocess.TimeoutExpired:
        return {"time_ms": 0, "validation_ok": False, "error": "Timeout"}
    except Exception as e:
        return {"time_ms": 0, "validation_ok": False, "error": str(e)}

    time_ms = None
    for line in (out.stdout or "").splitlines():
        if "AvgTime=" in line:
            part = line.split("AvgTime=")[1].split()[0]
            time_ms = float(part)
            break
    if time_ms is None:
        return {"time_ms": 0, "validation_ok": False, "error": "Could not parse output"}

    validation_ok = "Validation C[0]=" in (out.stdout or "")
    return {"time_ms": time_ms, "validation_ok": validation_ok, "error": None}


class Handler(BaseHTTPRequestHandler):
    def do_OPTIONS(self):
        self.send_response(204)
        self._cors()
        self.end_headers()

    def _cors(self):
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Access-Control-Allow-Methods", "GET, POST, OPTIONS")
        self.send_header("Access-Control-Allow-Headers", "Content-Type")

    def do_GET(self):
        if self.path == "/health" or self.path == "/health/":
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self._cors()
            self.end_headers()
            self.wfile.write(b'{"ok":true}')
            return
        self.send_response(404)
        self.end_headers()

    def do_POST(self):
        if self.path == "/run" or self.path == "/run/":
            try:
                length = int(self.headers.get("Content-Length", 0))
                body = self.rfile.read(length).decode("utf-8")
                data = json.loads(body)
                variant = data.get("variant", "naive")
                N = int(data.get("N", 512))
            except (ValueError, KeyError, json.JSONDecodeError) as e:
                self.send_response(400)
                self.send_header("Content-Type", "application/json")
                self._cors()
                self.end_headers()
                self.wfile.write(json.dumps({"error": str(e)}).encode())
                return
            result = run_benchmark(variant, N)
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self._cors()
            self.end_headers()
            self.wfile.write(json.dumps(result).encode())
            return
        self.send_response(404)
        self.end_headers()

    def log_message(self, format, *args):
        print("[%s] %s" % (self.log_date_time_string(), format % args))


def main():
    os.chdir(ROOT)
    server = HTTPServer(("127.0.0.1", PORT), Handler)
    print("Local CUDA helper: http://127.0.0.1:%s (GET /health, POST /run)" % PORT)
    server.serve_forever()


if __name__ == "__main__":
    main()
