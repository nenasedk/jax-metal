"""Hold one REA MCP session open and forward tool calls to it over a Unix socket.

REA re-imports and re-analyzes a binary for every new session. For the 159 MB
jax-metal plugin that takes ~36 minutes, so this keeps a single session alive
and lets short-lived clients query it.

    python3 -I re/scripts/rea_session.py serve          # foreground server
    python3 -I re/scripts/rea_session.py call TOOL '{"arg": "value"}'
"""

import json
import os
import socket
import subprocess
import sys
import threading

REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
WORK = os.path.join(REPO, "re", "work")
SOCKET_PATH = os.path.join(WORK, "rea.sock")
LOG_PATH = os.path.join(WORK, "rea-session.log")
REA_COMMAND = ["npx", "-y", "rea-agents@6.7.0", "mcp"]

# Phase 0 profile: fits an 8 GB machine (1.8 GB peak RSS, ~36 min analysis).
LEAN_ANALYZERS = ",".join([
    "Decompiler Switch Analysis",
    "Stack",
    "Call Convention ID",
    "Non-Returning Functions - Discovered",
    "Function ID",
    "Function Start Search",
    "Function Start Search After Code",
    "Function Start Search After Data",
    "Create Address Tables",
    "Embedded Media",
    "Shared Return Calls",
    "Apply Data Archives",
    "Objective-C Message Analyzer",
])

REA_ENV = {
    "GHIDRA_INSTALL_DIR": os.path.expanduser("~/tools/ghidra_12.1.4_PUBLIC"),
    "JAVA_HOME": "/opt/homebrew/opt/openjdk@21/libexec/openjdk.jdk/Contents/Home",
    "GHIDRA_HEADLESS_MAXMEM": "3G",
    "REA_GHIDRA_DISABLED_ANALYZERS": LEAN_ANALYZERS,
    # The default 330 s startup timeout aborts before analysis of this binary finishes.
    "REA_GHIDRA_STARTUP_TIMEOUT_MS": "3600000",
}


def log(message):
    with open(LOG_PATH, "a") as handle:
        handle.write(message + "\n")


class ReaServer:
    def __init__(self):
        env = dict(os.environ, **REA_ENV)
        self.process = subprocess.Popen(
            REA_COMMAND, stdin=subprocess.PIPE, stdout=subprocess.PIPE,
            stderr=open(os.path.join(WORK, "rea-mcp.stderr"), "a"), text=True, env=env)
        self.lock = threading.Lock()
        self.next_id = 1

    def _send(self, message):
        self.process.stdin.write(json.dumps(message) + "\n")
        self.process.stdin.flush()

    def request(self, method, params):
        with self.lock:
            request_id = self.next_id
            self.next_id += 1
            self._send({"jsonrpc": "2.0", "id": request_id, "method": method, "params": params})
            while True:
                line = self.process.stdout.readline()
                if not line:
                    raise RuntimeError("REA MCP server exited")
                message = json.loads(line)
                if message.get("id") == request_id and "method" not in message:
                    return message
                if "method" in message and "id" in message:
                    # Server-to-client request (roots, sampling, ...): decline politely.
                    result = {"roots": []} if message["method"] == "roots/list" else None
                    reply = {"jsonrpc": "2.0", "id": message["id"]}
                    if result is None:
                        reply["error"] = {"code": -32601, "message": "not supported by rea_session"}
                    else:
                        reply["result"] = result
                    self._send(reply)
                elif "method" in message:
                    log("notification: " + line.strip()[:500])

    def initialize(self):
        response = self.request("initialize", {
            "protocolVersion": "2025-06-18",
            "capabilities": {},
            "clientInfo": {"name": "rea_session", "version": "1"},
        })
        self._send({"jsonrpc": "2.0", "method": "notifications/initialized"})
        return response


def serve():
    os.makedirs(WORK, exist_ok=True)
    if os.path.exists(SOCKET_PATH):
        os.unlink(SOCKET_PATH)
    server = ReaServer()
    log("initialize: " + json.dumps(server.initialize().get("result", {}).get("serverInfo")))
    listener = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
    listener.bind(SOCKET_PATH)
    listener.listen()
    print("ready", flush=True)
    while True:
        connection, _ = listener.accept()
        with connection:
            data = b""
            while not data.endswith(b"\n"):
                chunk = connection.recv(65536)
                if not chunk:
                    break
                data += chunk
            call = json.loads(data)
            log("call: " + json.dumps(call)[:500])
            try:
                response = server.request("tools/call", {"name": call["tool"], "arguments": call["args"]})
            except Exception as error:  # report, keep serving
                response = {"error": {"message": str(error)}}
            log("done: " + call["tool"])
            connection.sendall(json.dumps(response).encode() + b"\n")


def call(tool, args_json):
    client = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
    client.connect(SOCKET_PATH)
    client.sendall(json.dumps({"tool": tool, "args": json.loads(args_json)}).encode() + b"\n")
    data = b""
    while not data.endswith(b"\n"):
        chunk = client.recv(1 << 20)
        if not chunk:
            break
        data += chunk
    response = json.loads(data)
    if "error" in response:
        print(json.dumps(response["error"], indent=1))
        sys.exit(1)
    result = response["result"]
    for item in result.get("content", []):
        if item.get("type") == "text":
            print(item["text"])
    if result.get("isError"):
        sys.exit(1)


if __name__ == "__main__":
    if sys.argv[1:2] == ["serve"]:
        serve()
    elif sys.argv[1:2] == ["call"] and len(sys.argv) in (3, 4):
        call(sys.argv[2], sys.argv[3] if len(sys.argv) == 4 else "{}")
    else:
        sys.exit(__doc__)
