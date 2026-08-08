#!/usr/bin/env python3
"""hub_exec.py — drive the JupyterHub box from your laptop.

`hubctl.py` can start and stop your server; `fetch_results.py` can pull artifacts
back. This is the missing middle: running a command *on* the box without opening a
browser terminal, so a long collection run can be launched, polled and recovered
from a script.

It authenticates the same way `hubctl.py` does (`JUPYTERHUB_TOKEN`, else
`JUPYTERHUB_PASSWORD` posted to the login form), spawns the single-user server if it
is not already up, then executes through a short-lived Python kernel on the box.

    ./run.sh exec -- nvidia-smi
    ./run.sh exec -- 'cd ~/ggma && git log --oneline -3'
    ./run.sh exec --status
    ./run.sh exec --put local.sh work/remote.sh
    ./run.sh exec --launch 'bash run_arm.sh' --log work/arm.log

`--launch` is the one that matters for a long run: the command is detached with
`setsid nohup`, so it outlives both the kernel and this process, and its console
goes to a file you can poll with `--exec 'tail -n 40 <log>'`.

Design notes
------------
- **The kernel is a remote control, not the job.** Anything that takes hours must go
  through `--launch`; a kernel dies with the websocket, and the websocket dies with
  your laptop's wifi.
- **Idle culling is the real hazard.** Most hubs stop a single-user server that has
  had no kernel activity for N minutes, and stopping the server usually kills
  everything the container was running. `--keepalive` holds a kernel open and pokes
  it, which is what registers as activity.
- Stdlib plus `websocket-client` (already in `requirements-hub.txt` via jupyter).
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
import time
import urllib.error
import urllib.parse
import urllib.request
import uuid
from http.cookiejar import CookieJar
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import env_loader  # noqa: E402

TIMEOUT = 60
SPAWN_WAIT = 300


class Hub:
    """An authenticated session against the hub and your single-user server."""

    def __init__(self, verbose=True):
        env_loader.load(verbose=verbose)
        self.url = self._base_url()
        self.user = env_loader.get("JUPYTERHUB_USER", required=True)
        self.token = env_loader.get("JUPYTERHUB_TOKEN", "")
        self.password = env_loader.get("JUPYTERHUB_PASSWORD", "")
        self.jar = CookieJar()
        self.opener = urllib.request.build_opener(
            urllib.request.HTTPCookieProcessor(self.jar))
        self.verbose = verbose
        self._authed = False

    # --- plumbing -------------------------------------------------------------
    @staticmethod
    def _base_url():
        url = env_loader.get("JUPYTERHUB_URL", required=True).strip().rstrip("/")
        if not url.startswith(("http://", "https://")):
            url = "https://" + url
        for suffix in ("/hub/login", "/hub/home", "/hub/spawn", "/hub", "/lab", "/tree"):
            if url.endswith(suffix):
                url = url[: -len(suffix)]
        return url.rstrip("/")

    def _say(self, msg):
        if self.verbose:
            print(msg, flush=True)

    def _cookie(self, name, prefix=False):
        for c in self.jar:
            if (c.name.startswith(name) if prefix else c.name == name):
                return c.value
        return ""

    def _req(self, url, *, data=None, headers=None, method=None, raw=False):
        req = urllib.request.Request(url, data=data, method=method)
        for k, v in (headers or {}).items():
            req.add_header(k, v)
        if self.token:
            req.add_header("Authorization", f"token {self.token}")
        with self.opener.open(req, timeout=TIMEOUT) as resp:
            body = resp.read()
            return resp.status, (body if raw else body.decode("utf-8", "replace"))

    # --- auth -----------------------------------------------------------------
    def login(self):
        """Token if we have one, otherwise the login form. Idempotent."""
        if self._authed:
            return
        if self.token:
            status, body = self._req(f"{self.url}/hub/api/user")
            me = json.loads(body)
            self._say(f"[auth] token accepted — {me.get('name')!r}")
            self._authed = True
            return
        if not self.password:
            raise SystemExit("Neither JUPYTERHUB_TOKEN nor JUPYTERHUB_PASSWORD is set.")

        # JupyterHub 5.x rejects a login POST that does not echo the _xsrf back, with a
        # bare 403 that reads exactly like a wrong password. GET the page first.
        _, page = self._req(f"{self.url}/hub/login")
        xsrf = self._cookie("_xsrf")
        if not xsrf:
            m = re.search(r'name="_xsrf"\s+value="([^"]+)"', page)
            xsrf = m.group(1) if m else ""
        data = urllib.parse.urlencode({
            "username": self.user, "password": self.password, "_xsrf": xsrf,
        }).encode()
        self._req(f"{self.url}/hub/login?next=", data=data, headers={
            "Content-Type": "application/x-www-form-urlencoded", "X-Xsrftoken": xsrf,
        })
        if not self._cookie("jupyterhub-session-id", prefix=True):
            raise SystemExit(f"the hub did not issue a session cookie for {self.user!r}.")
        self._say(f"[auth] password accepted — {self.user!r}")
        self._authed = True

    # --- the single-user server ----------------------------------------------
    def server_up(self):
        try:
            status, body = self._req(f"{self.url}/user/{self.user}/api/status")
            return status == 200 and body.strip().startswith("{")
        except urllib.error.HTTPError:
            return False
        except Exception:                                        # noqa: BLE001
            return False

    def ensure_server(self):
        self.login()
        if self.server_up():
            self._say("[server] already running")
            return
        self._say("[server] not running — spawning ...")
        # /hub/spawn/<user> starts with default options. A hub with a spawner options
        # form serves the form instead of spawning; that needs a human, so say so.
        try:
            _, body = self._req(f"{self.url}/hub/spawn/{self.user}")
            if "<form" in body.lower() and "spawn" in body.lower() \
                    and "profile" in body.lower():
                raise SystemExit(
                    f"this hub shows a server-options form. Start it once in a browser:"
                    f"\n  {self.url}/hub/spawn/{self.user}")
        except urllib.error.HTTPError as e:
            if e.code not in (302, 303):
                self._say(f"[server] spawn returned HTTP {e.code}; polling anyway")
        deadline = time.time() + SPAWN_WAIT
        while time.time() < deadline:
            if self.server_up():
                self._say("[server] up")
                return
            time.sleep(5)
        raise SystemExit(f"server did not come up within {SPAWN_WAIT}s.")

    def _server_xsrf(self):
        """The single-user server sets its own _xsrf, scoped to /user/<name>."""
        self._req(f"{self.url}/user/{self.user}/tree")
        for c in self.jar:
            if c.name == "_xsrf" and f"/user/{self.user}" in (c.path or ""):
                return c.value
        return self._cookie("_xsrf")

    # --- contents API ---------------------------------------------------------
    def put(self, local: Path, remote: str):
        self.ensure_server()
        xsrf = self._server_xsrf()
        raw = Path(local).read_bytes()
        try:
            text = raw.decode()
            payload = {"type": "file", "format": "text", "content": text}
        except UnicodeDecodeError:
            import base64
            payload = {"type": "file", "format": "base64",
                       "content": base64.b64encode(raw).decode()}
        url = f"{self.url}/user/{self.user}/api/contents/{remote.lstrip('/')}"
        self._req(url, data=json.dumps(payload).encode(), method="PUT", headers={
            "Content-Type": "application/json", "X-XSRFToken": xsrf,
        })
        self._say(f"[put] {local} -> {remote} ({len(raw):,} B)")

    def get(self, remote: str, local: Path):
        self.ensure_server()
        url = (f"{self.url}/user/{self.user}/api/contents/"
               f"{remote.lstrip('/')}?content=1")
        _, body = self._req(url)
        node = json.loads(body)
        if node.get("format") == "base64":
            import base64
            data = base64.b64decode(node["content"])
        else:
            data = str(node.get("content", "")).encode()
        Path(local).parent.mkdir(parents=True, exist_ok=True)
        Path(local).write_bytes(data)
        self._say(f"[get] {remote} -> {local} ({len(data):,} B)")

    # --- execution ------------------------------------------------------------
    def _start_kernel(self):
        xsrf = self._server_xsrf()
        url = f"{self.url}/user/{self.user}/api/kernels"
        _, body = self._req(url, data=b"{}", method="POST", headers={
            "Content-Type": "application/json", "X-XSRFToken": xsrf,
        })
        return json.loads(body)["id"], xsrf

    def _kill_kernel(self, kid, xsrf):
        try:
            self._req(f"{self.url}/user/{self.user}/api/kernels/{kid}",
                      method="DELETE", headers={"X-XSRFToken": xsrf})
        except Exception:                                        # noqa: BLE001
            pass

    def run_code(self, code, *, timeout=600, stream=True):
        """Execute Python on the box. Returns (stdout_text, error_text)."""
        import websocket                                          # noqa: PLC0415

        self.ensure_server()
        kid, xsrf = self._start_kernel()
        ws_url = (self.url.replace("https://", "wss://").replace("http://", "ws://")
                  + f"/user/{self.user}/api/kernels/{kid}/channels"
                  + f"?session_id={uuid.uuid4().hex}")
        cookie = "; ".join(f"{c.name}={c.value}" for c in self.jar)
        header = [f"Authorization: token {self.token}"] if self.token else []
        out, err = [], []
        try:
            ws = websocket.create_connection(
                ws_url, cookie=cookie, header=header, timeout=timeout,
                origin=self.url, suppress_origin=False)
            msg_id = uuid.uuid4().hex
            ws.send(json.dumps({
                "header": {"msg_id": msg_id, "username": self.user,
                           "session": uuid.uuid4().hex, "msg_type": "execute_request",
                           "version": "5.3"},
                "parent_header": {}, "metadata": {},
                "content": {"code": code, "silent": False, "store_history": False,
                            "user_expressions": {}, "allow_stdin": False,
                            "stop_on_error": True},
                "channel": "shell",
            }))
            deadline = time.time() + timeout
            while time.time() < deadline:
                ws.settimeout(max(1, deadline - time.time()))
                try:
                    msg = json.loads(ws.recv())
                except websocket.WebSocketTimeoutException:
                    break
                if msg.get("parent_header", {}).get("msg_id") != msg_id:
                    continue
                mt = msg.get("msg_type")
                c = msg.get("content", {})
                if mt == "stream":
                    out.append(c.get("text", ""))
                    if stream:
                        sys.stdout.write(c.get("text", "")); sys.stdout.flush()
                elif mt in ("execute_result", "display_data"):
                    t = c.get("data", {}).get("text/plain", "")
                    out.append(t + "\n")
                    if stream:
                        print(t, flush=True)
                elif mt == "error":
                    e = "\n".join(c.get("traceback", []))
                    err.append(e)
                    if stream:
                        print(e, file=sys.stderr, flush=True)
                elif mt == "status" and c.get("execution_state") == "idle":
                    break
            ws.close()
        finally:
            self._kill_kernel(kid, xsrf)
        return "".join(out), "\n".join(err)

    def run_shell(self, cmd, *, timeout=600, stream=True):
        """Run a shell command on the box, streaming its output."""
        code = (
            "import subprocess, sys\n"
            f"p = subprocess.run({cmd!r}, shell=True, executable='/bin/bash',\n"
            "                   stdout=subprocess.PIPE, stderr=subprocess.STDOUT,\n"
            "                   text=True, errors='replace')\n"
            "sys.stdout.write(p.stdout)\n"
            "print(f'__RC__={p.returncode}')\n"
        )
        out, err = self.run_code(code, timeout=timeout, stream=stream)
        m = re.search(r"__RC__=(-?\d+)", out)
        return (int(m.group(1)) if m else -1), out, err

    def launch(self, cmd, log, *, cwd=""):
        """Start `cmd` detached so it outlives the kernel, the server and this process."""
        pidfile = f"{log}.pid"
        # Newline-separated, not `&&`-joined: bash parses `a && b & c` as `(a && b) & c`,
        # so an `&&` before the backgrounded command drags the mkdir into the background
        # too and the pidfile write races the directory that is meant to hold it.
        full = "\n".join(filter(None, [
            f"cd {cwd}" if cwd else "",
            f"mkdir -p $(dirname {log})",
            f"setsid nohup bash -lc {json.dumps(cmd)} > {log} 2>&1 < /dev/null &",
            f"echo $! > {pidfile}",
            "sleep 2",
            f"cat {pidfile}",
        ]))
        rc, out, _ = self.run_shell(full, timeout=120)
        self._say(f"[launch] rc={rc} log={log}")
        return out.strip()


def _status(hub: Hub):
    hub.ensure_server()
    probe = r"""
set -e
echo "=== host ==="; hostname; uname -sr; date -u
echo; echo "=== gpus ==="
nvidia-smi --query-gpu=index,name,memory.total,memory.used,utilization.gpu \
  --format=csv 2>&1 || echo "no nvidia-smi"
echo; echo "=== cpu/mem ==="; nproc; free -g | head -2
echo; echo "=== disk ==="; df -h / /workspace $HOME 2>/dev/null | sort -u
echo; echo "=== repo ==="
for d in ~/ggma /workspace/ggma ./ggma; do
  [ -d "$d/.git" ] && { echo "found $d"; git -C "$d" log --oneline -1; \
    git -C "$d" rev-parse --abbrev-ref HEAD; }
done
echo; echo "=== python ==="; which python3; python3 -V
"""
    rc, _, _ = hub.run_shell(probe, timeout=180)
    return 0 if rc == 0 else rc


def main():
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--status", action="store_true", help="host, GPUs, disk, repo")
    ap.add_argument("--exec", dest="cmd", default="", help="shell command to run")
    ap.add_argument("--code", default="", help="python code to run in a kernel")
    ap.add_argument("--put", nargs=2, metavar=("LOCAL", "REMOTE"))
    ap.add_argument("--get", nargs=2, metavar=("REMOTE", "LOCAL"))
    ap.add_argument("--launch", default="", help="detached command (survives the kernel)")
    ap.add_argument("--log", default="", help="log file for --launch")
    ap.add_argument("--cwd", default="", help="working directory for --launch")
    ap.add_argument("--timeout", type=int, default=600)
    ap.add_argument("--keepalive", type=int, default=0,
                    help="hold a kernel open for N minutes, poking it, so idle "
                         "culling does not stop the server under a running job")
    ap.add_argument("--quiet", action="store_true")
    args = ap.parse_args()

    hub = Hub(verbose=not args.quiet)

    if args.put:
        hub.put(Path(args.put[0]), args.put[1]); return 0
    if args.get:
        hub.get(args.get[0], Path(args.get[1])); return 0
    if args.launch:
        if not args.log:
            raise SystemExit("--launch needs --log")
        print(hub.launch(args.launch, args.log, cwd=args.cwd)); return 0
    if args.status:
        return _status(hub)
    if args.code:
        _, err = hub.run_code(args.code, timeout=args.timeout)
        return 1 if err else 0
    if args.cmd:
        rc, _, _ = hub.run_shell(args.cmd, timeout=args.timeout)
        return rc
    if args.keepalive:
        deadline = time.time() + args.keepalive * 60
        while time.time() < deadline:
            hub.run_shell("date -u", timeout=60, stream=False)
            time.sleep(120)
        return 0
    ap.print_help()
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
