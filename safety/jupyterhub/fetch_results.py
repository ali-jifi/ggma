#!/usr/bin/env python3
"""fetch_results.py — copy run outputs off the JupyterHub box and onto this machine.

WHY THIS EXISTS. The GPU box is borrowed. Anything under safety/outputs/ there is on
someone else's disk and goes away when the loan does — and `safety/outputs/` is
git-ignored (.gitignore:17), so pushing a branch does NOT carry it home either. A run
that only exists on the hub is a run you are one cleanup away from losing.

HOW. JupyterHub's single-user server exposes the standard Jupyter Contents API at
    {JUPYTERHUB_URL}/user/{JUPYTERHUB_USER}/api/contents/{path}
authenticated with the same JUPYTERHUB_TOKEN that hubctl.py already uses. No SSH needed,
which matters because most hubs hand out a browser and nothing else.

    ./run.sh fetch --list                 what run directories exist on the hub
    ./run.sh fetch --run drift-clip       pull one run into safety/outputs/
    ./run.sh fetch --all                  pull every run the hub has
    ./run.sh fetch --run drift-clip --remote-root work/ggma/safety/outputs

PATHS ARE RELATIVE TO THE NOTEBOOK SERVER'S ROOT, which is normally your home directory
on the hub — NOT to this repo. If you cloned to ~/ggma the default below is right; if you
cloned somewhere else, pass --remote-root (or set SAFETY_HUB_REMOTE_ROOT in .env).

If the hub blocks the Contents API, the fallback is a terminal on the box:
    tar czf ~/outputs.tgz -C <repo>/safety outputs
then download that single file from the Jupyter file browser.
"""

import argparse
import base64
import json
import sys
import urllib.error
import urllib.parse
import urllib.request
from pathlib import Path

HUB_DIR = Path(__file__).resolve().parent
SAFETY_DIR = HUB_DIR.parent
for _p in (str(HUB_DIR),):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import env_loader  # noqa: E402
import hubctl  # noqa: E402

TIMEOUT = 120
DEFAULT_REMOTE_ROOT = "ggma/safety/outputs"

# Written by run_local.py / run.sh. Pulled in this order so a partial fetch still leaves
# the most valuable file first: partial.jsonl and rows.json are the only ones that cannot
# be regenerated without paying for the GPU again.
_PRIORITY = ("partial.jsonl", "rows.json", "traces.jsonl", "summary.json",
             "run_meta.json", "harmbench_scored.json", "judge_report.json",
             "terminal_output.txt")


def _contents_url(remote_path):
    user = env_loader.get("JUPYTERHUB_USER", required=True)
    base = hubctl.base_url()
    quoted = urllib.parse.quote(str(remote_path).strip("/"))
    return f"{base}/user/{urllib.parse.quote(user)}/api/contents/{quoted}"


# Auth state, built once. Either a token, or a cookie session from the password form.
_AUTH = {"mode": None, "opener": None, "xsrf": ""}


def _login_with_password():
    """Cookie session for the single-user Contents API, using JUPYTERHUB_PASSWORD.

    A token is nicer, but it is optional in .env while the password is not, and a tool
    that refuses to run without the optional credential is a tool that does not run.
    Three steps, and the last two are the ones that are easy to miss:

      1. GET  /hub/login           -> the hub's _xsrf cookie
      2. POST /hub/login + _xsrf   -> JupyterHub 5.x returns a bare 403 without it,
                                      which looks exactly like a wrong password
      3. GET  /user/<u>/lab        -> completes the OAuth handshake that actually
                                      authorises /user/<u>/api/*. Skip it and every
                                      Contents call is 403 even with valid session
                                      cookies AND a correct X-XSRFToken header.
    """
    import http.cookiejar
    import re as _re

    base = hubctl.base_url()
    user = env_loader.get("JUPYTERHUB_USER", required=True)
    pw = env_loader.get("JUPYTERHUB_PASSWORD", "")
    if not pw:
        raise SystemExit(
            "neither JUPYTERHUB_TOKEN nor JUPYTERHUB_PASSWORD is set in .env — "
            "the Contents API needs one of them."
        )
    jar = http.cookiejar.CookieJar()
    opener = urllib.request.build_opener(urllib.request.HTTPCookieProcessor(jar))

    with opener.open(f"{base}/hub/login", timeout=TIMEOUT) as r:
        page = r.read().decode("utf-8", "replace")
    xsrf = next((c.value for c in jar if c.name == "_xsrf"), "")
    if not xsrf:
        m = _re.search(r'name="_xsrf"\s+value="([^"]+)"', page)
        xsrf = m.group(1) if m else ""

    data = urllib.parse.urlencode(
        {"username": user, "password": pw, "_xsrf": xsrf}).encode()
    req = urllib.request.Request(f"{base}/hub/login?next=", data=data)
    req.add_header("Content-Type", "application/x-www-form-urlencoded")
    try:
        opener.open(req, timeout=TIMEOUT).read()
    except urllib.error.HTTPError as e:
        raise SystemExit(f"hub login failed: HTTP {e.code}") from e

    quser = urllib.parse.quote(user)
    opener.open(f"{base}/user/{quser}/lab", timeout=TIMEOUT).read()   # OAuth handshake
    xsrf = next((c.value for c in jar if c.name == "_xsrf"), xsrf)
    return opener, xsrf


def _auth():
    if _AUTH["mode"] is None:
        token = env_loader.get("JUPYTERHUB_TOKEN", "")
        if token:
            _AUTH.update(mode="token", opener=urllib.request.build_opener(), xsrf="")
        else:
            opener, xsrf = _login_with_password()
            _AUTH.update(mode="password", opener=opener, xsrf=xsrf)
            print("[auth] no JUPYTERHUB_TOKEN — using the password + OAuth session")
    return _AUTH


def _get(remote_path, *, content=True):
    a = _auth()
    url = _contents_url(remote_path) + ("?content=1" if content else "?content=0")
    req = urllib.request.Request(url)
    req.add_header("Accept", "application/json")
    if a["mode"] == "token":
        req.add_header("Authorization", f"token {env_loader.get('JUPYTERHUB_TOKEN')}")
    else:
        req.add_header("X-XSRFToken", a["xsrf"])
    with a["opener"].open(req, timeout=TIMEOUT) as resp:
        return json.loads(resp.read().decode("utf-8", "replace"))


def _local_base():
    return Path(env_loader.get("SAFETY_OUTPUT_DIR", "") or (SAFETY_DIR / "outputs"))


def _write_file(node, dest: Path):
    """Materialise one Contents-API file node. `format` is 'text' or 'base64'; anything
    that is not valid UTF-8 (there should be none here, but activations would be) comes
    back base64 and must not be decoded as text."""
    dest.parent.mkdir(parents=True, exist_ok=True)
    if node.get("format") == "base64":
        dest.write_bytes(base64.b64decode(node.get("content") or ""))
    else:
        dest.write_text(node.get("content") or "")
    return dest.stat().st_size


def fetch_dir(remote_dir, local_dir: Path, *, indent=""):
    """Recursively pull one remote directory. Returns (n_files, n_bytes)."""
    node = _get(remote_dir)
    if node.get("type") != "directory":
        size = _write_file(node, local_dir)
        print(f"{indent}  {local_dir.name}  ({size:,} B)")
        return 1, size

    entries = node.get("content") or []
    # Priority files first, then everything else alphabetically. A fetch interrupted
    # halfway then still has the irreplaceable rows on disk.
    def _rank(e):
        try:
            return (_PRIORITY.index(e["name"]), e["name"])
        except ValueError:
            return (len(_PRIORITY), e["name"])

    n_files = n_bytes = 0
    for e in sorted(entries, key=_rank):
        if e["type"] == "directory":
            f, b = fetch_dir(e["path"], local_dir / e["name"], indent=indent + "  ")
        else:
            try:
                child = _get(e["path"])
            except urllib.error.HTTPError as err:
                print(f"{indent}  {e['name']}: HTTP {err.code} — skipped")
                continue
            size = _write_file(child, local_dir / e["name"])
            print(f"{indent}  {e['name']}  ({size:,} B)")
            f, b = 1, size
        n_files += f
        n_bytes += b
    return n_files, n_bytes


def cmd_list(args):
    node = _get(args.remote_root)
    if node.get("type") != "directory":
        raise SystemExit(f"{args.remote_root} is not a directory on the hub")
    runs = sorted(e["name"] for e in (node.get("content") or [])
                  if e["type"] == "directory")
    if not runs:
        print(f"no run directories under {args.remote_root} on the hub")
        return
    print(f"runs under {args.remote_root} on the hub:")
    for r in runs:
        print(f"  {r}")
    print(f"\n  ./run.sh fetch --run <name>    to pull one")
    print(f"  ./run.sh fetch --all          to pull all {len(runs)}")


def cmd_fetch(args):
    local_base = _local_base()
    if args.all:
        node = _get(args.remote_root)
        runs = sorted(e["name"] for e in (node.get("content") or [])
                      if e["type"] == "directory")
        if not runs:
            raise SystemExit(f"nothing to fetch under {args.remote_root}")
    else:
        runs = [args.run]

    total_f = total_b = 0
    for r in runs:
        remote = f"{args.remote_root.strip('/')}/{r}"
        local = local_base / r
        if local.exists() and not args.force:
            existing = sorted(p.name for p in local.iterdir())
            print(f"[skip] {r}: {local} already exists ({len(existing)} file(s)). "
                  f"Pass --force to overwrite.")
            continue
        print(f"\n[fetch] {remote} -> {local}")
        try:
            f, b = fetch_dir(remote, local)
        except urllib.error.HTTPError as err:
            if err.code == 404:
                raise SystemExit(
                    f"{remote} not found on the hub (HTTP 404).\n"
                    f"  Contents-API paths are relative to the notebook server's root "
                    f"(usually $HOME), not to this repo.\n"
                    f"  Try `./run.sh fetch --list`, or pass --remote-root."
                ) from err
            raise
        total_f += f
        total_b += b
        print(f"[fetch] {r}: {f} file(s), {b:,} B")

    if total_f:
        print(f"\nDone: {total_f} file(s), {total_b:,} B under {local_base}")
        print("These are git-ignored by design. They are now on THIS machine, which is "
              "the only copy that survives the hub loan.")


def main():
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--run", default="", help="run directory name on the hub")
    ap.add_argument("--all", action="store_true", help="fetch every run directory")
    ap.add_argument("--list", action="store_true", help="list remote runs and exit")
    ap.add_argument("--remote-root", default="",
                    help=f"hub path holding the run dirs, relative to the notebook "
                         f"server root (default: SAFETY_HUB_REMOTE_ROOT or "
                         f"{DEFAULT_REMOTE_ROOT})")
    ap.add_argument("--force", action="store_true",
                    help="overwrite a run directory that already exists locally")
    args = ap.parse_args()

    env_loader.load(verbose=False)
    args.remote_root = (args.remote_root
                        or env_loader.get("SAFETY_HUB_REMOTE_ROOT", "")
                        or DEFAULT_REMOTE_ROOT)

    if args.list:
        return cmd_list(args)
    if not args.run and not args.all:
        raise SystemExit("pass --run <name>, --all, or --list")
    return cmd_fetch(args)


if __name__ == "__main__":
    main()
