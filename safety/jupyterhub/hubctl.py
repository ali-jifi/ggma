#!/usr/bin/env python3
"""hubctl.py — talk to the JupyterHub with the credentials in .env. Stdlib only.

This is what makes JUPYTERHUB_URL / _USER / _PASSWORD in .env load-bearing rather
than a note-to-self. The moment the research program's email arrives, paste the three
values and run:

    ./run.sh hub-check

and you find out immediately whether they work, instead of finding out in a browser
five minutes before you wanted to start a run.

    ./run.sh hub check          URL reachable, hub version, credentials accepted
    ./run.sh hub servers        which of your servers are running
    ./run.sh hub start          start your single-user server
    ./run.sh hub stop           stop it (frees the GPUs for the group)
    ./run.sh hub token          instructions for minting an API token

The GPU work does NOT go through here — it runs INSIDE the hub, where these
credentials are not needed. This is only for driving the hub from outside.

TWO AUTH PATHS, and the difference matters:

  * JUPYTERHUB_TOKEN — an API token from the hub UI (File -> Hub Control Panel ->
    Token). This is the only path that can drive the REST API: start/stop/servers all
    require it. Preferred once you have one.
  * JUPYTERHUB_PASSWORD — posted to the hub's login form. This proves the credentials
    are correct, which is what `check` is for, and nothing more. It CANNOT work at all
    if your hub authenticates through SSO/OAuth (Google, GitHub, institutional login),
    because there is no local password to post. That is not a bug in this script and
    `check` says so rather than reporting a mysterious failure.
"""

import argparse
import json
import os
import re
import sys
import urllib.error
import urllib.parse
import urllib.request
from http.cookiejar import CookieJar
from pathlib import Path

HUB_DIR = Path(__file__).resolve().parent
if str(HUB_DIR) not in sys.path:
    sys.path.insert(0, str(HUB_DIR))

import env_loader  # noqa: E402

TIMEOUT = 20


def base_url():
    url = env_loader.get("JUPYTERHUB_URL", required=True).strip().rstrip("/")
    if not url.startswith(("http://", "https://")):
        url = "https://" + url
    # People paste the address bar, which is usually a page under the hub, not the
    # hub root. Trim the known suffixes rather than 404 on every call.
    for suffix in ("/hub/login", "/hub/home", "/hub/spawn", "/hub", "/lab", "/tree"):
        if url.endswith(suffix):
            url = url[: -len(suffix)]
    return url.rstrip("/")


def _request(url, *, data=None, headers=None, method=None, opener=None):
    req = urllib.request.Request(url, data=data, method=method)
    for k, v in (headers or {}).items():
        req.add_header(k, v)
    op = opener or urllib.request.build_opener()
    with op.open(req, timeout=TIMEOUT) as resp:
        body = resp.read().decode("utf-8", "replace")
        return resp.status, body, dict(resp.headers)


def _api(path, *, token=None, data=None, method=None):
    url = f"{base_url()}/hub/api{path}"
    headers = {"Accept": "application/json"}
    if token:
        headers["Authorization"] = f"token {token}"
    if data is not None:
        data = json.dumps(data).encode()
        headers["Content-Type"] = "application/json"
    return _request(url, data=data, headers=headers, method=method)


def _token():
    return env_loader.get("JUPYTERHUB_TOKEN", "")


def _user():
    return env_loader.get("JUPYTERHUB_USER", "")


# --- commands -----------------------------------------------------------------
def cmd_check(args):
    ok = True
    url = base_url()
    print(f"hub url : {url}")
    print(f"user    : {_user() or '(unset)'}")
    print(f"token   : {env_loader.redact(_token())}")
    print(f"password: {env_loader.redact(env_loader.get('JUPYTERHUB_PASSWORD', ''))}")
    print()

    # 1. Is it a JupyterHub, and is it up? /hub/api needs no auth.
    try:
        status, body, _ = _api("/")
        info = json.loads(body) if body.strip().startswith("{") else {}
        print(f"[ok]   reachable — JupyterHub {info.get('version', '?')} (HTTP {status})")
    except urllib.error.HTTPError as e:
        print(f"[FAIL] {url}/hub/api returned HTTP {e.code}. Check JUPYTERHUB_URL — it "
              f"should be the hub root, e.g. https://hub.example.edu")
        return 1
    except Exception as e:                                # noqa: BLE001
        print(f"[FAIL] cannot reach {url}: {type(e).__name__}: {e}")
        print("       Check the URL, and whether you need the institution's VPN.")
        return 1

    if not _user():
        print("[warn] JUPYTERHUB_USER is unset — cannot check credentials.")
        ok = False

    # 2. Credentials.
    token = _token()
    if token:
        try:
            status, body, _ = _api("/user", token=token)
            me = json.loads(body)
            print(f"[ok]   token accepted — authenticated as {me.get('name')!r}"
                  + (f", admin={me.get('admin')}" if "admin" in me else ""))
            if _user() and me.get("name") != _user():
                print(f"[warn] JUPYTERHUB_USER={_user()!r} but the token belongs to "
                      f"{me.get('name')!r}. The token wins; fix the .env.")
        except urllib.error.HTTPError as e:
            print(f"[FAIL] token rejected (HTTP {e.code}). Mint a new one: "
                  f"{url}/hub/token")
            ok = False
    elif env_loader.get("JUPYTERHUB_PASSWORD", ""):
        ok = _check_password(url) and ok
    else:
        print("[warn] neither JUPYTERHUB_TOKEN nor JUPYTERHUB_PASSWORD is set — "
              "nothing to check.")
        ok = False

    print()
    print("OK" if ok else "Some checks failed (see above).")
    return 0 if ok else 1


def _check_password(url):
    """POST the login form. A hub that accepts the credentials redirects and sets a
    session cookie; one that rejects them re-renders the login page with an error."""
    jar = CookieJar()
    opener = urllib.request.build_opener(urllib.request.HTTPCookieProcessor(jar))

    # GET the login page FIRST, for its _xsrf. JupyterHub 5.x rejects a login POST that
    # does not echo the token back -- with a bare HTTP 403 and no message, which reads
    # exactly like a wrong password and is why this check reported a credential failure
    # against a hub whose credentials were fine (verified against 5.5.0, 2026-08-02).
    xsrf = ""
    try:
        _, page, _ = _request(f"{url}/hub/login", opener=opener)
        for c in jar:
            if c.name == "_xsrf":
                xsrf = c.value
        if not xsrf:
            m = re.search(r'name="_xsrf"\s+value="([^"]+)"', page)
            xsrf = m.group(1) if m else ""
    except Exception:                                     # noqa: BLE001
        pass                                              # fall through; POST may still work

    data = urllib.parse.urlencode({
        "username": _user(), "password": env_loader.get("JUPYTERHUB_PASSWORD", ""),
        "_xsrf": xsrf,
    }).encode()
    try:
        status, body, _ = _request(
            f"{url}/hub/login?next=", data=data,
            headers={"Content-Type": "application/x-www-form-urlencoded"},
            opener=opener,
        )
    except urllib.error.HTTPError as e:
        print(f"[FAIL] login POST returned HTTP {e.code}.")
        return False
    except Exception as e:                                # noqa: BLE001
        print(f"[FAIL] login POST failed: {type(e).__name__}: {e}")
        return False

    names = {c.name for c in jar}
    got_session = any(n.startswith("jupyterhub-session-id") or n.startswith("jupyterhub-hub-login")
                      for n in names)
    lowered = body.lower()
    if "invalid username or password" in lowered or "login error" in lowered:
        print("[FAIL] the hub rejected the username/password.")
        return False
    if got_session:
        print(f"[ok]   password accepted for {_user()!r} (session cookie issued)")
        print("       Tip: mint an API token in the hub UI and set JUPYTERHUB_TOKEN — "
              "start/stop need one, and it keeps the password out of every request.")
        return True
    # No cookie and no error string: almost always an SSO hub, where /hub/login is a
    # redirect to the identity provider and there is no password to post.
    print("[warn] could not confirm the password. This hub most likely uses SSO/OAuth "
          "rather than a username+password form.")
    print("       Log in through the browser once, then File -> Hub Control Panel -> "
          "Token, and put that in JUPYTERHUB_TOKEN.")
    return False


def _require_token():
    token = _token()
    if not token:
        raise SystemExit(
            "This needs JUPYTERHUB_TOKEN (the password cannot drive the REST API).\n"
            f"  Mint one at {base_url()}/hub/token and add it to .env."
        )
    return token


def cmd_servers(args):
    token, user = _require_token(), _user()
    if not user:
        raise SystemExit("JUPYTERHUB_USER is not set.")
    _, body, _ = _api(f"/users/{urllib.parse.quote(user)}", token=token)
    me = json.loads(body)
    servers = me.get("servers") or {}
    if not servers:
        print(f"{user}: no server running. Start one with:  ./run.sh hub start")
        return 0
    for name, s in servers.items():
        print(f"{user}/{name or '(default)'}: ready={s.get('ready')} "
              f"url={s.get('url')} started={s.get('started')}")
    return 0


def cmd_start(args):
    token, user = _require_token(), _user()
    path = f"/users/{urllib.parse.quote(user)}/server"
    try:
        status, body, _ = _api(path, token=token, method="POST", data={})
    except urllib.error.HTTPError as e:
        if e.code == 400 and b"already" in (e.read() or b"").lower():
            print("server is already running.")
            return 0
        raise
    # 201 = started; 202 = spawning (hubs with a slow spawner return this).
    print({201: "server started.", 202: "server is spawning — check `./run.sh hub servers`."}
          .get(status, f"HTTP {status}: {body[:200]}"))
    return 0


def cmd_stop(args):
    token, user = _require_token(), _user()
    path = f"/users/{urllib.parse.quote(user)}/server"
    status, body, _ = _api(path, token=token, method="DELETE")
    print({204: "server stopped.", 202: "server is stopping."}
          .get(status, f"HTTP {status}: {body[:200]}"))
    return 0


def cmd_token(args):
    url = base_url()
    print(f"Mint an API token at:  {url}/hub/token")
    print("(or in the running server: File -> Hub Control Panel -> Token)")
    print("Then add it to .env as:  JUPYTERHUB_TOKEN=<the token>")
    print("\nA token is preferred over the password: it is revocable, it is the only "
          "thing the REST API accepts, and it works on SSO hubs where there is no "
          "password to post.")
    return 0


def main():
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd")
    for name, fn, helptext in (
        ("check", cmd_check, "URL reachable + credentials accepted"),
        ("servers", cmd_servers, "list your running servers (needs a token)"),
        ("start", cmd_start, "start your single-user server (needs a token)"),
        ("stop", cmd_stop, "stop your server, freeing the GPUs (needs a token)"),
        ("token", cmd_token, "how to mint an API token"),
    ):
        sp = sub.add_parser(name, help=helptext)
        sp.set_defaults(func=fn)
    args = ap.parse_args()
    if not getattr(args, "func", None):
        ap.print_help()
        return 2

    env_loader.load(verbose=True)
    if not os.environ.get("JUPYTERHUB_URL"):
        raise SystemExit(
            "JUPYTERHUB_URL is not set.\n"
            f"  Copy {env_loader.REPO_ROOT / '.env.example'} to "
            f"{env_loader.REPO_ROOT / '.env'} and paste in the three values from the "
            f"research program's email (JupyterHub, Username, Password)."
        )
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main() or 0)
