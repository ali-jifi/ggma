"""env_loader.py — read .env into os.environ. Stdlib only.

No python-dotenv dependency: this has to work on the hub before anything is pip
installed, and inside the lean Modal orchestrator env, and in the torch-free CI.

PRECEDENCE, highest first:
    1. the real process environment (so `SAFETY_GPUS=6 ./run.sh` is a one-off)
    2. safety/jupyterhub/.env      machine-local overrides on the hub box
    3. <repo root>/.env            the shared file — credentials, defaults

`GGMA_ENV_FILE=/path/to/file` replaces the search entirely with that one file.

Not clobbering an already-set variable is the standard dotenv contract, and it is
what makes the shell override work. `load(override=True)` inverts it, which is
only correct in a notebook where you edited .env and want the new value now.
"""

import os
import sys
from pathlib import Path

SAFETY_DIR = Path(__file__).resolve().parents[1]     # .../safety
REPO_ROOT = SAFETY_DIR.parent                        # repo root
HUB_DIR = Path(__file__).resolve().parent            # .../safety/jupyterhub

# Lowest precedence first; later files override earlier ones.
SEARCH_ORDER = (REPO_ROOT / ".env", HUB_DIR / ".env")

_loaded: list = []
_announced = False


def env_files() -> list:
    """The .env files that exist, lowest precedence first."""
    override = os.environ.get("GGMA_ENV_FILE")
    if override:
        p = Path(override).expanduser()
        if not p.exists():
            raise SystemExit(f"GGMA_ENV_FILE={override} does not exist.")
        return [p]
    return [p for p in SEARCH_ORDER if p.exists()]


def parse(path) -> dict:
    """Minimal .env parser: KEY=VALUE, `#` comments, optional `export ` prefix,
    optional surrounding quotes. Deliberately NOT a shell: no interpolation, no
    command substitution, so a stray `$` or backtick in a password is safe."""
    out = {}
    for raw in Path(path).read_text().splitlines():
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        if line.startswith("export "):
            line = line[len("export "):].lstrip()
        if "=" not in line:
            continue
        key, _, val = line.partition("=")
        key = key.strip()
        if not key:
            continue
        val = val.strip()
        # Strip ONE matched pair of quotes. A password containing '#' must be
        # quoted, so the inline-comment strip below only applies to bare values.
        if len(val) >= 2 and val[0] == val[-1] and val[0] in ("'", '"'):
            val = val[1:-1]
        elif " #" in val:
            val = val.split(" #", 1)[0].rstrip()
        out[key] = val
    return out


def load(override=False, verbose=False) -> list:
    """Inject every .env file's values into os.environ. Returns the files used."""
    global _loaded, _announced
    files = env_files()
    for path in files:
        for key, val in parse(path).items():
            if val == "" and not override:
                # A blank in the template means "not set". Writing it would turn
                # `if HF_TOKEN:` into a false positive on the empty string.
                continue
            if override or key not in os.environ or os.environ[key] == "":
                os.environ[key] = val
    _loaded = files
    # Announce once per process: pin_from_env() calls load() too, and the same
    # "[env] loaded ..." line twice reads like two different files were read.
    if verbose and not _announced:
        _announced = True
        if files:
            print(f"[env] loaded {', '.join(str(f) for f in files)}")
        else:
            print(f"[env] no .env found (looked in {', '.join(str(f) for f in SEARCH_ORDER)}); "
                  f"using the process environment only")
    return files


def loaded_files() -> list:
    return list(_loaded)


# --- typed accessors ----------------------------------------------------------
_MISSING = object()


def get(key, default=None, required=False):
    val = os.environ.get(key, "")
    if val == "":
        if required:
            where = ", ".join(str(f) for f in env_files()) or str(REPO_ROOT / ".env")
            raise SystemExit(
                f"{key} is not set.\n"
                f"  Add it to {where} (start from {REPO_ROOT / '.env.example'}), "
                f"or export it in your shell."
            )
        return default
    return val


def get_int(key, default=None, required=False):
    val = get(key, _MISSING if required else None, required=required)
    if val is None or val is _MISSING:
        return default
    try:
        return int(val)
    except ValueError:
        raise SystemExit(f"{key}={val!r} is not an integer.")


def get_bool(key, default=False):
    val = os.environ.get(key, "").strip().lower()
    if val == "":
        return default
    if val in ("1", "true", "yes", "on"):
        return True
    if val in ("0", "false", "no", "off"):
        return False
    raise SystemExit(f"{key}={val!r} is not a boolean (use 1/0, true/false, yes/no).")


def get_path(key, default=None):
    val = get(key)
    return Path(val).expanduser() if val else default


def redact(val, keep=4) -> str:
    """For printing a token/password in a report without printing it."""
    if not val:
        return "(unset)"
    if len(val) <= keep:
        return "*" * len(val)
    return val[:keep] + "*" * (len(val) - keep)


if __name__ == "__main__":
    import json

    # `env_loader.py --get KEY` prints one value (empty line if unset). run.sh uses it
    # for SAFETY_OUTPUT_DIR: the wrapper decides where to tee terminal_output.txt and
    # the runner decides where to write the artifacts, and if those disagree the log
    # ends up in a different folder from the run it describes. Deliberately one key at
    # a time -- dumping the whole file as shell exports would put the hub password into
    # the environment of every child process.
    if len(sys.argv) == 3 and sys.argv[1] == "--get":
        load()
        print(os.environ.get(sys.argv[2], ""))
        raise SystemExit(0)

    files = load(verbose=True)
    secret = ("JUPYTERHUB_PASSWORD", "JUPYTERHUB_TOKEN", "HF_TOKEN")
    keys = ("JUPYTERHUB_URL", "JUPYTERHUB_USER", *secret, "SAFETY_GPUS",
            "SAFETY_GPU_BUSY_MIB", "SAFETY_GPU_ALLOW_BUSY", "SAFETY_DEVICE_MAP",
            "HF_HOME", "SAFETY_MODEL_PATH", "SAFETY_MODEL_ALIAS", "SAFETY_OUTPUT_DIR")
    print(json.dumps({
        "files": [str(f) for f in files],
        "values": {k: (redact(os.environ.get(k, "")) if k in secret
                       else os.environ.get(k, "") or "(unset)") for k in keys},
    }, indent=2))
