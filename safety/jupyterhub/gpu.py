"""gpu.py — pin this process to specific physical GPUs on a shared box, and prove it.

THE PROBLEM THIS SOLVES. The research program's JupyterHub A100 machine is shared.
We agreed to use GPUs 4 and 5 so two people's runs do not land on the same card and
halve each other's throughput (or OOM each other). "Agreed" is not enforcement: a
default `device_map="auto"` grabs every visible card on the box.

So: `pin("4,5")` sets CUDA_VISIBLE_DEVICES **before torch is imported**, and refuses
to start if either card is already occupied.

TWO WAYS THIS SILENTLY GOES WRONG, both handled here:

  1. CUDA does not number GPUs the way nvidia-smi does. CUDA's default
     CUDA_DEVICE_ORDER is FASTEST_FIRST, so `CUDA_VISIBLE_DEVICES=4,5` can select
     two different physical cards from the two nvidia-smi calls "4" and "5" — on a
     shared box that means running on somebody else's GPU while believing you are
     not. `pin()` forces CUDA_DEVICE_ORDER=PCI_BUS_ID, which makes the two
     numberings agree.

  2. Setting CUDA_VISIBLE_DEVICES after torch has initialised CUDA does nothing at
     all, with no error. `pin()` raises if torch is already imported.

And then `verify()`, called after torch is imported, checks the UUIDs of the cards
torch actually sees against the UUIDs nvidia-smi reported for the requested indices.
That is the only check that can actually catch (1); the rest is prevention.

Stdlib only — it must run before the venv has torch in it.
"""

import os
import subprocess
import sys

# This directory on sys.path, so `from env_loader import ...` resolves whether the
# module is run as a script (`python3 safety/jupyterhub/gpu.py`) or imported as
# `safety.jupyterhub.gpu` from a notebook. Same pattern the commitment/ drivers use.
_HERE = os.path.dirname(os.path.abspath(__file__))
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)

_SMI_GPU_FIELDS =("index", "uuid", "name", "memory.total", "memory.used",
                   "utilization.gpu")
_SMI_APP_FIELDS = ("gpu_uuid", "pid", "used_memory", "process_name")


class GpuSelectionError(RuntimeError):
    """Raised when the requested GPUs cannot be pinned safely."""


def _smi(query, fields):
    out = subprocess.run(
        ["nvidia-smi", f"--query-{query}={','.join(fields)}",
         "--format=csv,noheader,nounits"],
        capture_output=True, text=True,
    )
    if out.returncode != 0:
        raise GpuSelectionError(
            f"nvidia-smi failed ({out.returncode}): {out.stderr.strip() or out.stdout.strip()}"
        )
    rows = []
    for line in out.stdout.splitlines():
        line = line.strip()
        if not line:
            continue
        parts = [p.strip() for p in line.split(",")]
        if len(parts) != len(fields):
            continue
        rows.append(dict(zip(fields, parts)))
    return rows


def _norm_uuid(u) -> str:
    """nvidia-smi says 'GPU-<hex>'; torch's device properties say '<hex>'."""
    return str(u).strip().lower().removeprefix("gpu-")


def query() -> list:
    """Every GPU on the box, as nvidia-smi sees it (so: physical indices)."""
    gpus = []
    for r in _smi("gpu", _SMI_GPU_FIELDS):
        gpus.append({
            "index": int(r["index"]),
            "uuid": r["uuid"],
            "name": r["name"],
            "memory_total_mib": int(r["memory.total"]),
            "memory_used_mib": int(r["memory.used"]),
            "utilization_pct": int(r["utilization.gpu"]) if r["utilization.gpu"].isdigit() else None,
        })
    return sorted(gpus, key=lambda g: g["index"])


def compute_apps() -> list:
    """Processes currently holding GPU memory — i.e. whose job you would be joining."""
    apps = []
    try:
        rows = _smi("compute-apps", _SMI_APP_FIELDS)
    except GpuSelectionError:
        return apps
    for r in rows:
        apps.append({
            "gpu_uuid": r["gpu_uuid"],
            "pid": r["pid"],
            "used_mib": int(r["used_memory"]) if r["used_memory"].isdigit() else None,
            "process": r["process_name"],
        })
    return apps


def parse_spec(spec):
    """'4,5' -> [4, 5].  '' / 'all' -> None (do not pin).  'cpu'/'none' -> [] (no GPU).

    Returning None rather than "every index" matters: not pinning leaves whatever
    CUDA_VISIBLE_DEVICES the hub already set alone, which is the right behaviour on a
    box where the scheduler allocates your cards for you.
    """
    if spec is None:
        return None
    s = str(spec).strip().lower()
    if s in ("", "all"):
        return None
    if s in ("cpu", "none"):
        return []
    out = []
    for part in s.replace(",", " ").split():
        if not part.isdigit():
            raise GpuSelectionError(
                f"SAFETY_GPUS={spec!r} is not a GPU list. Use nvidia-smi indices, "
                f"e.g. '4,5'; or 'all' to leave the selection alone; or 'cpu'."
            )
        idx = int(part)
        if idx not in out:
            out.append(idx)
    return out


def describe(gpus=None, selected=None) -> str:
    """A one-GPU-per-line table for the preflight report and for run logs."""
    gpus = gpus if gpus is not None else query()
    apps = compute_apps()
    by_uuid = {}
    for a in apps:
        by_uuid.setdefault(_norm_uuid(a["gpu_uuid"]), []).append(a)
    lines = []
    for g in gpus:
        mark = "  "
        if selected is not None:
            mark = "->" if g["index"] in selected else "  "
        procs = by_uuid.get(_norm_uuid(g["uuid"]), [])
        who = ("  busy: " + ", ".join(f"pid {p['pid']} ({p['used_mib']}MiB)" for p in procs)
               if procs else "")
        lines.append(
            f"{mark} [{g['index']}] {g['name']:<28} "
            f"{g['memory_used_mib']:>6}/{g['memory_total_mib']:<6} MiB  "
            f"util {str(g['utilization_pct']) + '%':>4}{who}"
        )
    return "\n".join(lines)


def pin(spec, *, allow_busy=False, busy_mib=512, verbose=True) -> dict:
    """Restrict this process to `spec` (nvidia-smi indices) and return what it got.

    MUST be called before torch is imported. Returns a selection dict; pass it to
    verify() once torch is available.
    """
    if "torch" in sys.modules:
        raise GpuSelectionError(
            "torch is already imported, so setting CUDA_VISIBLE_DEVICES now would be "
            "silently ignored — torch reads it when CUDA first initialises. Call "
            "gpu.pin() BEFORE any import that pulls in torch (transformers and "
            "harness both do).\n"
            "  In a script: import this module first.\n"
            "  In a notebook: run the pin cell first, and restart the kernel if you "
            "have already imported torch."
        )

    indices = parse_spec(spec)
    if indices is None:
        inherited = os.environ.get("CUDA_VISIBLE_DEVICES", "")
        if verbose:
            print(f"[gpu] SAFETY_GPUS={spec!r}: not pinning. "
                  f"CUDA_VISIBLE_DEVICES={inherited or '(unset -> all GPUs)'}")
        return {"spec": spec, "indices": None, "pinned": False,
                "cuda_visible_devices": inherited, "gpus": []}

    if indices == []:
        os.environ["CUDA_VISIBLE_DEVICES"] = ""
        if verbose:
            print("[gpu] SAFETY_GPUS=cpu: CUDA hidden from this process.")
        return {"spec": spec, "indices": [], "pinned": True,
                "cuda_visible_devices": "", "gpus": []}

    gpus = query()
    by_index = {g["index"]: g for g in gpus}
    missing = [i for i in indices if i not in by_index]
    if missing:
        raise GpuSelectionError(
            f"GPU {missing} not present on this machine. Available:\n{describe(gpus)}"
        )

    # --- the shared-box guard -------------------------------------------------
    apps = compute_apps()
    busy = []
    for i in indices:
        g = by_index[i]
        if g["memory_used_mib"] > busy_mib:
            procs = [a for a in apps if _norm_uuid(a["gpu_uuid"]) == _norm_uuid(g["uuid"])]
            busy.append((g, procs))
    if busy and not allow_busy:
        detail = "\n".join(
            f"  GPU {g['index']} ({g['name']}): {g['memory_used_mib']}MiB in use"
            + ("".join(f"\n      pid {p['pid']} {p['process']} ({p['used_mib']}MiB)"
                       for p in procs) if procs else "  (no process listed — another container?)")
            for g, procs in busy
        )
        raise GpuSelectionError(
            f"Refusing to start: GPU(s) already busy (threshold {busy_mib}MiB).\n{detail}\n\n"
            f"Somebody else is very likely running on them, and sharing a card halves "
            f"both jobs (or OOMs them).\n"
            f"  * pick free cards:  SAFETY_GPUS=<a,b> ...\n"
            f"  * it is your own job, or the memory is a stale context: "
            f"SAFETY_GPU_ALLOW_BUSY=1 ...\n\n"
            f"Full board:\n{describe(gpus, selected=indices)}"
        )

    # PCI_BUS_ID first: without it CUDA enumerates "fastest first", so index 4 in
    # CUDA_VISIBLE_DEVICES need not be the card nvidia-smi calls 4. See module docstring.
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"] = ",".join(str(i) for i in indices)

    selection = {
        "spec": spec,
        "indices": indices,
        "pinned": True,
        "cuda_visible_devices": os.environ["CUDA_VISIBLE_DEVICES"],
        "cuda_device_order": "PCI_BUS_ID",
        "busy_mib_threshold": busy_mib,
        "allow_busy": allow_busy,
        "gpus": [{
            "index": by_index[i]["index"],
            "uuid": by_index[i]["uuid"],
            "name": by_index[i]["name"],
            "memory_total_mib": by_index[i]["memory_total_mib"],
            "memory_used_before_mib": by_index[i]["memory_used_mib"],
        } for i in indices],
    }
    if verbose:
        print(f"[gpu] pinned to physical GPU {indices} "
              f"(CUDA_VISIBLE_DEVICES={selection['cuda_visible_devices']}, "
              f"CUDA_DEVICE_ORDER=PCI_BUS_ID)")
        print(describe(gpus, selected=indices))
        if busy:
            print(f"[gpu] WARNING: {len(busy)} pinned GPU(s) already in use and "
                  f"SAFETY_GPU_ALLOW_BUSY=1 — you are sharing.")
    return selection


def verify(selection, verbose=True) -> dict:
    """Check that torch sees exactly the physical cards pin() selected.

    Call AFTER torch is imported. This is the check that catches a device-order
    mismatch: the count check alone would pass while running on the wrong cards.
    """
    import torch

    count = torch.cuda.device_count()
    result = {"device_count": count, "devices": [], "verified": True, "checks": []}

    if not selection.get("pinned"):
        result["checks"].append("not pinned — nothing to verify")
        if verbose:
            print(f"[gpu] torch sees {count} GPU(s); selection was not pinned.")
        return result

    want = selection["indices"]
    if count != len(want):
        result["verified"] = False
        result["checks"].append(
            f"torch.cuda.device_count()={count}, expected {len(want)} "
            f"(CUDA_VISIBLE_DEVICES={os.environ.get('CUDA_VISIBLE_DEVICES')!r}). "
            f"The mask did not take effect — was torch imported before pin()?"
        )
        if verbose:
            print("[gpu] FAIL: " + result["checks"][-1])
        return result

    for i in range(count):
        props = torch.cuda.get_device_properties(i)
        want_gpu = selection["gpus"][i]
        dev = {
            "cuda_index": i,
            "physical_index": want_gpu["index"],
            "name": props.name,
            "total_mib": round(props.total_memory / 1024 ** 2),
            "expected_uuid": want_gpu["uuid"],
        }
        got_uuid = getattr(props, "uuid", None)
        if got_uuid is not None:
            dev["uuid"] = str(got_uuid)
            if _norm_uuid(got_uuid) != _norm_uuid(want_gpu["uuid"]):
                result["verified"] = False
                result["checks"].append(
                    f"cuda:{i} is UUID {got_uuid}, but physical GPU "
                    f"{want_gpu['index']} is {want_gpu['uuid']}. CUDA is enumerating "
                    f"devices in a different order than nvidia-smi — you are NOT on "
                    f"the cards you asked for."
                )
        else:
            # Older torch without .uuid: fall back to the model name. Weak on a
            # homogeneous box, so say so rather than implying a real identity check.
            result["checks"].append(
                f"cuda:{i}: torch exposes no device UUID; identity checked by name only")
            if props.name != want_gpu["name"]:
                result["verified"] = False
                result["checks"].append(
                    f"cuda:{i} is a {props.name}, expected {want_gpu['name']}")
        result["devices"].append(dev)

    if verbose:
        for d in result["devices"]:
            print(f"[gpu] cuda:{d['cuda_index']} = physical {d['physical_index']} "
                  f"{d['name']} ({d['total_mib']}MiB)"
                  + (f" uuid={d['uuid']}" if "uuid" in d else ""))
        print("[gpu] identity verified." if result["verified"]
              else "[gpu] IDENTITY CHECK FAILED:\n  " + "\n  ".join(result["checks"]))
    return result


def pin_from_env(verbose=True) -> dict:
    """pin() using SAFETY_GPUS / SAFETY_GPU_BUSY_MIB / SAFETY_GPU_ALLOW_BUSY.

    Loads .env first, so importing this and calling it is all a notebook needs.
    """
    from env_loader import get, get_bool, get_int, load

    load(verbose=verbose)
    return pin(
        get("SAFETY_GPUS", ""),
        allow_busy=get_bool("SAFETY_GPU_ALLOW_BUSY", False),
        busy_mib=get_int("SAFETY_GPU_BUSY_MIB", 512),
        verbose=verbose,
    )


if __name__ == "__main__":
    import argparse
    import contextlib

    ap = argparse.ArgumentParser(description="Show the GPU board, or emit the pin as "
                                             "shell exports.")
    ap.add_argument("--export", action="store_true",
                    help="print `export CUDA_...` lines for `eval \"$(...)\"`, so the "
                         "pin applies to every child process (pytest, a jupyter "
                         "kernel, the batch-equivalence test) and not just to python "
                         "code that remembers to call pin()")
    _args = ap.parse_args()

    if not _args.export:
        print(describe())
        raise SystemExit(0)

    # The board and any warning go to stderr; only the export lines go to stdout, or
    # `eval` would try to run the table as shell.
    try:
        with contextlib.redirect_stdout(sys.stderr):
            _sel = pin_from_env(verbose=True)
    except GpuSelectionError as e:
        # A traceback here is noise -- the message IS the diagnosis, and the caller is
        # a shell script under `set -e` that just needs the non-zero exit.
        print(f"\n{e}", file=sys.stderr)
        raise SystemExit(1) from None
    if _sel.get("pinned"):
        if "cuda_device_order" in _sel:
            print(f"export CUDA_DEVICE_ORDER={_sel['cuda_device_order']}")
        print(f"export CUDA_VISIBLE_DEVICES={_sel['cuda_visible_devices']}")
