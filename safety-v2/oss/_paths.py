"""Path bootstrap shared by the `oss` modules.

The local tree and the Modal container disagree about where the repo root is, so both are resolved
here once instead of in five places. In the container the image sets OSS_REPO_ROOT / OSS_DATA_DIR.

This module deliberately puts `safety/drift` and `commitment` on the path: the v2 tree REUSES
`drift_screen` (labels, cluster statistics) and `harm_judge` (the one judge) rather than forking
them. Forking either would create two definitions of a reported number, which is the defect the
program already paid for once.
"""
import os
import sys
from pathlib import Path

_HERE = Path(__file__).resolve().parent                     # safety-v2/oss | /root/ggma/oss
REPO = Path(os.environ.get("OSS_REPO_ROOT", _HERE.parents[1]))
DATA = Path(os.environ.get("OSS_DATA_DIR", REPO / "safety" / "vendor" / "data"))
OUT = Path(os.environ.get("OSS_OUT_DIR", REPO / "do_not_track" / "oss"))

for _p in (REPO / "safety" / "drift", REPO / "commitment", _HERE):
    if _p.exists() and str(_p) not in sys.path:
        sys.path.insert(0, str(_p))
