# safety/

Workspace for the safety-side experiments. The shared toolkit lives at the repo
root (`harness.py`); local scratch outputs stay in a git-ignored directory.

## Two launchers, one experiment

The eval can run on Modal's serverless GPUs or on the research program's JupyterHub
A100 box. They are launchers only — everything that defines the *experiment* is shared,
so the same config produces the same rows either way:

| file | what it owns |
|---|---|
| `run_config.py` | the experiment constants: model, extraction layer/pos, decode settings, dataset filenames |
| `collect_core.py` | the collection loop, the per-budget roll-up, the warm-up-hijack guard, the previews |
| `judge_core.py` | the HarmBench-13B pass, including the generation-clipping rule that decides what the judge sees |
| `modal_safety_test/` | Modal: the image, the Volume, the card (`SAFETY_GPU`), per-second cost |
| `jupyterhub/` | JupyterHub: `.env`, GPU pinning (we are on 4,5), artifacts to disk |

Start at `jupyterhub/README.md` or `modal_safety_test/README.md` depending on where you
are running.
