# safety/jupyterhub — running the safety eval on the research program's A100 box

The same eval as `safety/modal_safety_test/`, on rented JupyterHub A100s instead of
Modal's serverless GPUs. **It is a launcher, not a second implementation**: the
collection loop (`safety/collect_core.py`) and the judge (`safety/judge_core.py`) are
shared with the Modal app, and the experiment constants live in one file
(`safety/run_config.py`), so a hub run and a Modal run of the same config produce the
same rows and the same summary. What lives here is what is genuinely local: reading
`.env`, pinning the GPUs, and writing artifacts to disk.

## The GPU rule

We agreed to stay on **GPU 4 and 5** so two people's runs do not land on the same card.
That is enforced, not just documented:

- `SAFETY_GPUS=4,5` in `.env` → `CUDA_VISIBLE_DEVICES` is set **before torch is
  imported** (after, it is silently ignored — see `gpu.py`).
- `CUDA_DEVICE_ORDER=PCI_BUS_ID` is forced, because CUDA's default order is "fastest
  first" and index 4 need not be the card `nvidia-smi` calls 4.
- A requested GPU with memory already allocated **aborts the run**, naming the PID
  holding it. `SAFETY_GPU_ALLOW_BUSY=1` overrides, for when the other job is yours.
- After torch loads, the **UUIDs** of the cards torch sees are compared against the
  ones `nvidia-smi` reported for indices 4 and 5. A count check alone would pass while
  running on the wrong cards.

Override for one run without editing anything: `SAFETY_GPUS=6,7 ./run.sh collect ...`

> **If `nvidia-smi` inside the hub shows only a couple of GPUs numbered from 0**, your
> hub hands out a container with cards passed through, and the host's "4,5" are not
> visible under those numbers. `pin` will refuse with `GPU [4, 5] not present` and list
> what *is* there. Set `SAFETY_GPUS` to the indices you actually see, or `all` if the
> container already gives you exactly your allocation.

## Setup

### 1. On your laptop — put the credentials in `.env`

When the research program's email arrives with [JupyterHub, Username, Password]:

```bash
cp .env.example .env       # from the repo root
$EDITOR .env               # fill in JUPYTERHUB_URL / _USER / _PASSWORD
safety/jupyterhub/run.sh hub-check
```

`hub-check` confirms the URL is a reachable JupyterHub and that the credentials are
accepted, so a typo surfaces now rather than five minutes before a run. `.env` is
git-ignored; keep it that way.

Once you are logged in, mint an API token (hub UI: File → Hub Control Panel → Token)
and set `JUPYTERHUB_TOKEN`. It is revocable, it is the only thing the REST API accepts
(`run.sh hub start|stop|servers`), and it works on SSO hubs where there is no password
to post.

### 2. On the hub — clone and bootstrap

In a hub terminal:

```bash
git clone <this repo> && cd ggma && git checkout safety
safety/jupyterhub/run.sh setup
```

`setup` builds `safety/jupyterhub/.venv`, installs the pins, copies `.env.example` to
`.env` if neither `.env` exists yet, and fetches the datasets (they are **not**
committed — `safety/vendor/fetch_datasets.py` pulls them from their own repos at
pinned commits).

If the hub image already ships a CUDA-matched torch, keep it:

```bash
SKIP_VENV=1 safety/jupyterhub/run.sh setup     # installs everything except torch
```

### 3. On the hub — a second `.env` for machine-local values

`safety/jupyterhub/.env` is loaded after the repo-root one and overrides it. Use it for
what differs on the box — most importantly the cache location, because `$HOME` on a hub
is usually a small quota'd volume and the 13B judge alone is ~26GB:

```bash
HF_HOME=/workspace/.cache/huggingface
SAFETY_OUTPUT_DIR=/workspace/ggma-outputs
```

HuggingFace auth needs nothing in `.env`: run `huggingface-cli login` once on the box and
the library picks the stored token up from `$HF_HOME/token` by itself. `HF_TOKEN` is only
for when you would rather not store one. Either way it is optional — the DeepSeek-R1-Distill
models and the HarmBench classifier are ungated, and a token only avoids anonymous rate
limits. `preflight` reports which of the two it found.

Then: `./run.sh preflight`.

### Everything else you can set

`.env.example` deliberately contains only what you must fill in. These all have working
defaults; add one to `.env` only to change it.

| variable | default | what it does |
|---|---|---|
| `SAFETY_GPU_BUSY_MIB` | `512` | a pinned GPU with more than this already allocated counts as busy and aborts the run |
| `SAFETY_GPU_ALLOW_BUSY` | `0` | `1` runs anyway. Only when you know the other job is yours |
| `SAFETY_DEVICE_MAP` | `auto` | `auto` shards across the pinned cards (needed for 14B/30B); `cuda:0` pins the model to the first one |
| `HF_HOME` | `~/.cache/huggingface` | where weights land. **Set this on the hub** |
| `HF_TOKEN` | *(stored login)* | only if you would rather not `huggingface-cli login` |
| `SAFETY_OUTPUT_DIR` | `safety/outputs/` | where run artifacts go |
| `SAFETY_MODEL_PATH` / `SAFETY_MODEL_ALIAS` | `safety/run_config.py` | override the model without editing code. Both or neither — the alias names the output dir and the direction cache |
| `GGMA_ENV_FILE` | — | use exactly this one file instead of searching the two default locations |

Every one is also a CLI flag on `run_local.py` where it makes sense (`--gpus`,
`--allow-busy`, `--device-map`, `--model-path`, `--run-name`), and the flag wins.

## Running

```bash
cd safety/jupyterhub

./run.sh gpus                       # the board: who is on which card
./run.sh preflight                  # imports, datasets, GPU identity. Every session.
./run.sh smoke                      # the full ladder, cheapest rung first
./run.sh smoke --skip-gpu           # the CPU rungs only

./run.sh collect --n 12 --budgets natural --batch-size 8 --samples 1
./run.sh collect --smoke-one                    # one trace, every stage, measured
./run.sh collect --n 30 --budgets 512,2048 --induction hijack_preamble \
                 --preamble-id hijack_bound_ordering --judge

./run.sh judge --run <run-name>     # re-score a checkpoint, no re-generation
./run.sh shell                      # a subshell with the GPUs pinned
```

`collect` and `judge` tee the whole console to `terminal_output.txt` in the run folder.

### The knobs that cost money and the ones that cost correctness

| flag | what it does |
|---|---|
| `--rollouts` | **the cost knob.** 0 = surface pass (everything task 1.0b needs). `k` adds `prefixes × k` generations **per trace** — ~90% of a full trace's cost. |
| `--batch-size` | the ~20× lever, but **surface pass at natural budget only**. Forced budgets share one suppression counter across the batch; rollouts already batch internally. Anything else raises. |
| `--budgets` | `natural`, or exact thinking-token counts forced by *suppression*. Suppression is not TTBC-2 — nothing here injects "Wait". At `--induction natural` with forced budgets, no induction mechanism is active and a flat curve tells you nothing. |
| `--judge` | HarmBench-13B, the **scorer of record** for 1.0b. Without it you have heuristic labels, which are for plumbing only. |
| `--device-map` | `auto` shards across both pinned cards (needed for 14B/30B). `cuda:0` keeps the model on GPU 4 so you can run a second job on GPU 5. |

### Two jobs, one per card

```bash
SAFETY_GPUS=4 ./run.sh collect --device-map cuda:0 --run-name arm-a &
SAFETY_GPUS=5 ./run.sh collect --device-map cuda:0 --run-name arm-b &
```

Each process sees exactly one card, so neither can grab the other's.

## The smoke ladder

`./run.sh smoke` runs eight rungs in increasing cost. Each one can only fail for
reasons the rungs below it have already ruled out, so it stops at the first failure
(`--keep-going` overrides).

| # | rung | needs |
|---|---|---|
| 1 | `safety/smoke_datasets.py` — data built, sampling seeded | — |
| 2 | `safety/smoke_prompts.py` — prompt dicts, split provenance | — |
| 3 | `safety/smoke_matcher.py` — the scorer labels what it should | — |
| 4 | `pytest commitment/tests safety/tests` — torch-free invariants | — |
| 5 | `preflight.py` — imports, registry, GPU identity | model config |
| 6 | `test_batch_equivalence.py` — batched == unbatched, token for token | GPU |
| 7 | `run_local.py collect --smoke-one` — one trace, every stage | GPU |
| 8 | `safety/smoke_curves.py` — rung 7's curve is readable | — |

Rung 6 is the one worth understanding: batching a decoder-only model needs left padding
and a correct attention mask, and getting it wrong makes the model attend to pad tokens
— output is subtly worse, nothing raises, and every batched number afterwards is quietly
wrong. Greedy admits an exact comparison, so that is what it pins.

## Output

`safety/outputs/<run-name>/` (git-ignored), or `SAFETY_OUTPUT_DIR`:

| file | what |
|---|---|
| `traces.jsonl` | harness `Trace` rows, schema trace-v0.3, one per line |
| `rows.json` | the same rows plus the bare request/category — re-judgeable without re-generating |
| `summary.json` | refusal/ASR vs budget, throughput, warm-up-hijack count |
| `run_meta.json` | full config, GPU selection **and the identity proof**, timings, torch version |
| `harmbench_scored.json`, `judge_report.json` | with `--judge` |
| `terminal_output.txt` | the console log (when launched via `./run.sh`) |

## Files

- `run.sh` — the entry point. Pins the GPUs once, in the parent shell, so every child
  process inherits it (pytest and the batch-equivalence test know nothing about
  `SAFETY_GPUS`).
- `env_loader.py` — `.env` → `os.environ`. Stdlib; no python-dotenv.
- `gpu.py` — discovery, pinning, the busy guard, the UUID identity check. Also
  `python3 gpu.py --export` for `eval "$(...)"`.
- `preflight.py` — the local counterpart of `app.py::smoke`.
- `smoke.py` — the ladder.
- `run_local.py` — `collect` and `judge`.
- `hubctl.py` — check credentials, start/stop your server.
- `quickstart.ipynb` — the same flow in cells, for working inside the hub.
- `requirements-hub.txt` — includes `../vendor/requirements-modal.txt` rather than
  restating the pins, so the hub and Modal cannot drift apart.

## Also read

- `safety/modal_safety_test/README.md` — the Modal side; the experiment design section
  applies to both.
- `papers/SAFETY_NEXT_SMOKE_PLAN.md` — which models to run, what it costs, and the two
  things that will silently ruin a run.
- `safety/vendor/DATASET_SOURCES.md` — what each dataset is and why.
