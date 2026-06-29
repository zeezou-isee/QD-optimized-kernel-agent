# batch/ — driving OperatorAgent across operator datasets

One runner, many sets. Each set is just a small config file declaring which
dataset to scan and what knobs to pass.

## Quick start

```bash
# whole miniset (~9 ops)
DEEPSEEK_API_KEY=sk-... .venv/bin/python batch/batch_runner.py --set miniset

# single op for debugging
DEEPSEEK_API_KEY=sk-... .venv/bin/python batch/batch_runner.py --set miniset --ops Abs

# subset (~30) or all (~183)
DEEPSEEK_API_KEY=sk-... .venv/bin/python batch/batch_runner.py --set subset
DEEPSEEK_API_KEY=sk-... .venv/bin/python batch/batch_runner.py --set all
```

Results land at `batch/results/<set>.json` and are resumable — a re-run skips
ops that already have a non-{crash,timeout} status.

## Layout

```
batch/
├── batch_runner.py        # the only place run logic lives
├── sets/<name>.py         # per-set constants (DATASET / TIMEOUT / MODEL / ...)
├── results/<name>.json    # one results file per set, written by the runner
└── README.md              # you are here
```

## Adding a new set

Drop a `batch/sets/<name>.py` with the same module-level constants as
`miniset.py`. The runner discovers it via `--set <name>`; no other registration
needed.

## Safety notes (already handled inside the runner)

- Each OperatorAgent runs in its own process group; on per-op timeout the
  whole tree is killed (`SIGTERM`, 3 s grace for the ncnn-tree guard,
  `SIGKILL` fallback). No orphan compilers leaking into the next op.
- `.venv/bin` is prepended to the subprocess PATH so `cmake` (pip-installed
  inside the venv) is always findable — no need to `export PATH=...` by hand.
- `--auto-cleanup` is passed to every OperatorAgent invocation so a leaked
  ncnn tree from a prior killed run is silently restored at startup.
