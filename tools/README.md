# tools/

Profiling harness for the streaming refactor. Both scripts predate the
refactor and are the source of the baseline measurements the plan's memory
goals are judged against. Install their dependencies with:

    pip install -e .[profile]

`codecarbon` (optional energy estimate in `profile_run.py`) is left out of
the extras deliberately; install it manually if wanted.

## Scripts

**`profile_run.py`** — runs the `seasound` CLI unchanged and samples the
whole process tree (coordinator + workers). Produces per-process peak RSS,
whole-run peak, core usage, and a sequential-run floor. This is the number
that gates the refactor's headline goal (max worker peak, currently ~2 GB).

**`profile_seasound_stages.py`** — runs one file through the exact stages
of `_process_one_file` and attributes peak RSS to each stage
(`read_audio`, `apply_calibration`, `compute_base_matrix`, optional
`get_stft_for_file`). This is the per-stage attribution that identified
`compute_base_matrix` as dominant.

## Baseline capture procedure (do once, at the golden tag)

1. Check out `golden-float32-baseline`.
2. Pick the representative deployment/config used for the original ~2 GB
   observation (same machine, same worker count).
3. Whole-run baseline:
   `python tools/profile_run.py --label golden_float32_baseline --detailed -- seasound --config <cfg> --auto`
4. Per-stage baseline (one representative file):
   `python tools/profile_seasound_stages.py --config <cfg>`
5. Commit the resulting summary CSVs to `docs/profiling/golden_float32_baseline/`.

Re-run step 3 with a new label after each refactor stage lands (mandatory
after Stage 2 and Stage 3); compare `max_worker_peak_mem_mb` and
`wall_clock_s` against the committed baseline. The plan's acceptance
criteria: per-worker peak independent of file duration, >10x below
baseline, with no wall-clock regression.