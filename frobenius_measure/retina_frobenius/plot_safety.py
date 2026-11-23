"""Run a plotting/rendering call in an isolated subprocess.

A plain try/except CANNOT catch a segfault -- a crash in a native library
(matplotlib's Agg backend, freetype, fontconfig, etc.) kills the process
before Python ever sees an exception, and every run_pipeline.sh uses
`set -euo pipefail`, so an uncaught crash silently takes the whole pipeline
down after that point. Running the render in a forked child process means a
crash there only kills the child; the parent sees a nonzero exit code and can
carry on.

Seen in production: aggregate_seeds.py segfaulted in matplotlib AFTER
successfully computing and saving aggregate_bins.csv, killing the rest of
run_pipeline.sh (Stage 2 never ran) even though the try/except around the
plot call was in place.
"""
from __future__ import annotations

import multiprocessing as mp


def safe_render(label: str, fn, *args, **kwargs) -> bool:
    """Run fn(*args, **kwargs) in a forked child process.

    Returns True on success. On failure (regular exception OR a crash/signal
    that killed the child outright) prints a one-line skip message and
    returns False -- never raises.
    """
    ctx = mp.get_context("fork")
    p = ctx.Process(target=fn, args=args, kwargs=kwargs)
    p.start()
    p.join()
    if p.exitcode == 0:
        return True
    print(f"{label} skipped: render subprocess exited with code {p.exitcode} "
          f"(nonzero/negative usually means a native-library crash, e.g. "
          f"matplotlib/fontconfig -- not a bug in the search itself)")
    return False
