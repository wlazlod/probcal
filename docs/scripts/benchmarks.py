"""Wall-time benchmarks for the grid-anchored LOESS / binned smECE fast paths.

Deterministic (seeded portfolio) and self-contained (probcal + stdlib + numpy):

    uv run python docs/scripts/benchmarks.py                # default sizes
    uv run python docs/scripts/benchmarks.py 10000 100000    # custom sizes

Prints one GitHub-markdown table of wall times per size, one row per
benchmarked call.
"""

import sys
import time
from collections.abc import Callable

from probcal import PlattCalibrator, make_pd_portfolio
from probcal.metrics import evaluate, ici, smooth_ece

DEFAULT_SIZES = (10_000, 100_000, 1_000_000)


def _rows(n: int) -> list[tuple[str, Callable[[], object]]]:
    """Benchmark rows for one portfolio size.

    Each lambda captures ``d`` via a default argument (``d=d``) rather than
    the enclosing scope, so rows generated in a loop (e.g. a future
    IVAP/ENIR/selector row per calibrator) stay ruff B023-safe.
    """
    d = make_pd_portfolio(n=n, random_state=42)
    return [
        ("ici(d.y, d.scores)", lambda d=d: ici(d.y, d.scores)),
        ("smooth_ece(d.y, d.scores)", lambda d=d: smooth_ece(d.y, d.scores)),
        (
            "evaluate(d.y, d.scores, n_boot=100)",
            lambda d=d: evaluate(d.y, d.scores, n_boot=100),
        ),
        (
            "PlattCalibrator().fit(d.scores, d.y)",
            lambda d=d: PlattCalibrator().fit(d.scores, d.y),
        ),
    ]


def main() -> None:
    sizes = [int(a) for a in sys.argv[1:]] or list(DEFAULT_SIZES)

    print("| n | call | wall time (s) |")
    print("| --- | --- | --- |")
    for n in sizes:
        for label, call in _rows(n):
            t0 = time.perf_counter()
            call()
            elapsed = time.perf_counter() - t0
            print(f"| {n:,} | `{label}` | {elapsed:.3f} |")


if __name__ == "__main__":
    main()
