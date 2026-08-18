"""Wall-time benchmarks for the grid-anchored LOESS / binned smECE fast paths,
plus IVAP, ENIR, and the default-menu selector.

Deterministic (seeded portfolio) and self-contained (probcal + stdlib + numpy):

    uv run python docs/scripts/benchmarks.py                # default sizes
    uv run python docs/scripts/benchmarks.py 10000 100000    # custom sizes

Prints one GitHub-markdown table of wall times (and, for ENIR, tracemalloc
peak memory) per size, one row per benchmarked call. IVAP/ENIR/selector rows
are feasibility-gated: a row is skipped for a requested size beyond its
measured-feasible range (see the ``_*_MAX_*`` constants) rather than running
for an impractical amount of time.
"""

import sys
import time
import tracemalloc
from collections.abc import Callable

from probcal import (
    CalibratorSelector,
    ENIRCalibrator,
    PlattCalibrator,
    VennAbersCalibrator,
    make_pd_portfolio,
)
from probcal.metrics import evaluate, ici, smooth_ece

DEFAULT_SIZES = (10_000, 100_000, 1_000_000)

# Feasibility ceilings for the newer rows, derived from measured wall times:
# IVAP fit/predict and the selector's default menu are cheap through n=1e5
# but a 1e6 default-sweep entry would dominate the run; ENIR's vectorized
# engine is only benchmarked through m=5e4 (see the footnote in main()).
_IVAP_MAX_N = 100_000
_ENIR_MAX_M = 50_000
_SELECTOR_MAX_N = 100_000


def _run(call: Callable[[], object], measure_memory: bool) -> tuple[float, float | None]:
    """Run ``call`` (twice, if ``measure_memory``), returning wall time and
    tracemalloc peak MB.

    Wall time and peak memory are measured in separate calls when
    ``measure_memory`` is set: tracemalloc's allocation tracing measurably
    inflates wall time for Python-loop-heavy calls (ENIR's retention/BIC
    bookkeeping), so timing the traced call would overstate its cost.
    """
    t0 = time.perf_counter()
    call()
    elapsed = time.perf_counter() - t0
    if not measure_memory:
        return elapsed, None
    tracemalloc.start()
    call()
    _, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    return elapsed, peak / (1024 * 1024)


def _rows(n: int) -> list[tuple[str, Callable[[], object], bool]]:
    """Benchmark rows for one portfolio size.

    Each row is ``(label, call, measure_memory)``. Every lambda captures its
    inputs via default arguments (``d=d``, etc.) rather than the enclosing
    scope, so rows generated in a loop stay ruff B023-safe.
    """
    d = make_pd_portfolio(n=n, random_state=42)
    rows: list[tuple[str, Callable[[], object], bool]] = [
        ("ici(d.y, d.scores)", lambda d=d: ici(d.y, d.scores), False),
        ("smooth_ece(d.y, d.scores)", lambda d=d: smooth_ece(d.y, d.scores), False),
        (
            "evaluate(d.y, d.scores, n_boot=100)",
            lambda d=d: evaluate(d.y, d.scores, n_boot=100),
            False,
        ),
        (
            "PlattCalibrator().fit(d.scores, d.y)",
            lambda d=d: PlattCalibrator().fit(d.scores, d.y),
            False,
        ),
    ]
    if n <= _IVAP_MAX_N:
        ivap = VennAbersCalibrator().fit(d.scores, d.y)
        rows.append(
            (
                "VennAbersCalibrator().fit(d.scores, d.y)",
                lambda d=d: VennAbersCalibrator().fit(d.scores, d.y),
                False,
            )
        )
        rows.append(
            (
                "ivap.predict_interval(d.scores)",
                lambda ivap=ivap, d=d: ivap.predict_interval(d.scores),
                False,
            )
        )
    if n <= _ENIR_MAX_M:
        rows.append(
            (
                "ENIRCalibrator().fit(d.scores, d.y)",
                lambda d=d: ENIRCalibrator().fit(d.scores, d.y),
                True,
            )
        )
    if n <= _SELECTOR_MAX_N:
        rows.append(
            (
                "CalibratorSelector().fit(d.scores, d.y)",
                lambda d=d: CalibratorSelector().fit(d.scores, d.y),
                False,
            )
        )
    return rows


def main() -> None:
    sizes = [int(a) for a in sys.argv[1:]] or list(DEFAULT_SIZES)

    print("| n | call | wall time (s) | tracemalloc peak (MB) |")
    print("| --- | --- | --- | --- |")
    for n in sizes:
        for label, call, measure_memory in _rows(n):
            elapsed, peak_mb = _run(call, measure_memory)
            peak_str = f"{peak_mb:.1f}" if peak_mb is not None else "-"
            print(f"| {n:,} | `{label}` | {elapsed:.3f} | {peak_str} |")

    print(
        "\nENIR is not benchmarked at n=10^6: even bounded retention would hold "
        "roughly 2 GB there, so its path is only measured through m=5e4. ENIR's "
        "fit time grows ~quadratically with m while its memory stays bounded -- "
        "that bound (not a matching wall-time improvement) is the production fix."
    )


if __name__ == "__main__":
    main()
