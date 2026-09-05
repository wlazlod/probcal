"""Coverage simulation for probcal.metrics.pluto_tasche.

Usage: ``uv run python docs/scripts/conservative_sim.py [--fast]``

Pluto-Tasche's most-prudent PD is a one-sided upper bound: it should cover
the true per-grade PD (``pd_upper_i >= pd_true_i``) in at least
``confidence`` of repeated draws, for every grade simultaneously (since the
bound pools nested sets across grades, per-grade coverage is not
independent -- the reported "all-grades" figure gives the joint rate).
``tests/test_conservative_sim.py`` enforces the same gate at a reduced run
count in CI; this script prints the full-size table pasted into
``docs/concepts/conservatism.md``.

Gate: the minimum per-grade coverage over grades is at least
``confidence - 2 * sqrt(confidence * (1 - confidence) / runs)``.
"""

import sys

import numpy as np

from probcal.metrics import pluto_tasche

N_PER_GRADE = (2000.0, 2000.0, 1000.0, 500.0)
PD_TRUE = (0.005, 0.01, 0.03, 0.08)


def coverage(
    runs: int,
    n_per_grade: object,
    pd_true: object,
    confidence: float = 0.9,
    seed: int = 42,
) -> dict:
    """Simulated coverage of ``pluto_tasche``'s upper bound across repeated draws.

    Parameters
    ----------
    runs : int
        Number of independent portfolios to draw.
    n_per_grade : array_like
        Obligor count per grade, best to worst (held fixed across runs).
    pd_true : array_like
        True per-grade PD (should be non-decreasing best to worst for a
        realistic rating structure); defaults per grade are drawn
        ``Binomial(n_per_grade, pd_true)``.
    confidence : float
        Confidence level passed to :func:`probcal.metrics.pluto_tasche`.
    seed : int
        Seed for ``numpy.random.default_rng``.

    Returns
    -------
    dict
        ``per_grade`` (coverage rate per grade, ``pd_upper_i >= pd_true_i``),
        ``min_per_grade`` (minimum over grades, the gated quantity), and
        ``all_grades`` (share of runs where every grade is covered
        simultaneously).
    """
    rng = np.random.default_rng(seed)
    n = np.asarray(n_per_grade, dtype=np.float64)
    pd_true_arr = np.asarray(pd_true, dtype=np.float64)
    k = len(n)
    covered_per_grade = np.zeros(k, dtype=np.int64)
    covered_all = 0
    for _ in range(runs):
        d = rng.binomial(n.astype(np.int64), pd_true_arr).astype(np.float64)
        res = pluto_tasche(n, d, confidence=confidence)
        ok = res.pd_upper >= pd_true_arr
        covered_per_grade += ok.astype(np.int64)
        covered_all += int(np.all(ok))
    per_grade = covered_per_grade / runs
    return {
        "per_grade": per_grade,
        "min_per_grade": float(np.min(per_grade)),
        "all_grades": covered_all / runs,
    }


def main() -> None:
    fast = "--fast" in sys.argv
    runs = 300 if fast else 2000
    rows = []
    for confidence in (0.9, 0.95):
        res = coverage(runs, N_PER_GRADE, PD_TRUE, confidence=confidence)
        gate = confidence - 2.0 * np.sqrt(confidence * (1.0 - confidence) / runs)
        rows.append(
            (
                confidence,
                runs,
                f"{res['min_per_grade']:.4f}",
                f"{res['all_grades']:.4f}",
                f">= {gate:.4f}",
            )
        )

    headers = (
        "confidence",
        "runs",
        "per-grade coverage (min over grades)",
        "all-grades coverage",
        "gate",
    )
    widths = [len(h) for h in headers]
    str_rows = [tuple(str(c) for c in row) for row in rows]
    for row in str_rows:
        for i, cell in enumerate(row):
            widths[i] = max(widths[i], len(cell))
    print("| " + " | ".join(h.ljust(widths[i]) for i, h in enumerate(headers)) + " |")
    print("|" + "|".join("-" * (w + 2) for w in widths) + "|")
    for row in str_rows:
        print("| " + " | ".join(c.ljust(widths[i]) for i, c in enumerate(row)) + " |")


if __name__ == "__main__":
    main()
