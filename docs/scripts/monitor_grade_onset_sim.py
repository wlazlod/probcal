"""Full-size reporting tables for per-grade CS coverage (spec M2) and drift-onset
localization (spec M3).

Usage: ``uv run python docs/scripts/monitor_grade_onset_sim.py [--fast]``

Both gates already run in CI at reduced size:
``tests/test_monitor_grades.py::test_two_grade_drift_confidence_sequence_coverage``
(20 runs) and ``tests/test_monitor_onset.py::test_onset_localizes_injected_drift``
(40 runs). This script reruns the same constructions at the documented size
(100 runs) and prints the table pasted into ``docs/concepts/monitoring.md``; it
adds no new gate, only a larger, independently reproducible reading of the two
existing ones.

``grade_cs_coverage`` mirrors ``tests/test_monitor_grades.py``'s two-grade setup
(grade A drifted by ``shift_a``, grade B held at 0, 6 batches of n=1500 per
grade) and reports the share of runs whose *last* step's per-grade CI contains
the true per-grade shift. ``onset_localization`` mirrors
``tests/test_monitor_onset.py::test_onset_localizes_injected_drift`` (drift
injected at batch 12 of 24, n=2000) and reports the median and IQR of
``|onset - 12|``; the ``shift=0.4`` row is weaker drift than the CI gate's
``shift=0.6`` and is reported, not gated.
"""

import sys

import numpy as np

from probcal._math import expit, logit
from probcal.datasets import make_pd_portfolio
from probcal.monitor import CalibrationMonitor


def _grade_batch(
    n: int = 1500, shift_a: float = 0.0, shift_b: float = 0.0, seed: int = 0
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    d_a = make_pd_portfolio(n=n, random_state=seed)
    d_b = make_pd_portfolio(n=n, random_state=seed + 1)
    p = np.concatenate([d_a.scores, d_b.scores])
    true = np.concatenate([expit(logit(d_a.scores) + shift_a), expit(logit(d_b.scores) + shift_b)])
    rng = np.random.default_rng(seed + 5000)
    y = (rng.random(len(p)) < true).astype(float)
    g = np.array(["A"] * n + ["B"] * n)
    return y, p, g


def grade_cs_coverage(runs: int = 100, shift_a: float = 0.6, shift_b: float = 0.0) -> dict:
    """Share of runs whose last-step per-grade CI covers the true per-grade shift."""
    hits_a = hits_b = 0
    for r in range(runs):
        mon = CalibrationMonitor(alpha=0.05, delta_ci_grid=(-2.0, 2.0, 81))
        step = None
        for k in range(6):
            y, p, g = _grade_batch(n=1500, shift_a=shift_a, shift_b=shift_b, seed=1000 * r + k)
            step = mon.update(y, p, grade=g, label=f"m{k}")
        lo_a, hi_a = step.grade_delta_ci["A"]
        lo_b, hi_b = step.grade_delta_ci["B"]
        hits_a += lo_a <= shift_a <= hi_a
        hits_b += lo_b <= shift_b <= hi_b
    return {"coverage_a": hits_a / runs, "coverage_b": hits_b / runs}


def _scores(n: int, seed: int = 42) -> np.ndarray:
    return make_pd_portfolio(n=n, event_rate=0.05, random_state=seed).scores


def onset_localization(runs: int = 100, shift: float = 0.6, onset_idx: int = 12) -> dict:
    """Median and IQR of |onset - onset_idx| over `runs` seeded drift injections."""
    z = logit(_scores(2000, seed=42))
    p = expit(z)
    errors = []
    for seed in range(runs):
        mon = CalibrationMonitor(delta_ci_grid=(-2.0, 2.0, 41))
        rng = np.random.default_rng(seed)
        for k in range(24):
            p_true = expit(z + shift) if k >= onset_idx else p
            y = (rng.random(2000) < p_true).astype(float)
            mon.update(y, p, label=f"m{k}")
        rep = mon.report()
        if rep.alarm_at is None:
            continue
        errors.append(abs(int(rep.onset_label[1:]) - onset_idx))
    errors = np.array(errors, dtype=float)
    q1, q3 = np.percentile(errors, [25, 75])
    return {
        "n_alarmed": len(errors),
        "median": float(np.median(errors)),
        "iqr": (float(q1), float(q3)),
    }


def main() -> None:
    fast = "--fast" in sys.argv
    runs = 20 if fast else 100
    rows = []

    cov = grade_cs_coverage(runs=runs)
    rows.append(
        (
            f"per-grade CS coverage, drifted grade (shift=0.6, n={runs})",
            f"{cov['coverage_a']:.4f}",
            ">= 0.9",
        )
    )
    rows.append(
        (
            f"per-grade CS coverage, stable grade (shift=0.0, n={runs})",
            f"{cov['coverage_b']:.4f}",
            ">= 0.9",
        )
    )

    onset_06 = onset_localization(runs=runs, shift=0.6)
    iqr_06 = f"IQR [{onset_06['iqr'][0]:.1f}, {onset_06['iqr'][1]:.1f}]"
    rows.append(
        (
            f"onset |onset-12| (shift=0.6, n={runs})",
            f"median {onset_06['median']:.1f}, {iqr_06}",
            "<= 2 (median)",
        )
    )
    onset_04 = onset_localization(runs=runs, shift=0.4)
    iqr_04 = f"IQR [{onset_04['iqr'][0]:.1f}, {onset_04['iqr'][1]:.1f}]"
    rows.append(
        (
            f"onset |onset-12| (shift=0.4, n={runs})",
            f"median {onset_04['median']:.1f}, {iqr_04}",
            "reported",
        )
    )

    width = max(len(r[0]) for r in rows)
    print(f"| {'experiment'.ljust(width)} | result | gate |")
    print(f"|{'-' * (width + 2)}|--------|------|")
    for name, res, gate in rows:
        print(f"| {name.ljust(width)} | {res} | {gate} |")


if __name__ == "__main__":
    main()
