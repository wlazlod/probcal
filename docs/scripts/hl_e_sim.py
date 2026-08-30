"""Type-I and power simulations for probcal.metrics.hl_e_test.

Usage: ``uv run python docs/scripts/hl_e_sim.py [--fast]``

``type1`` draws ``y ~ Bernoulli(p)`` under H0 (the assigned ``p`` is exactly
right) and checks the e-value's Ville/Markov tail bounds hold across
repeated draws: ``P(e >= 20) <= 0.05``, ``P(e >= 100) <= 0.01``, and
``mean(e) <= 1`` (an e-value's defining property under H0). ``power`` drifts
the true probability away from the assigned ``p`` by a logit-scale
``shift``/``slope`` and reports the detection rate at the conventional
``alpha=0.05`` threshold (``e >= 20``). ``tests/test_hl_e_sim.py`` enforces
reduced-size versions of the type-I gates in CI; the full-size table printed
here (``runs=2000``) is pasted into ``docs/concepts/monitoring.md``.

``_scores`` replicates ``docs/scripts/monitor_sim.py::_scores`` (the same
seeded ``make_pd_portfolio`` call used as a realistic, right-skewed score
vector) rather than importing that sibling docs script by path.
"""

import sys

import numpy as np

from probcal._math import expit, logit
from probcal.datasets import make_pd_portfolio
from probcal.metrics import hl_e_test


def _scores(n: int, seed: int = 42) -> np.ndarray:
    return make_pd_portfolio(n=n, event_rate=0.05, random_state=seed).scores


def _equal_mass_grades(p: np.ndarray, n_grades: int) -> np.ndarray:
    """Assign each score to one of ``n_grades`` equal-mass bins by rank."""
    order = np.argsort(p)
    bin_id = np.empty(len(p), dtype=np.int64)
    bin_id[order] = (np.arange(len(p)) * n_grades) // len(p)
    return np.array([f"G{i}" for i in bin_id])


def type1(runs: int = 2000, n: int = 2000, seed: int = 42, n_grades: int = 5) -> dict:
    """Type-I simulation: ``y ~ Bernoulli(p)`` under H0, fixed ``p``/grades.

    Parameters
    ----------
    runs : int
        Number of independent draws.
    n : int
        Portfolio size (fixed across runs; only ``y`` is redrawn).
    seed : int
        Seed for ``numpy.random.default_rng``.
    n_grades : int
        Number of equal-mass grades to bin ``p`` into.

    Returns
    -------
    dict
        ``p_ge_20`` / ``p_ge_100`` (fraction of runs with ``e_value >= 20``
        / ``>= 100``), ``mean_e`` and ``se_mean`` (mean e-value and its
        standard error across runs).
    """
    rng = np.random.default_rng(seed)
    p = _scores(n, seed=seed)
    grades = _equal_mass_grades(p, n_grades)
    e_values = np.empty(runs)
    for r in range(runs):
        y = (rng.random(n) < p).astype(np.float64)
        e_values[r] = hl_e_test(y, p, grades).e_value
    return {
        "p_ge_20": float(np.mean(e_values >= 20.0)),
        "p_ge_100": float(np.mean(e_values >= 100.0)),
        "mean_e": float(np.mean(e_values)),
        "se_mean": float(np.std(e_values, ddof=1) / np.sqrt(runs)),
    }


def power(
    shift: float = 0.0,
    slope: float = 1.0,
    runs: int = 2000,
    n: int = 2000,
    seed: int = 43,
    n_grades: int = 5,
    alpha: float = 0.05,
) -> dict:
    """Power simulation: true probability drifts from the assigned ``p``.

    Parameters
    ----------
    shift : float
        Logit-scale level shift: ``p_true = sigma(slope * z + shift)``.
    slope : float
        Logit-scale shape shift.
    runs : int
        Number of independent draws.
    n : int
        Portfolio size.
    seed : int
        Seed for ``numpy.random.default_rng``.
    n_grades : int
        Number of equal-mass grades.
    alpha : float
        Nominal level; detection is ``e_value >= 1 / alpha``.

    Returns
    -------
    dict
        ``detect_rate`` (fraction of runs flagged at ``e_value >= 1 /
        alpha``) and ``mean_e``.
    """
    rng = np.random.default_rng(seed)
    p = _scores(n, seed=seed)
    z = logit(p)
    p_true = expit(slope * z + shift)
    grades = _equal_mass_grades(p, n_grades)
    thr = 1.0 / alpha
    e_values = np.empty(runs)
    for r in range(runs):
        y = (rng.random(n) < p_true).astype(np.float64)
        e_values[r] = hl_e_test(y, p, grades).e_value
    return {
        "detect_rate": float(np.mean(e_values >= thr)),
        "mean_e": float(np.mean(e_values)),
    }


def main() -> None:
    fast = "--fast" in sys.argv
    runs = 300 if fast else 2000
    rows = []

    t1 = type1(runs=runs)
    bound_05 = 0.05 + 2.0 * np.sqrt(0.05 * 0.95 / runs)
    bound_01 = 0.01 + 2.0 * np.sqrt(0.01 * 0.99 / runs)
    rows.append(("type-I P(e >= 20) [alpha=0.05]", f"{t1['p_ge_20']:.4f}", f"<= {bound_05:.4f}"))
    rows.append(("type-I P(e >= 100) [alpha=0.01]", f"{t1['p_ge_100']:.4f}", f"<= {bound_01:.4f}"))
    rows.append(
        (
            "type-I mean(e)",
            f"{t1['mean_e']:.4f} (se={t1['se_mean']:.4f})",
            f"<= {1.0 + 3.0 * t1['se_mean']:.4f}",
        )
    )

    p_shift = power(shift=0.4, runs=runs)
    rows.append((f"power shift=0.4 (n={runs})", f"detect {p_shift['detect_rate']:.4f}", "reported"))
    p_slope = power(slope=0.8, runs=runs)
    rows.append((f"power slope=0.8 (n={runs})", f"detect {p_slope['detect_rate']:.4f}", "reported"))

    width = max(len(r[0]) for r in rows)
    print(f"| {'experiment'.ljust(width)} | result | gate |")
    print(f"|{'-' * (width + 2)}|--------|------|")
    for name, res, gate in rows:
        print(f"| {name.ljust(width)} | {res} | {gate} |")


if __name__ == "__main__":
    main()
