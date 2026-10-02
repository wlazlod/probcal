"""Equivalence gate for the IVAP precomputation: the fitted
``F0_``/``F1_`` tables read through ``predict_interval`` must reproduce the
brute-force pair refit on the tie-pooled calibration set
(``tests/_ivap_reference.py``) exactly.

The gate defines correctness. Both paths solve the same weighted isotonic problem, so
the only admitted difference is floating-point summation grouping; a discrepancy is a
bug in the sweep (hull maintenance, journal resurfacing, bridge walk, or indexing),
never a reason to loosen the tolerance.
"""

import numpy as np
import pytest

from _ivap_reference import pair_at, pooled_calibration
from probcal._validation import EPS, validate_scores, validate_weights
from probcal.vennabers import VennAbersCalibrator

# 20 seeded calibration sets: n in {30 x 7, 500 x 7, 3000 x 6}; ties injected in half
# the cases, unit and non-uniform weights alternating independently of the tie flag.
_SIZES = [30] * 7 + [500] * 7 + [3000] * 6
CASES = [(n, i % 2 == 0, (i // 2) % 2 == 1, i) for i, n in enumerate(_SIZES)]


def _make_dataset(
    n: int, tied: bool, weighted: bool, seed: int
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    s = rng.uniform(0.02, 0.98, n)
    if tied:
        s = np.round(s, 2)
    y = (rng.random(n) < s).astype(float)
    w = rng.uniform(0.3, 4.0, n) if weighted else np.ones(n)
    return s, y, w


def _queries(s: np.ndarray, seed: int) -> np.ndarray:
    """Uniform draws, exact calibration scores (tie hits), the clipped boundaries
    ``0.0``/``1.0``/``EPS``/``1 - EPS``, and values outside the calibration range."""
    rng = np.random.default_rng(seed + 10_000)
    lo, hi = float(s.min()), float(s.max())
    return np.concatenate(
        [
            rng.uniform(0.0, 1.0, 200),
            rng.choice(s, size=50, replace=True),
            np.array([0.0, 1.0, EPS, 1.0 - EPS]),
            rng.uniform(0.0, lo, 5),
            rng.uniform(hi, 1.0, 5),
        ]
    )


def _pooled_calibration(
    s: np.ndarray, y: np.ndarray, w: np.ndarray
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """The validated arrays the calibrator fits on, ties pooled."""
    return pooled_calibration(validate_scores(s), y, validate_weights(w, len(s)))


@pytest.mark.parametrize(
    ("n", "tied", "weighted", "seed"),
    CASES,
    ids=[f"n{n}-tied{int(t)}-weighted{int(g)}-seed{s}" for n, t, g, s in CASES],
)
def test_interval_matches_v012_pair_refit(n: int, tied: bool, weighted: bool, seed: int) -> None:
    s, y, w = _make_dataset(n, tied, weighted, seed)
    s_u, y_u, w_u = _pooled_calibration(s, y, w)
    queries = _queries(s, seed)

    cal = VennAbersCalibrator().fit(s, y, w)
    got = cal.predict_interval(queries)
    expected = np.array([pair_at(s_u, y_u, w_u, float(x)) for x in validate_scores(queries)])
    np.testing.assert_allclose(got, expected, rtol=0, atol=1e-12)


@pytest.mark.parametrize("seed", range(5))
def test_interval_is_row_order_invariant_with_ties(seed: int) -> None:
    """0.3.x inserted the query among unpooled tied rows: permuting the rows
    moved the output. Pooled ties make it a function of the data set only."""
    s, y, w = _make_dataset(300, True, seed % 2 == 1, seed)
    queries = _queries(s, seed)
    perm = np.random.default_rng(seed).permutation(len(s))
    a = VennAbersCalibrator().fit(s, y, w).predict_interval(queries)
    b = VennAbersCalibrator().fit(s[perm], y[perm], w[perm]).predict_interval(queries)
    np.testing.assert_array_equal(a, b)
