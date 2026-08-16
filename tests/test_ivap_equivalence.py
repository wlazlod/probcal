"""Equivalence gate for the W9 IVAP precomputation (task-10-brief.md): the fitted
``F0_``/``F1_`` tables read through ``predict_interval`` must reproduce the frozen
v0.1.2 brute-force pair refit (``tests/_ivap_reference.py``) exactly.

The gate defines correctness. Both paths solve the same weighted isotonic problem, so
the only admitted difference is floating-point summation grouping; a discrepancy is a
bug in the sweep (hull maintenance, journal resurfacing, bridge walk, or indexing),
never a reason to loosen the tolerance.
"""

import numpy as np

from _ivap_reference import pair_at
from probcal._validation import validate_scores, validate_weights
from probcal.vennabers import VennAbersCalibrator


def _sorted_calibration(
    s: np.ndarray, y: np.ndarray, w: np.ndarray
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """The validated, stably sorted arrays the calibrator fits on."""
    s_v = validate_scores(s)
    w_v = validate_weights(w, len(s))
    order = np.argsort(s_v, kind="stable")
    return s_v[order], y[order], w_v[order]


def test_reference_matches_current_calibrator() -> None:
    """Sanity check that the frozen reference reproduces the shipped calibrator."""
    rng = np.random.default_rng(11)
    s = rng.uniform(0.02, 0.98, 120)
    y = (rng.random(120) < s).astype(float)
    w = rng.uniform(0.3, 4.0, 120)
    s_sorted, y_sorted, w_sorted = _sorted_calibration(s, y, w)

    queries = np.concatenate([rng.uniform(0.0, 1.0, 40), s[:10], [0.0, 1.0]])
    cal = VennAbersCalibrator().fit(s, y, w)
    got = cal.predict_interval(queries)
    expected = np.array(
        [pair_at(s_sorted, y_sorted, w_sorted, float(x)) for x in validate_scores(queries)]
    )
    np.testing.assert_allclose(got, expected, rtol=0, atol=1e-12)
