"""Equivalence gate for the W8 ENIR vectorized path rewrite (task-8b-brief.md): the
new ``ENIRCalibrator._path_fit`` must reproduce the frozen v0.1.2 reference
(``tests/_enir_reference.py``) on ``path_lambdas_``, ``path_solutions_`` and
predictions.

The engine is an elementwise-vectorized transcription of the reference: it keeps the
reference's per-breakpoint global recompute of every pair's violation flag, so the
merge sequence is identical by construction and the only admitted differences are
floating-point summation grouping in the BIC log-likelihood (relative ~1e-15) and
breakpoints whose BIC weight the engine prunes to exactly 0 (reference weight below
1e-15 relative). Any structural divergence — different breakpoint count, different
merge order, different level count — is a bug, not a tolerance question.
"""

import numpy as np
import pytest

from _enir_reference import reference_aggregate, reference_path, reference_weights
from probcal._validation import EPS, validate_scores, validate_weights
from probcal.bayesian import ENIRCalibrator

PROBE = np.linspace(0.01, 0.99, 251)

# 20 seeded datasets: m in {50 x 7, 200 x 7, 2000 x 6}, cycling four dataset kinds.
_SIZES = [50] * 7 + [200] * 7 + [2000] * 6
CASES = [(m, i % 4, i) for i, m in enumerate(_SIZES)]


def _make_dataset(m: int, kind: int, seed: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Four dataset kinds: unique scores / aggregated ties with fractional ``y_u`` /
    non-uniform weights / near-perfectly-sorted labels with ~2% flips."""
    rng = np.random.default_rng(seed)
    if kind == 0:
        s = rng.uniform(0.02, 0.98, m)
        y = (rng.random(m) < s).astype(float)
        w = np.ones(m)
    elif kind == 1:
        grid = np.linspace(0.02, 0.98, m)
        s = rng.choice(grid, size=4 * m)
        y = (rng.random(4 * m) < s).astype(float)
        w = np.ones(4 * m)
    elif kind == 2:
        s = rng.uniform(0.02, 0.98, m)
        y = (rng.random(m) < s).astype(float)
        w = rng.uniform(0.2, 5.0, m)
    else:
        s = np.sort(rng.uniform(0.02, 0.98, m))
        y = (s > 0.5).astype(float)
        # A fixed ~2% flip count, not a per-point 2% chance: at m=50 the latter
        # draws zero flips often enough to leave the dataset already isotonic,
        # which turns the case into a no-op (path length 1).
        flip = rng.choice(m, size=max(2, round(0.02 * m)), replace=False)
        y[flip] = 1.0 - y[flip]
        w = np.ones(m)
    return s, y, w


def _reference_fit(
    s: np.ndarray, y: np.ndarray, w: np.ndarray
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Run the frozen v0.1.2 path on the same validated inputs the calibrator sees."""
    s_v = validate_scores(s)
    w_v = validate_weights(w, len(s))
    s_u, y_u, w_u = reference_aggregate(s_v, y, w_v)
    lambdas, solutions = reference_path(y_u, w_u)
    weights = reference_weights(y_u, w_u, solutions)
    return s_u, np.asarray(lambdas), np.asarray(solutions), weights


def _reference_predict(
    s_u: np.ndarray, solutions: np.ndarray, weights: np.ndarray, probe: np.ndarray
) -> np.ndarray:
    """Verbatim copy of the v0.1.2 ``ENIRCalibrator._predict``."""
    p = validate_scores(probe)
    idx = np.clip(np.searchsorted(s_u, p, side="right") - 1, 0, len(s_u) - 1)
    out = np.zeros(len(p))
    for weight, sol in zip(weights, solutions, strict=True):
        out += weight * sol[idx]
    return np.clip(out, EPS, 1.0 - EPS)


@pytest.mark.parametrize(
    ("m", "kind", "seed"),
    CASES,
    ids=[f"m{m}-kind{kind}-seed{seed}" for m, kind, seed in CASES],
)
def test_path_matches_v012_reference(m: int, kind: int, seed: int) -> None:
    s, y, w = _make_dataset(m, kind, seed)
    s_u, ref_lambdas, ref_solutions, ref_weights = _reference_fit(s, y, w)
    ref_pred = _reference_predict(s_u, ref_solutions, ref_weights, PROBE)

    cal = ENIRCalibrator(max_solutions=None).fit(s, y, w)
    np.testing.assert_allclose(cal._x, s_u)
    np.testing.assert_allclose(cal.path_lambdas_, ref_lambdas, rtol=1e-9, atol=1e-12)
    np.testing.assert_allclose(cal.path_solutions_, ref_solutions, rtol=0, atol=1e-10)
    np.testing.assert_allclose(cal.predict_proba(PROBE), ref_pred, rtol=0, atol=1e-10)

    bounded = ENIRCalibrator().fit(s, y, w)
    np.testing.assert_allclose(bounded.path_lambdas_, ref_lambdas, rtol=1e-9, atol=1e-12)
    np.testing.assert_allclose(bounded.predict_proba(PROBE), ref_pred, rtol=0, atol=1e-9)
    assert bounded.dropped_weight_ < 1e-6


def test_unit_total_weight_matches_reference() -> None:
    """Weights summing to exactly 1 make ``log(n_tot)`` exactly 0, which the BIC
    pruning bound divides by. Pruning must switch off and the fit must still
    reproduce the frozen reference."""
    s, y, _ = _make_dataset(64, 0, 0)
    w = np.full(64, 2.0**-6)  # exact halves: sums to exactly 1.0
    assert float(w.sum()) == 1.0

    s_u, ref_lambdas, ref_solutions, ref_weights = _reference_fit(s, y, w)
    ref_pred = _reference_predict(s_u, ref_solutions, ref_weights, PROBE)

    cal = ENIRCalibrator(max_solutions=None).fit(s, y, w)
    np.testing.assert_allclose(cal.path_lambdas_, ref_lambdas, rtol=1e-9, atol=1e-12)
    np.testing.assert_allclose(cal.predict_proba(PROBE), ref_pred, rtol=0, atol=1e-10)
    np.testing.assert_allclose(cal.weights_.sum(), 1.0, atol=1e-12)


def test_sub_unit_total_weight_fits() -> None:
    """Weights summing to less than 1 make ``log(n_tot)`` negative, which would turn
    the pruning bound into "prune everything". The fit must stay valid."""
    s, y, _ = _make_dataset(64, 0, 0)
    w = np.full(64, 2.0**-10)  # sums to 0.0625

    cal = ENIRCalibrator().fit(s, y, w)
    assert np.all(np.isfinite(cal.weights_))
    np.testing.assert_allclose(cal.weights_.sum(), 1.0, atol=1e-12)
    assert cal.path_solutions_.shape[0] == len(cal.kept_breakpoints_) >= 1
    p = cal.predict_proba(PROBE)
    assert np.all(np.isfinite(p))
    assert np.all((p > 0.0) & (p < 1.0))
