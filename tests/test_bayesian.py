"""Tests for probcal.bayesian: BBQ and ENIR."""

import numpy as np

from probcal._math import pava
from probcal.bayesian import BBQCalibrator, ENIRCalibrator

RNG = np.random.default_rng(31)
GRID = np.linspace(0.01, 0.99, 200)


def _sample(n: int = 3000) -> tuple[np.ndarray, np.ndarray]:
    s = RNG.uniform(0.01, 0.99, n)
    y = (RNG.random(n) < s).astype(float)
    return s, y


# ---------------------------------------------------------------- BBQ


def test_bbq_weights_normalized() -> None:
    cal = BBQCalibrator().fit(*_sample(800))
    np.testing.assert_allclose(cal.weights_.sum(), 1.0, atol=1e-12)
    assert len(cal.weights_) == len(cal.bins_grid_)


def test_bbq_tracks_identity_data() -> None:
    cal = BBQCalibrator().fit(*_sample(6000))
    p = cal.predict_proba(GRID)
    assert np.max(np.abs(p - GRID)) < 0.12


def test_bbq_interpret_top_models() -> None:
    cal = BBQCalibrator().fit(*_sample(800))
    interp = cal.interpret()
    assert any("top" in m.lower() or "weight" in m.lower() for m in interp.messages)


def test_bbq_predictions_in_unit_interval() -> None:
    cal = BBQCalibrator().fit(*_sample(400))
    p = cal.predict_proba(GRID)
    assert np.all((p > 0.0) & (p < 1.0))


# ---------------------------------------------------------------- ENIR


def test_enir_path_ends_at_isotonic_fit() -> None:
    s, y = _sample(300)
    cal = ENIRCalibrator().fit(s, y)
    # Aggregate ties the same way the calibrator does, then compare the final
    # path solution with plain isotonic regression.
    order = np.argsort(s, kind="stable")
    s_sorted, y_sorted = s[order], y[order]
    uniq, start = np.unique(s_sorted, return_index=True)
    counts = np.diff(np.append(start, len(s_sorted)))
    y_agg = np.add.reduceat(y_sorted, start) / counts
    iso = pava(y_agg, counts.astype(float)).fitted
    np.testing.assert_allclose(cal.path_solutions_[-1], iso, atol=1e-10)


def test_enir_path_starts_at_raw_data() -> None:
    s = np.array([0.1, 0.3, 0.5, 0.7, 0.9])
    y = np.array([0.0, 1.0, 0.0, 1.0, 1.0])
    cal = ENIRCalibrator().fit(s, y)
    np.testing.assert_allclose(cal.path_solutions_[0], y)
    assert cal.path_lambdas_[0] == 0.0


def test_enir_bic_weights_normalized() -> None:
    cal = ENIRCalibrator().fit(*_sample(400))
    np.testing.assert_allclose(cal.weights_.sum(), 1.0, atol=1e-12)


def test_enir_predictions_valid() -> None:
    cal = ENIRCalibrator().fit(*_sample(500))
    p = cal.predict_proba(GRID)
    assert np.all(np.isfinite(p))
    assert np.all((p > 0.0) & (p < 1.0))


def test_enir_not_monotone_flag() -> None:
    assert ENIRCalibrator.is_monotone_ is False


def test_enir_interpret_warns_nonmonotone() -> None:
    cal = ENIRCalibrator().fit(*_sample(300))
    interp = cal.interpret()
    assert any("monoton" in m.lower() for m in interp.messages)


def test_exports() -> None:
    import probcal

    for name in (
        "HistogramBinningCalibrator",
        "ScalingBinningCalibrator",
        "BBQCalibrator",
        "ENIRCalibrator",
    ):
        assert name in probcal.__all__
