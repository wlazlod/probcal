"""Tests for probcal.integrations.optbinning. Skipped without optbinning."""

import numpy as np
import pytest

pytest.importorskip("optbinning")
pd = pytest.importorskip("pandas")

from sklearn.linear_model import LogisticRegression  # noqa: E402

from probcal import BetaCalibrator, PlattCalibrator, UnattainableTargetError  # noqa: E402
from probcal._math import expit, logit  # noqa: E402
from probcal.integrations.optbinning import CalibratedScorecard, calibrate_scorecard  # noqa: E402


def _scorecard(rounding: bool = False, n: int = 4000, seed: int = 0):
    from optbinning import BinningProcess, Scorecard

    rng = np.random.default_rng(seed)
    X = pd.DataFrame(
        {
            "a": rng.normal(size=n),
            "b": rng.normal(size=n),
            "c": rng.integers(0, 5, n).astype(float),
        }
    )
    z = 1.2 * X["a"] - 0.7 * X["b"] + 0.3 * X["c"] - 2.5
    y = (rng.random(n) < expit(z.to_numpy())).astype(int)
    sc = Scorecard(
        binning_process=BinningProcess(variable_names=list(X.columns)),
        estimator=LogisticRegression(),
        scaling_method="pdo_odds",
        scaling_method_params={"pdo": 20, "odds": 50, "scorecard_points": 600},
        rounding=rounding,
    ).fit(X, y)
    return sc, X, y


@pytest.fixture(scope="module")
def fitted():
    sc, X, y = _scorecard()
    X_cal, y_cal = X.iloc[2000:], y[2000:]
    return calibrate_scorecard(sc, X_cal, y_cal), sc, X_cal, y_cal


def test_predict_proba_is_calibrated_scorecard_output(fitted) -> None:
    cs, sc, X_cal, y_cal = fitted
    manual = BetaCalibrator().fit(sc.predict_proba(X_cal)[:, 1], y_cal)
    np.testing.assert_array_equal(
        cs.predict_proba(X_cal), manual.predict_proba(sc.predict_proba(X_cal)[:, 1])
    )
    assert isinstance(cs.calibrator_, BetaCalibrator)


def test_score_is_untouched(fitted) -> None:
    cs, sc, X_cal, _ = fitted
    np.testing.assert_array_equal(cs.score(X_cal), sc.score(X_cal))


def test_points_affine_coeffs_verified(fitted) -> None:
    cs, sc, X_cal, _ = fitted
    a_pts, b_pts = cs.points_affine_coeffs_
    z = logit(sc.predict_proba(X_cal)[:, 1])
    np.testing.assert_allclose(cs.score(X_cal), a_pts + b_pts * z, atol=1e-6)
    assert b_pts < 0  # pdo_odds scaling: higher points = safer


def test_masterscale_orders_point_cutoffs(fitted) -> None:
    cs, _, _, _ = fitted
    bands = {"A": (0.0, 0.01), "B": (0.01, 0.05), "C": (0.05, 1.0)}
    table = cs.masterscale(bands)
    assert set(table) == {"A", "B", "C"}
    # rising PD -> falling points; the A band reaches +inf points (PD -> 0)
    assert table["A"][1] == np.inf and table["C"][0] == -np.inf
    assert table["A"][0] >= table["B"][1] - 1e-6 >= table["B"][0] >= table["C"][1] - 1e-6
    # cut-off round trip: the B/C boundary in points maps back to PD 0.05
    a_pts, b_pts = cs.points_affine_coeffs_
    z_cut = (table["B"][0] - a_pts) / b_pts
    p_cut = cs.calibrator_.predict_proba(expit(np.array([z_cut])))
    np.testing.assert_allclose(p_cut, [0.05], atol=1e-6)


def test_protocol_delegation(fitted) -> None:
    cs, _, _, _ = fitted
    assert cs.is_monotone_ is True
    lo_z, hi_z = cs.interval_inverse(0.0, 0.02, space="logit")
    assert np.isneginf(lo_z) and np.isfinite(hi_z)
    s = cs.point_inverse(np.array([0.03]))
    np.testing.assert_allclose(cs.calibrator_.predict_proba(s), [0.03], atol=1e-9)
    with pytest.raises(UnattainableTargetError):
        cs.interval_inverse(0.02, 0.021, buffer_logit=3.0)


def test_rounding_scorecard_warns_and_refuses_masterscale() -> None:
    sc, X, y = _scorecard(rounding=True, n=2500, seed=3)
    with pytest.warns(UserWarning, match="not affine"):
        cs = calibrate_scorecard(sc, X, y)
    assert cs.points_affine_coeffs_ is None
    with pytest.raises(RuntimeError, match="interval_inverse"):
        cs.masterscale({"A": (0.0, 0.05), "B": (0.05, 1.0)})
    lo, hi = cs.interval_inverse(0.0, 0.05)  # the documented fallback still works
    assert 0.0 <= lo <= hi <= 1.0


def test_serialization_reattaches_and_verifies_scorecard(fitted) -> None:
    cs, sc, X_cal, _ = fitted
    d = cs.to_dict()
    assert d["class"] == "CalibratedScorecard"
    loaded = CalibratedScorecard.from_dict(d, scorecard=sc)
    np.testing.assert_array_equal(cs.predict_proba(X_cal), loaded.predict_proba(X_cal))
    assert cs.fingerprint() == loaded.fingerprint()
    other_sc, _, _ = _scorecard(n=2500, seed=9)
    with pytest.raises(ValueError, match="fingerprint"):
        CalibratedScorecard.from_dict(d, scorecard=other_sc)


def test_custom_calibrator_prototype(fitted) -> None:
    _, sc, X_cal, y_cal = fitted
    cs = calibrate_scorecard(sc, X_cal, y_cal, calibrator=PlattCalibrator())
    assert isinstance(cs.calibrator_, PlattCalibrator)
    assert cs.affine_logit_coeffs_ is not None
