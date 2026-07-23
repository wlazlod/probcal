"""Tests for probcal.attribution."""

import numpy as np
import pytest

from probcal._math import expit, logit
from probcal.attribution import AdjustedAttribution, adjust_attributions
from probcal.isotonic import IsotonicCalibrator
from probcal.offset import LogitOffset
from probcal.parametric import BetaCalibrator, PlattCalibrator, TemperatureCalibrator

RNG = np.random.default_rng(101)


def _shap_setup(n: int = 200, d: int = 5) -> tuple[np.ndarray, float, np.ndarray]:
    """Synthetic SHAP rows on the logit (margin) scale: z = base + sum(phi)."""
    phi = RNG.normal(scale=0.6, size=(n, d))
    base = -2.0
    z = base + phi.sum(axis=1)
    return phi, base, z


def _fitted_platt() -> PlattCalibrator:
    s = expit(RNG.normal(-1.0, 1.4, 6000))
    y = (RNG.random(6000) < expit(0.8 * logit(s) - 0.4)).astype(float)
    return PlattCalibrator().fit(s, y)


def _fitted(cal_cls) -> object:
    s = expit(RNG.normal(-1.0, 1.4, 6000))
    y = (RNG.random(6000) < expit(0.8 * logit(s) - 0.4)).astype(float)
    return cal_cls().fit(s, y)


def test_affine_exact_platt_hand_computed() -> None:
    phi, base, z = _shap_setup()
    cal = _fitted_platt()
    a, b = cal.affine_logit_coeffs_
    res = adjust_attributions(phi, base, cal)
    assert isinstance(res, AdjustedAttribution)
    assert res.method_used == "affine-exact"
    np.testing.assert_allclose(res.phi_adj, a * phi, atol=1e-14)
    np.testing.assert_allclose(res.base_adj, np.full(len(z), a * base + b), atol=1e-14)
    np.testing.assert_allclose(res.target, a * z + b, atol=1e-12)


def test_affine_equals_aumann_shapley_for_affine_calibrators() -> None:
    phi, base, _ = _shap_setup()
    for cal in (_fitted(TemperatureCalibrator), _fitted_platt()):
        res_aff = adjust_attributions(phi, base, cal, method="affine")
        res_as = adjust_attributions(phi, base, cal, method="aumann-shapley")
        np.testing.assert_allclose(res_aff.phi_adj, res_as.phi_adj, atol=1e-12)
        np.testing.assert_allclose(res_aff.base_adj, res_as.base_adj, atol=1e-12)


def test_offset_is_affine_stage() -> None:
    phi, base, z = _shap_setup()
    off = LogitOffset(delta=-0.35).fit(expit(z))
    res = adjust_attributions(phi, base, off)
    assert res.method_used == "affine-exact"
    np.testing.assert_allclose(res.phi_adj, phi, atol=1e-14)  # a = 1
    np.testing.assert_allclose(res.base_adj, np.full(len(z), base - 0.35), atol=1e-14)


def test_reconstruction_for_nonaffine_calibrators() -> None:
    phi, base, z = _shap_setup()
    for cal in (_fitted(BetaCalibrator), _fitted(IsotonicCalibrator)):
        res = adjust_attributions(phi, base, cal)
        assert res.method_used == "aumann-shapley"
        recon = res.base_adj + res.phi_adj.sum(axis=1)
        np.testing.assert_allclose(recon, res.target, atol=1e-10)
        assert res.max_reconstruction_error < 1e-10
        expected_target = logit(cal.predict_proba(expit(z)))
        np.testing.assert_allclose(res.target, expected_target, atol=1e-10)


def test_sign_and_ranking_invariance_for_monotone_g() -> None:
    phi, base, _ = _shap_setup()
    cal = _fitted(BetaCalibrator)
    res = adjust_attributions(phi, base, cal)
    # Positive row multiplier: signs preserved, within-row |phi| order preserved.
    assert np.all(np.sign(res.phi_adj) == np.sign(phi))
    for i in range(len(phi)):
        np.testing.assert_array_equal(
            np.argsort(np.abs(res.phi_adj[i])), np.argsort(np.abs(phi[i]))
        )


def test_degenerate_row_falls_back_to_local_slope() -> None:
    cal = _fitted(BetaCalibrator)
    phi = np.zeros((1, 4))
    phi[0, 0] = 1e-12  # s == s0 to machine precision
    res = adjust_attributions(phi, -1.5, cal)
    assert np.all(np.isfinite(res.phi_adj))
    assert res.max_reconstruction_error < 1e-8


def test_probability_scale_reconstruction() -> None:
    cal = _fitted(BetaCalibrator)
    n, d = 100, 3
    phi = RNG.normal(scale=0.05, size=(n, d))
    base = 0.3
    s = np.clip(base + phi.sum(axis=1), 0.01, 0.99)
    phi[:, 0] -= (base + phi.sum(axis=1)) - s  # keep rows inside (0,1)
    res = adjust_attributions(phi, base, cal, scale="probability")
    recon = res.base_adj + res.phi_adj.sum(axis=1)
    np.testing.assert_allclose(recon, res.target, atol=1e-10)
    np.testing.assert_allclose(res.target, cal.predict_proba(base + phi.sum(axis=1)))


def test_ducktyped_explanation_object() -> None:
    phi, base, _ = _shap_setup(50)

    class FakeExplanation:
        def __init__(self) -> None:
            self.values = phi
            self.base_values = np.full(50, base)

    res = adjust_attributions(FakeExplanation(), None, _fitted_platt())
    assert res.phi_adj.shape == (50, 5)


def test_forcing_affine_on_nonaffine_raises() -> None:
    phi, base, _ = _shap_setup(20)
    with pytest.raises(ValueError, match="affine"):
        adjust_attributions(phi, base, _fitted(BetaCalibrator), method="affine")


def test_export() -> None:
    import probcal

    assert "adjust_attributions" in probcal.__all__
