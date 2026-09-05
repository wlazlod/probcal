"""Tests for probcal.Chain: calibrator + offsets composition."""

import numpy as np
import pytest

from probcal import (
    BetaCalibrator,
    CalibratedModel,
    Chain,
    IsotonicCalibrator,
    LogitOffset,
    PlattCalibrator,
    UnattainableTargetError,
    make_pd_portfolio,
)
from probcal._math import expit

_D = make_pd_portfolio(n=2000, random_state=13)
_Q = make_pd_portfolio(n=400, random_state=14).scores


def _chain(cal_cls=BetaCalibrator, deltas=(0.25, -0.1)):
    cal = cal_cls().fit(_D.scores, _D.y)
    offs = [LogitOffset(delta=d).fit(cal.predict_proba(_D.scores)) for d in deltas]
    return Chain([cal, *offs]), cal, offs


def test_forward_composition_is_stagewise() -> None:
    chain, cal, offs = _chain()
    expected = cal.predict_proba(_Q)
    for off in offs:
        expected = off.transform(expected)
    np.testing.assert_array_equal(chain.predict_proba(_Q), expected)


def test_point_inverse_round_trips_through_all_stages() -> None:
    chain, _, _ = _chain()
    p = np.linspace(0.03, 0.6, 15)
    s = chain.point_inverse(p)
    np.testing.assert_allclose(chain.predict_proba(s), p, atol=1e-9)
    z = chain.point_inverse(p, space="logit")
    np.testing.assert_allclose(expit(z), s, atol=1e-12)


def test_interval_inverse_matches_point_on_degenerate_interval() -> None:
    chain, _, _ = _chain()
    lo_s, hi_s = chain.interval_inverse(0.05, 0.05)
    s_pt = float(chain.point_inverse(np.array([0.05]))[0])
    assert lo_s == pytest.approx(s_pt, abs=1e-6) and hi_s == pytest.approx(s_pt, abs=1e-6)
    lo_z, hi_z = chain.interval_inverse(0.0, 0.02, space="logit")
    assert np.isneginf(lo_z) and np.isfinite(hi_z)
    with pytest.raises(UnattainableTargetError, match="buffer"):
        chain.interval_inverse(0.049, 0.05, buffer_logit=2.0)


def test_affine_coeffs_compose_or_none() -> None:
    chain_platt, cal, offs = _chain(cal_cls=PlattCalibrator)
    a, b = cal.affine_logit_coeffs_
    total = sum(o.delta_ for o in offs)
    got = chain_platt.affine_logit_coeffs_
    assert got == pytest.approx((a, b + total))
    chain_beta, _, _ = _chain(cal_cls=BetaCalibrator)  # abm: not affine
    assert chain_beta.affine_logit_coeffs_ is None


def test_step_calibrator_chain_is_monotone_and_invertible() -> None:
    chain, _, _ = _chain(cal_cls=IsotonicCalibrator, deltas=(0.3,))
    assert chain.is_monotone_ is True
    lo, hi = chain.interval_inverse(0.02, 0.10)
    p = chain.predict_proba(np.array([np.clip(lo, 1e-9, 1 - 1e-9)]))
    assert p[0] >= 0.02 - 1e-12


def test_interpret_concatenates_stages() -> None:
    chain, _, _ = _chain(deltas=(0.4,))
    text = repr(chain.interpret())
    assert "BetaCalibrator" in text and "delta" in text


def test_chain_validation() -> None:
    cal = BetaCalibrator().fit(_D.scores, _D.y)
    off = LogitOffset(delta=0.1).fit(_D.scores)
    with pytest.raises(ValueError, match="first"):
        Chain([off, cal])
    with pytest.raises(ValueError, match="LogitOffset"):
        Chain([cal, cal])
    with pytest.raises(ValueError, match="at least"):
        Chain([])
    # Unfitted stages are legal at construction now; the not-fitted error
    # only surfaces from the reading methods (see the tests below).
    unfitted = Chain([BetaCalibrator()])
    assert unfitted.fitted_ is False
    with pytest.raises(RuntimeError, match="not fitted; call fit\\(\\) first"):
        unfitted.predict_proba(_D.scores)


def test_unfitted_chain_fits_like_the_manual_two_step() -> None:
    chain = Chain([BetaCalibrator(), LogitOffset(target_mean=0.03)]).fit(_D.scores, _D.y)
    cal = BetaCalibrator().fit(_D.scores, _D.y)
    off = LogitOffset(target_mean=0.03).fit(cal.predict_proba(_D.scores))
    manual = Chain([cal, off])
    np.testing.assert_array_equal(chain.predict_proba(_Q), manual.predict_proba(_Q))


def test_unfitted_chain_methods_raise_the_standard_not_fitted_error() -> None:
    chain = Chain([BetaCalibrator(), LogitOffset(delta=0.1)])
    assert chain.fitted_ is False
    assert chain.__sklearn_is_fitted__() is False
    with pytest.raises(RuntimeError, match=r"not fitted; call fit\(\) first"):
        chain.predict_proba(_D.scores)
    with pytest.raises(RuntimeError, match="not fitted"):
        chain.to_dict()
    with pytest.raises(RuntimeError, match="not fitted"):
        chain.interval_inverse(0.1, 0.2)


def test_constructor_still_validates_types_and_order() -> None:
    with pytest.raises(ValueError, match="first stage must be a calibrator"):
        Chain([LogitOffset(delta=0.1)])
    with pytest.raises(ValueError, match="must be a LogitOffset"):
        Chain([BetaCalibrator(), BetaCalibrator()])


def test_fit_refits_already_fitted_stages() -> None:
    cal = BetaCalibrator().fit(_D.scores, _D.y)
    before = cal.fingerprint()
    _D2 = make_pd_portfolio(n=2000, random_state=15)
    Chain([cal, LogitOffset(delta=0.1)]).fit(_D2.scores, _D2.y)  # different data
    assert cal.fingerprint() != before


def test_chain_serialization_round_trip() -> None:
    chain, _, _ = _chain()
    loaded = Chain.from_json(chain.to_json())
    np.testing.assert_array_equal(chain.predict_proba(_Q), loaded.predict_proba(_Q))
    assert chain.fingerprint() == loaded.fingerprint()


def test_calibrated_model_chain_property() -> None:
    class _Stub:
        def predict_proba(self, X):
            s = np.asarray(X)[:, 0]
            return np.column_stack([1.0 - s, s])

    wrapped = CalibratedModel(_Stub(), BetaCalibrator(), flow="prefit").fit(
        _D.scores.reshape(-1, 1), _D.y
    )
    wrapped.offset_to(delta=0.2)
    chain = wrapped.chain_
    np.testing.assert_array_equal(chain.predict_proba(_Q), wrapped.predict_proba(_Q.reshape(-1, 1)))
    # And the raw-margin logit contract matches the wrapper's own inverse.
    np.testing.assert_allclose(
        chain.interval_inverse(0.0, 0.05, space="logit"),
        wrapped.interval_inverse(0.0, 0.05, space="logit"),
        atol=1e-9,
    )
