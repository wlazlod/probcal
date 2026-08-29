import numpy as np

from probcal._math import expit
from probcal.metrics import brier_score, log_loss

RNG = np.random.default_rng(31)


def _calibrated(n=2000):
    p = expit(RNG.normal(-1.5, 1.0, n))
    y = (RNG.random(n) < p).astype(float)
    return y, p


def test_corp_identity_holds_for_both_scores():
    from probcal.curves import corp_reliability

    y, p = _calibrated()
    r = corp_reliability(y, p, bands=None)
    assert abs(r.brier - (r.brier_mcb - r.brier_dsc + r.brier_unc)) < 1e-12
    assert abs(r.log_loss - (r.log_loss_mcb - r.log_loss_dsc + r.log_loss_unc)) < 1e-12
    assert abs(r.brier - brier_score(y, p)) < 1e-12
    assert abs(r.log_loss - log_loss(y, p)) < 1e-12
    assert r.brier_mcb >= -1e-12 and r.brier_dsc >= -1e-12


def test_corp_hand_fixture_blocks_and_decomposition():
    from probcal.curves import corp_reliability

    # 8 points, sorted p; PAV pools (0.3,0.4), then cascades (0.5,0.6,0.7)
    # into one block (the isolated 1.0 at p=0.5 is itself a violator against
    # the 0.5-level block that follows at p=0.6,0.7) -- verified against
    # sklearn.isotonic.IsotonicRegression.
    p = np.array([0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.9])
    y = np.array([0.0, 0.0, 1.0, 0.0, 1.0, 1.0, 0.0, 1.0])
    r = corp_reliability(y, p, bands=None)
    np.testing.assert_allclose(r.block_level, [0.0, 0.0, 0.5, 2 / 3, 1.0])
    np.testing.assert_allclose(r.block_lo, [0.1, 0.2, 0.3, 0.5, 0.9])
    np.testing.assert_allclose(r.block_hi, [0.1, 0.2, 0.4, 0.7, 0.9])
    ybar = y.mean()
    unc = ybar * (1 - ybar)
    pav = np.array([0.0, 0.0, 0.5, 0.5, 2 / 3, 2 / 3, 2 / 3, 1.0])
    mcb = np.mean((y - p) ** 2) - np.mean((y - pav) ** 2)
    dsc = unc - np.mean((y - pav) ** 2)
    assert abs(r.brier_unc - unc) < 1e-12
    assert abs(r.brier_mcb - mcb) < 1e-12
    assert abs(r.brier_dsc - dsc) < 1e-12


def test_corp_pav_is_in_input_order_and_weighted():
    from probcal.curves import corp_reliability

    y, p = _calibrated(500)
    perm = RNG.permutation(500)
    r1 = corp_reliability(y, p, bands=None)
    r2 = corp_reliability(y[perm], p[perm], bands=None)
    np.testing.assert_allclose(r2.pav, r1.pav[perm])
    w = np.where(y == 1, 2.0, 1.0)
    r3 = corp_reliability(y, p, sample_weight=w, bands=None)
    # duplicating events equals weighting them by 2
    yy = np.concatenate([y, y[y == 1]])
    pp = np.concatenate([p, p[y == 1]])
    r4 = corp_reliability(yy, pp, bands=None)
    np.testing.assert_allclose(r3.block_level, r4.block_level)
    assert abs(r3.brier_mcb - r4.brier_mcb) < 1e-12


def test_corp_log_loss_degenerate_levels_are_clipped():
    from probcal.curves import corp_reliability

    p = np.array([0.1, 0.2, 0.8, 0.9])
    y = np.array([0.0, 0.0, 1.0, 1.0])
    r = corp_reliability(y, p, bands=None)  # PAV levels are exactly 0 and 1
    assert np.isfinite(r.log_loss_mcb) and np.isfinite(r.log_loss_dsc)
