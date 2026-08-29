"""Tests for probcal.metrics.binned."""

import numpy as np
import pytest

from probcal._math import expit
from probcal.datasets import make_pd_portfolio
from probcal.metrics.binned import (
    _bin_gaps,
    _ece_sweep_best_b_sorted,
    _ece_sweep_presorted,
    adaptive_ece,
    ece,
    ece_debiased,
    ece_sweep,
    hosmer_lemeshow,
)

RNG = np.random.default_rng(59)


def _calibrated(n: int = 6000) -> tuple[np.ndarray, np.ndarray]:
    p = expit(RNG.normal(-0.8, 1.2, n))
    y = (RNG.random(n) < p).astype(float)
    return y, p


def test_ece_two_bin_hand_case() -> None:
    p = np.array([0.1, 0.1, 0.9, 0.9])
    y = np.array([0.0, 1.0, 1.0, 1.0])
    # Bin 1: mean p 0.1, rate 0.5, gap 0.4; bin 2: mean p 0.9, rate 1.0, gap 0.1.
    assert abs(ece(y, p, n_bins=2) - (0.4 + 0.1) / 2) < 1e-12
    assert abs(ece(y, p, n_bins=2, norm="max") - 0.4) < 1e-12
    expected_l2 = np.sqrt((0.4**2 + 0.1**2) / 2)
    assert abs(ece(y, p, n_bins=2, norm="l2") - expected_l2) < 1e-12


def test_adaptive_ece_is_equal_mass_alias() -> None:
    y, p = _calibrated(1000)
    assert adaptive_ece(y, p, n_bins=10) == ece(y, p, n_bins=10, strategy="mass")


def test_ece_debiased_below_plain_and_near_zero_when_calibrated() -> None:
    y, p = _calibrated(8000)
    plain = ece(y, p, n_bins=15)
    debiased = ece_debiased(y, p, n_bins=15)
    assert debiased <= plain
    assert debiased < 0.02


def test_ece_positive_bias_on_calibrated_data() -> None:
    # A perfectly calibrated model still shows positive plain ECE (the bias).
    y, p = _calibrated(2000)
    assert ece(y, p, n_bins=30) > 0.0


def test_ece_sweep_returns_reasonable_value() -> None:
    y, p = _calibrated(4000)
    v = ece_sweep(y, p)
    assert 0.0 <= v < 0.1


def test_hosmer_lemeshow_result_fields() -> None:
    y, p = _calibrated(2000)
    res = hosmer_lemeshow(y, p, g=10)
    assert res.df == 8
    assert res.statistic >= 0.0
    assert 0.0 <= res.p_value <= 1.0


def test_hosmer_lemeshow_rejects_gross_miscalibration() -> None:
    y, p = _calibrated(5000)
    p_bad = np.clip(p * 0.3, 1e-6, 1 - 1e-6)
    assert hosmer_lemeshow(y, p_bad).p_value < 0.001


@pytest.mark.reference
def test_hl_pvalue_vs_scipy_chi2() -> None:
    stats = pytest.importorskip("scipy.stats")
    y, p = _calibrated(2000)
    res = hosmer_lemeshow(y, p, g=10)
    expected = float(stats.chi2.sf(res.statistic, res.df))
    assert abs(res.p_value - expected) < 1e-9


# --------------------------------------------------- vectorized ece_sweep scan (0.3.0)


def _ece_sweep_best_b_reference(y: np.ndarray, p: np.ndarray, w: np.ndarray) -> int:
    """The pre-0.3.0 ``ece_sweep`` scan loop, kept verbatim as the oracle.

    ``_ece_sweep_best_b_sorted`` replaces this bin-index/bincount loop with a
    presorted cut-position scan; only ``best_b`` may be decided differently,
    and this reference pins that it is not.
    """
    best_b = 1
    for b in range(2, min(len(p), 100) + 1):
        _, _, rates, _ = _bin_gaps(y, p, w, b, "mass")
        if np.all(np.diff(rates) >= 0.0):
            best_b = b
    return best_b


def _ece_sweep_reference(y: np.ndarray, p: np.ndarray, w: np.ndarray, norm: str = "l1") -> float:
    """The pre-0.3.0 ``ece_sweep`` body, verbatim, over prepped arrays."""
    best_b = _ece_sweep_best_b_reference(y, p, w)
    if best_b == 1:
        return abs(float(np.average(p, weights=w)) - float(np.average(y, weights=w)))
    return ece(y, p, n_bins=best_b, strategy="mass", norm=norm, sample_weight=w)


def _sweep_fixtures() -> list[object]:
    """(y, p, w) cases: two portfolio sizes, heavy score ties, non-uniform weights."""
    out = []
    for n in (500, 5000):
        d = make_pd_portfolio(n=n, random_state=3)
        out.append(pytest.param(d.y, d.scores, np.ones(n), id=f"portfolio-{n}"))
    d = make_pd_portfolio(n=2000, random_state=4)
    out.append(pytest.param(d.y, np.round(d.scores, 2), np.ones(2000), id="tied-scores"))
    rng = np.random.default_rng(17)
    out.append(pytest.param(d.y, d.scores, rng.uniform(0.5, 2.0, 2000), id="weighted"))
    return out


@pytest.mark.parametrize("y,p,w", _sweep_fixtures())
def test_sorted_sweep_scan_picks_the_same_best_b(
    y: np.ndarray, p: np.ndarray, w: np.ndarray
) -> None:
    order = np.argsort(p, kind="stable")
    assert _ece_sweep_best_b_sorted(p[order], y[order], w[order]) == _ece_sweep_best_b_reference(
        y, p, w
    )


@pytest.mark.parametrize("y,p,w", _sweep_fixtures())
def test_ece_sweep_value_is_bit_identical_to_the_old_body(
    y: np.ndarray, p: np.ndarray, w: np.ndarray
) -> None:
    assert ece_sweep(y, p, sample_weight=w) == _ece_sweep_reference(y, p, w)
    assert ece_sweep(y, p, norm="max", sample_weight=w) == _ece_sweep_reference(y, p, w, "max")


@pytest.mark.parametrize("y,p,w", _sweep_fixtures())
def test_ece_sweep_presorted_matches_the_public_value(
    y: np.ndarray, p: np.ndarray, w: np.ndarray
) -> None:
    # The presorted variant reaches the same ``best_b`` and the same unchanged
    # final ``ece`` call; the remaining difference is bincount summation order
    # over the reordered rows (~1e-16), which is why this is not ``==``.
    order = np.argsort(p, kind="stable")
    got = _ece_sweep_presorted(y[order], p[order], w[order])
    assert got == pytest.approx(ece_sweep(y, p, sample_weight=w), rel=1e-12, abs=1e-15)
