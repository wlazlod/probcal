"""Tests for probcal.segmented.SegmentedCalibrator."""

import warnings

import numpy as np
import pytest

from probcal import SegmentedCalibrator
from probcal._math import expit, logit
from probcal.isotonic import IsotonicCalibrator
from probcal.parametric import BetaCalibrator, PlattCalibrator


def _pd_data(n: int, seed: int, event_shift: float = 0.0) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    s = expit(rng.normal(-1.0, 1.0, n))
    p_true = expit(logit(s) + event_shift)
    y = (rng.random(n) < p_true).astype(float)
    return s, y


def test_zero_arg_fit_is_a_single_global_segment_with_affine_coeffs() -> None:
    s, y = _pd_data(500, seed=6)
    cal = SegmentedCalibrator(base=PlattCalibrator()).fit(s, y)
    assert cal.segments_ == ("__all__",)
    assert cal.tau2_ == 0.0
    np.testing.assert_allclose(cal.delta_tilde_, [0.0])
    a, b = cal.affine_logit_coeffs_
    base_a, base_b = cal.base_.affine_logit_coeffs_
    assert a == base_a and b == base_b + cal.delta_tilde_[0]

    segments = np.array(["a", "b"])[np.arange(500) % 2]
    cal_multi = SegmentedCalibrator(base=PlattCalibrator()).fit(s, y, segments=segments)
    assert cal_multi.affine_logit_coeffs_ is None


def test_tau2_zero_gives_complete_pooling_and_segments_none_is_the_base_map() -> None:
    s, y = _pd_data(400, seed=0)
    segments = np.array(["a", "b", "c", "d"])[np.arange(400) % 4]
    cal = SegmentedCalibrator().fit(s, y, segments=segments)
    assert cal.tau2_ == 0.0
    np.testing.assert_allclose(cal.delta_tilde_, np.zeros(4), atol=1e-15)
    np.testing.assert_allclose(cal.shrink_, np.zeros(4), atol=1e-15)
    np.testing.assert_array_equal(cal.predict_proba(s), cal.base_.predict_proba(s))


def test_small_segment_is_strongly_shrunk_toward_large_segments() -> None:
    rng = np.random.default_rng(7)
    n_big, n_small = 2000, 20
    s_big1 = expit(rng.normal(-1.0, 1.0, n_big))
    s_big2 = expit(rng.normal(-1.0, 1.0, n_big))
    s_small = expit(rng.normal(-1.0, 1.0, n_small))

    def _draw(s: np.ndarray, delta: float) -> np.ndarray:
        p_true = expit(logit(s) + delta)
        return (rng.random(len(s)) < p_true).astype(float)

    y_big1, y_big2, y_small = _draw(s_big1, 0.6), _draw(s_big2, 0.6), _draw(s_small, 0.6)
    s = np.concatenate([s_big1, s_big2, s_small])
    y = np.concatenate([y_big1, y_big2, y_small])
    segments = np.array(["big1"] * n_big + ["big2"] * n_big + ["small"] * n_small)

    cal = SegmentedCalibrator().fit(s, y, segments=segments)
    shrink_by_segment = dict(zip(cal.segments_, cal.shrink_, strict=True))
    assert shrink_by_segment["small"] < 0.15
    # the two large, precise segments keep most of their own estimate
    assert shrink_by_segment["big1"] > 0.5
    assert shrink_by_segment["big2"] > 0.5


def test_single_class_segment_is_fully_shrunk() -> None:
    rng = np.random.default_rng(3)
    n = 600
    s = expit(rng.normal(-1.0, 1.0, n))
    y = (rng.random(n) < s).astype(float)
    segments = np.array(["normal"] * (n - 10) + ["all_negative"] * 10)
    y[-10:] = 0.0  # single-class segment
    cal = SegmentedCalibrator().fit(s, y, segments=segments)
    idx = cal.segments_.index("all_negative")
    assert cal.delta_hat_[idx] == 0.0
    assert np.isinf(cal.se_[idx])
    assert cal.shrink_[idx] == 0.0
    assert cal.delta_tilde_[idx] == 0.0


def test_unseen_labels_global_warns_only_when_nothing_matches_and_raise_raises() -> None:
    # Regression: int labels at fit ("0", "1" after str()) vs. float labels at
    # predict ("0.0", "1.0") never match, so every row falls back to the
    # global map under unseen="global" -- this must warn rather than fail silently.
    s, y = _pd_data(600, seed=10)
    int_segments = np.array([0, 1])[np.arange(600) % 2]
    cal = SegmentedCalibrator().fit(s, y, segments=int_segments)
    float_segments = np.array([0.0, 1.0])[np.arange(5) % 2]
    with pytest.warns(UserWarning, match="none of the 5 segment labels"):
        p_mismatch = cal.predict_proba(s[:5], segments=float_segments)
    np.testing.assert_allclose(p_mismatch, cal.base_.predict_proba(s[:5]))
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        cal.predict_proba(s[:5], segments=np.array([0, 7, 1, 7, 0]))  # some rows match

    strict = SegmentedCalibrator(unseen="raise").fit(s, y, segments=int_segments)
    with pytest.raises(ValueError, match="unseen"):
        strict.predict_proba(s[:5], segments=np.array(["z"] * 5))


def test_fit_validation() -> None:
    s, y = _pd_data(200, seed=0)
    with pytest.raises(ValueError, match="unseen"):
        SegmentedCalibrator(unseen="bogus").fit(s, y)
    with pytest.raises(ValueError, match="length"):
        SegmentedCalibrator().fit(s, y, segments=np.array(["a", "b"]))


def test_interval_and_point_inverse_with_segment_kwarg() -> None:
    s, y = _pd_data(700, seed=8)
    segments = np.array(["a", "b"])[np.arange(700) % 2]
    cal = SegmentedCalibrator(base=BetaCalibrator(variant="ab")).fit(s, y, segments=segments)

    lo_g, hi_g = cal.interval_inverse(0.1, 0.5)
    lo_base, hi_base = cal.base_.interval_inverse(0.1, 0.5)
    assert (lo_g, hi_g) == (lo_base, hi_base)

    seg_label = cal.segments_[0]
    lo_s, hi_s = cal.interval_inverse(0.1, 0.5, segment=seg_label)
    p_at_lo = cal.predict_proba(np.array([lo_s]), segments=np.array([seg_label]))
    np.testing.assert_allclose(p_at_lo, [0.1], atol=1e-9)

    raw = cal.point_inverse(np.array([0.3]), segment=seg_label)
    p_back = cal.predict_proba(raw, segments=np.array([seg_label]))
    np.testing.assert_allclose(p_back, [0.3], atol=1e-9)


# ---------------------------------------------------------------- 0.3.4 regressions


def test_is_monotone_survives_serialization() -> None:
    """CAL-3: 0.3.x kept is_monotone_ out of the state, so a non-monotone base
    came back from JSON as monotone (False -> True)."""
    from probcal.binning import HistogramBinningCalibrator

    rng = np.random.default_rng(3)
    s = rng.uniform(0.05, 0.95, 400)
    y = (rng.random(400) < 0.3).astype(float)
    cal = SegmentedCalibrator(base=HistogramBinningCalibrator(n_bins=12, shrinkage=None)).fit(s, y)
    assert cal.base_.is_monotone_ is False
    assert cal.is_monotone_ is False
    assert SegmentedCalibrator.from_json(cal.to_json()).is_monotone_ is False


def test_selector_base_with_custom_candidates_round_trips_with_segments() -> None:
    """CAL-15: the base prototype is encoded through its own param hook."""
    from probcal import CalibratorSelector, TemperatureCalibrator

    s, y = _pd_data(900, seed=4)
    segments = np.array(["a", "b", "c"])[np.arange(900) % 3]
    base = CalibratorSelector(
        candidates={"p": PlattCalibrator(), "t": TemperatureCalibrator()}, cv=3
    )
    cal = SegmentedCalibrator(base=base).fit(s, y, segments=segments)
    again = SegmentedCalibrator.from_json(cal.to_json())
    np.testing.assert_array_equal(
        again.predict_proba(s, segments=segments), cal.predict_proba(s, segments=segments)
    )
    assert set(again.base.candidates) == {"p", "t"}
    assert not base.fitted_  # the prototype is cloned, never fitted in place


def test_complexity_rank_and_interpret_follow_the_class() -> None:
    """CAL-24: rank delegates to the base map; interpret names the actual class."""

    class _Mine(SegmentedCalibrator):
        pass

    assert SegmentedCalibrator().complexity_rank == BetaCalibrator().complexity_rank
    assert SegmentedCalibrator(base=PlattCalibrator()).complexity_rank == 2.0
    s, y = _pd_data(400, seed=0)
    segments = np.array(["a", "b", "c", "d"])[np.arange(400) % 4]
    cal = _Mine().fit(s, y, segments=segments)
    result = cal.interpret()
    assert result.method == "_Mine"
    assert result.param_names == ("tau2", *(f"delta.{g}" for g in cal.segments_))


def test_segmented_shrinkage_invariant_to_weight_scale() -> None:
    rng = np.random.default_rng(7)
    n = 6000
    seg = rng.integers(0, 6, n)
    s = rng.uniform(0.02, 0.4, n)
    y = (rng.random(n) < np.clip(s * (1 + 0.1 * seg), 0, 1)).astype(float)
    w = rng.lognormal(0.0, 1.0, n)
    fits = [
        SegmentedCalibrator(IsotonicCalibrator()).fit(s, y, segments=seg, sample_weight=c * w)
        for c in (0.1, 1.0, 10.0)
    ]
    for f in fits[1:]:
        np.testing.assert_allclose(f.shrink_, fits[0].shrink_, rtol=1e-9, atol=1e-12)
        np.testing.assert_allclose(
            f.predict_proba(s, segments=seg), fits[0].predict_proba(s, segments=seg), rtol=1e-9
        )
