"""Tests for probcal.segmented.SegmentedCalibrator."""

import warnings

import numpy as np
import pytest

from probcal import Chain, SegmentedCalibrator, moc_offset_from_counts
from probcal._math import expit, logit
from probcal.parametric import BetaCalibrator, PlattCalibrator


def _pd_data(n: int, seed: int, event_shift: float = 0.0) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    s = expit(rng.normal(-1.0, 1.0, n))
    p_true = expit(logit(s) + event_shift)
    y = (rng.random(n) < p_true).astype(float)
    return s, y


def test_zero_arg_fit_is_a_single_global_segment() -> None:
    s, y = _pd_data(500, seed=1)
    cal = SegmentedCalibrator().fit(s, y)
    assert cal.segments_ == ("__all__",)
    assert cal.tau2_ == 0.0
    np.testing.assert_allclose(cal.delta_tilde_, [0.0])


def test_predict_proba_segments_none_equals_base_map() -> None:
    s, y = _pd_data(900, seed=2)
    segments = np.array(["a", "b", "c"])[np.arange(900) % 3]
    cal = SegmentedCalibrator().fit(s, y, segments=segments)
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


def test_tau2_zero_gives_complete_pooling() -> None:
    s, y = _pd_data(400, seed=0)
    segments = np.array(["a", "b", "c", "d"])[np.arange(400) % 4]
    cal = SegmentedCalibrator().fit(s, y, segments=segments)
    assert cal.tau2_ == 0.0
    np.testing.assert_allclose(cal.delta_tilde_, np.zeros(4), atol=1e-15)
    np.testing.assert_allclose(cal.shrink_, np.zeros(4), atol=1e-15)


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


def test_unseen_global_defaults_delta_to_zero() -> None:
    s, y = _pd_data(400, seed=0)
    segments = np.array(["a", "b", "c", "d"])[np.arange(400) % 4]
    cal = SegmentedCalibrator(unseen="global").fit(s, y, segments=segments)
    with pytest.warns(UserWarning, match="none of the"):
        p_unseen = cal.predict_proba(s[:5], segments=np.array(["z"] * 5))
    np.testing.assert_allclose(p_unseen, cal.base_.predict_proba(s[:5]))


def test_unseen_raise_raises_value_error() -> None:
    s, y = _pd_data(400, seed=0)
    segments = np.array(["a", "b", "c", "d"])[np.arange(400) % 4]
    cal = SegmentedCalibrator(unseen="raise").fit(s, y, segments=segments)
    with pytest.raises(ValueError, match="unseen"):
        cal.predict_proba(s[:5], segments=np.array(["z"] * 5))


def test_int_fit_float_predict_label_mismatch_warns_and_falls_back_to_base_map() -> None:
    # Regression: int labels at fit ("0", "1" after str()) vs. float labels at
    # predict ("0.0", "1.0") never match, so every row falls back to the
    # global map under unseen="global" -- this must warn rather than fail
    # silently.
    s, y = _pd_data(600, seed=10)
    int_segments = np.array([0, 1])[np.arange(600) % 2]
    cal = SegmentedCalibrator().fit(s, y, segments=int_segments)
    float_segments = np.array([0.0, 1.0])[np.arange(5) % 2]
    with pytest.warns(UserWarning, match="none of the 5 segment labels"):
        p_mismatch = cal.predict_proba(s[:5], segments=float_segments)
    np.testing.assert_allclose(p_mismatch, cal.base_.predict_proba(s[:5]))


def test_partial_overlap_with_unseen_labels_does_not_warn() -> None:
    s, y = _pd_data(400, seed=0)
    segments = np.array(["a", "b", "c", "d"])[np.arange(400) % 4]
    cal = SegmentedCalibrator(unseen="global").fit(s, y, segments=segments)
    mixed = np.array(["a", "z", "b", "z", "c"])
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        cal.predict_proba(s[:5], segments=mixed)  # some rows match -> no warning


def test_int_fit_float_predict_label_mismatch_still_raises_under_unseen_raise() -> None:
    s, y = _pd_data(600, seed=10)
    int_segments = np.array([0, 1])[np.arange(600) % 2]
    cal = SegmentedCalibrator(unseen="raise").fit(s, y, segments=int_segments)
    float_segments = np.array([0.0, 1.0])[np.arange(5) % 2]
    with pytest.raises(ValueError, match="unseen"):
        cal.predict_proba(s[:5], segments=float_segments)


def test_invalid_unseen_raises_at_fit_time() -> None:
    s, y = _pd_data(200, seed=0)
    with pytest.raises(ValueError, match="unseen"):
        SegmentedCalibrator(unseen="bogus").fit(s, y)


def test_segments_length_mismatch_raises() -> None:
    s, y = _pd_data(200, seed=0)
    with pytest.raises(ValueError, match="length"):
        SegmentedCalibrator().fit(s, y, segments=np.array(["a", "b"]))


def test_custom_base_is_cloned_and_fitted() -> None:
    s, y = _pd_data(600, seed=5)
    segments = np.array(["a", "b"])[np.arange(600) % 2]
    cal = SegmentedCalibrator(base=PlattCalibrator()).fit(s, y, segments=segments)
    assert isinstance(cal.base_, PlattCalibrator)
    assert cal.base_.fitted_
    assert cal.base is None or not getattr(cal.base, "fitted_", False)  # ctor arg untouched


def test_round_trip_and_fingerprint_stability() -> None:
    s, y = _pd_data(500, seed=9)
    segments = np.array(["a", "b", "c"])[np.arange(500) % 3]
    cal = SegmentedCalibrator().fit(s, y, segments=segments)
    cal2 = SegmentedCalibrator.from_dict(cal.to_dict())
    np.testing.assert_array_equal(
        cal.predict_proba(s, segments=segments), cal2.predict_proba(s, segments=segments)
    )
    assert cal.fingerprint() == cal2.fingerprint()
    assert cal2.to_dict() == cal.to_dict()

    twin = SegmentedCalibrator().fit(s, y, segments=segments)
    assert cal.fingerprint() == twin.fingerprint()
    other_s, other_y = _pd_data(500, seed=99)
    alt = SegmentedCalibrator().fit(other_s, other_y, segments=segments)
    assert cal.fingerprint() != alt.fingerprint()


def test_works_inside_chain_with_moc_offset_using_global_map() -> None:
    s, y = _pd_data(800, seed=4)
    segments = np.array(["a", "b"])[np.arange(800) % 2]
    seg = SegmentedCalibrator().fit(s, y, segments=segments)
    off = moc_offset_from_counts(y, seg.predict_proba(s))
    chain = Chain([seg, off])
    # Chain has no segments= slot: seg predicts through its global map (delta=0).
    np.testing.assert_allclose(chain.predict_proba(s), off.transform(seg.predict_proba(s)))


def test_affine_logit_coeffs_single_segment_only() -> None:
    s, y = _pd_data(500, seed=6)
    cal_single = SegmentedCalibrator(base=PlattCalibrator()).fit(s, y)
    assert cal_single.segments_ == ("__all__",)
    coeffs = cal_single.affine_logit_coeffs_
    assert coeffs is not None
    a, b = coeffs
    base_a, base_b = cal_single.base_.affine_logit_coeffs_
    assert a == base_a
    assert b == base_b + cal_single.delta_tilde_[0]

    segments = np.array(["a", "b"])[np.arange(500) % 2]
    cal_multi = SegmentedCalibrator(base=PlattCalibrator()).fit(s, y, segments=segments)
    assert cal_multi.affine_logit_coeffs_ is None


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


def test_interpret_reports_tau2_and_per_segment_row() -> None:
    s, y = _pd_data(400, seed=0)
    segments = np.array(["a", "b", "c", "d"])[np.arange(400) % 4]
    cal = SegmentedCalibrator().fit(s, y, segments=segments)
    result = cal.interpret()
    assert result.method == "SegmentedCalibrator"
    assert result.param_names[0] == "tau2"
    assert set(result.param_names[1:]) == {f"delta.{g}" for g in cal.segments_}
    assert any("tau2" in m for m in result.messages)
    for g in cal.segments_:
        assert any(repr(g) in m for m in result.messages)
