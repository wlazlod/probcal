"""Tests for probcal.bayesian: BBQ and ENIR."""

import warnings

import numpy as np
import pytest

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
    np.testing.assert_allclose(cal.model_weights_.sum(), 1.0, atol=1e-12)
    assert len(cal.model_weights_) == len(cal.bins_grid_)


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
    cal = ENIRCalibrator(max_solutions=None).fit(s, y)
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
    cal = ENIRCalibrator(max_solutions=None).fit(s, y)
    np.testing.assert_allclose(cal.path_solutions_[0], y)
    assert cal.path_lambdas_[0] == 0.0


def test_enir_bic_weights_normalized() -> None:
    cal = ENIRCalibrator().fit(*_sample(400))
    np.testing.assert_allclose(cal.model_weights_.sum(), 1.0, atol=1e-12)


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


def test_enir_fit_warns_above_unique_score_threshold(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # The real threshold (50,000 unique scores, ~35s quadratic fit) is too
    # slow to cross in a fast test; patching the module constant exercises
    # the real emission path at test scale.
    import probcal.bayesian as bayesian_mod

    monkeypatch.setattr(bayesian_mod, "_ENIR_UNIQUE_WARN", 100)
    with pytest.warns(UserWarning, match="quadratic in unique scores"):
        ENIRCalibrator().fit(*_sample(300))


def test_enir_fit_no_scale_warning_below_threshold() -> None:
    # Filter on the message, not simplefilter("error"): an unrelated future
    # warning from the path solver must not fail this test for the wrong
    # reason.
    s, y = _sample(300)
    with warnings.catch_warnings(record=True) as rec:
        warnings.simplefilter("always")
        ENIRCalibrator().fit(s, y)
    assert not any("quadratic in unique scores" in str(w.message) for w in rec)


# ---------------------------------------------------------------- 0.4.0 regressions


def _low_tail_inversion(seed: int = 0) -> tuple[np.ndarray, np.ndarray]:
    """Event rate drops from ~0.6 to 0 inside s < 1e-3, then rises with s."""
    rng = np.random.default_rng(seed)
    s = np.concatenate(
        [rng.uniform(1e-4, 4e-4, 100), rng.uniform(6e-4, 1e-3, 100), rng.uniform(0.1, 0.9, 200)]
    )
    y = np.concatenate(
        [rng.random(100) < 0.6, np.zeros(100, dtype=bool), rng.random(200) < s[200:]]
    ).astype(float)
    return s, y


def test_bbq_monotonicity_checked_over_the_whole_domain() -> None:
    """CAL-4: 0.3.x probed only linspace(0.01, 0.99) and missed low-PD inversions."""
    s, y = _low_tail_inversion()
    cal = BBQCalibrator(min_bins=4, max_bins=4).fit(s, y)
    p = cal.predict_proba(np.array([2e-4, 8e-4]))
    assert p[1] < p[0]  # genuinely decreasing below the old probe grid
    assert cal.is_monotone_ is False
    with pytest.raises(NotImplementedError, match="monotone"):
        cal.interval_inverse(0.1, 0.5)


def test_bbq_monotone_flag_agrees_with_every_breakpoint() -> None:
    """The flag equals a brute-force check at every edge of every candidate."""
    for _ in range(6):
        cal = BBQCalibrator().fit(*_sample(600))
        edges = np.unique(np.concatenate(cal.model_edges_))
        probe = np.concatenate([[1e-12], edges, np.nextafter(edges, 0.0), [1.0 - 1e-12]])
        probe.sort()
        brute = bool(np.all(np.diff(cal.predict_proba(probe)) >= 0.0))
        assert cal.is_monotone_ == brute


def test_bbq_equal_mass_bins_use_sample_weights() -> None:
    """CAL-8: candidate binnings are equal-mass in the sample weights."""
    s = np.linspace(0.01, 0.99, 400)
    y = (RNG.random(400) < s).astype(float)
    w = np.where(s < 0.5, 9.0, 1.0)
    cal = BBQCalibrator(min_bins=4, max_bins=4).fit(s, y, sample_weight=w)
    idx = np.searchsorted(cal.model_edges_[0], s, side="right")
    mass = np.bincount(idx, weights=w, minlength=4) / w.sum()
    np.testing.assert_allclose(mass, 0.25, atol=0.02)


def test_bbq_bin_range_validated_at_fit() -> None:
    s, y = _sample(300)
    for kwargs in ({"min_bins": 0}, {"max_bins": -1}, {"min_bins": 2.5}):
        cal = BBQCalibrator(**kwargs)  # type: ignore[arg-type]
        with pytest.raises(ValueError, match="bins"):
            cal.fit(s, y)


@pytest.mark.parametrize("cls", [BBQCalibrator, ENIRCalibrator])
def test_weights_alias_is_deprecated(cls: type) -> None:
    """CAL-23: ``weights_`` (ensemble weights, easily confused with sample
    weights) is now ``model_weights_``; the old name warns until 0.5.0."""
    cal = cls().fit(*_sample(300))
    with pytest.warns(DeprecationWarning, match=r"model_weights_.*0\.5\.0|0\.5\.0.*model_weights_"):
        old = cal.weights_
    np.testing.assert_array_equal(old, cal.model_weights_)


def test_bbq_and_enir_round_trip_public_state() -> None:
    s, y = _sample(400)
    for cal in (BBQCalibrator().fit(s, y), ENIRCalibrator().fit(s, y)):
        state = cal.to_dict()["state"]
        assert not any(k.startswith("_") for k in state), sorted(state)
        again = type(cal).from_json(cal.to_json())
        np.testing.assert_array_equal(again.predict_proba(GRID), cal.predict_proba(GRID))
        assert again.is_monotone_ == cal.is_monotone_


def test_enir_tie_aggregation_is_row_order_invariant() -> None:
    s, y = _sample(400)
    s = np.round(s, 2)
    perm = RNG.permutation(len(s))
    a = ENIRCalibrator().fit(s, y)
    b = ENIRCalibrator().fit(s[perm], y[perm])
    np.testing.assert_array_equal(a.predict_proba(GRID), b.predict_proba(GRID))
