"""Tests for probcal.bayesian: BBQ and ENIR (ENIR's path is also gated by
test_enir_equivalence.py)."""

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


# ---------------------------------------------------------------- ENIR


def test_enir_path_runs_from_raw_data_to_the_isotonic_fit() -> None:
    s = np.array([0.1, 0.3, 0.5, 0.7, 0.9])
    y = np.array([0.0, 1.0, 0.0, 1.0, 1.0])
    small = ENIRCalibrator(max_solutions=None).fit(s, y)
    np.testing.assert_allclose(small.path_solutions_[0], y)
    assert small.path_lambdas_[0] == 0.0

    s, y = _sample(300)
    cal = ENIRCalibrator(max_solutions=None).fit(s, y)
    # Aggregate ties the same way the calibrator does, then compare the final
    # path solution with plain isotonic regression.
    order = np.argsort(s, kind="stable")
    s_sorted, y_sorted = s[order], y[order]
    _, start = np.unique(s_sorted, return_index=True)
    counts = np.diff(np.append(start, len(s_sorted)))
    y_agg = np.add.reduceat(y_sorted, start) / counts
    iso = pava(y_agg, counts.astype(float)).fitted
    np.testing.assert_allclose(cal.path_solutions_[-1], iso, atol=1e-10)


def test_enir_fit_warns_above_unique_score_threshold(monkeypatch: pytest.MonkeyPatch) -> None:
    # The real threshold (50,000 unique scores, ~35s quadratic fit) is too
    # slow to cross in a fast test; patching the module constant exercises
    # the real emission path at test scale.
    import probcal.bayesian as bayesian_mod

    monkeypatch.setattr(bayesian_mod, "_ENIR_UNIQUE_WARN", 100)
    with pytest.warns(UserWarning, match="quadratic in unique scores"):
        ENIRCalibrator().fit(*_sample(300))


def test_enir_tie_aggregation_is_row_order_invariant() -> None:
    s, y = _sample(400)
    s = np.round(s, 2)
    perm = RNG.permutation(len(s))
    a = ENIRCalibrator().fit(s, y)
    b = ENIRCalibrator().fit(s[perm], y[perm])
    np.testing.assert_array_equal(a.predict_proba(GRID), b.predict_proba(GRID))


# ---------------------------------------------------------------- BBQ


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
    for _ in range(3):
        cal = BBQCalibrator().fit(*_sample(600))
        np.testing.assert_allclose(cal.model_weights_.sum(), 1.0, atol=1e-12)
        edges = np.unique(np.concatenate(cal.model_edges_))
        probe = np.concatenate([[1e-12], edges, np.nextafter(edges, 0.0), [1.0 - 1e-12]])
        probe.sort()
        brute = bool(np.all(np.diff(cal.predict_proba(probe)) >= 0.0))
        assert cal.is_monotone_ == brute


def _floor_tied_sample() -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(0)
    s = np.maximum(np.round(rng.beta(0.6, 40, 2000), 3), 0.001)
    y = (rng.random(2000) < s).astype(float)
    return s, y


def test_bbq_below_tied_minimum_is_not_the_prior() -> None:
    """0.3.5: an empty first bin below a tied minimum fell back to the prior."""
    s, y = _floor_tied_sample()
    cal = BBQCalibrator().fit(s, y)
    below, at_min = cal.predict_proba([0.0005, 0.001])
    assert below == at_min
    assert below < 0.01
    for edges, rates in zip(cal.model_edges_, cal.model_rates_, strict=True):
        assert np.all(edges > s.min())
        assert len(rates) == len(edges) + 1


def test_bbq_constant_scores_constant_map() -> None:
    s = np.full(50, 0.3)
    y = (np.arange(50) < 17).astype(float)
    cal = BBQCalibrator().fit(s, y)
    np.testing.assert_allclose(cal.predict_proba([0.0, 0.3, 1.0]), (17 + 0.5) / (50 + 1.0))
    assert cal.is_monotone_


def test_bbq_monotone_option_isotonizes_candidates() -> None:
    s, y = _low_tail_inversion()
    free = BBQCalibrator(min_bins=4, max_bins=4).fit(s, y)
    cal = BBQCalibrator(min_bins=4, max_bins=4, monotone=True).fit(s, y)
    assert free.is_monotone_ is False and cal.is_monotone_ is True
    np.testing.assert_array_equal(cal.model_weights_, free.model_weights_)
    assert all(np.all(np.diff(r) >= 0) for r in cal.model_rates_)
    lo, hi = cal.interval_inverse(0.1, 0.5)
    assert lo < hi
    back = BBQCalibrator.from_json(cal.to_json())
    assert back.monotone is True
    np.testing.assert_array_equal(back.predict_proba(s), cal.predict_proba(s))
    # Default payloads carry no "monotone" key, so their fingerprints are unchanged.
    assert "monotone" not in free.to_dict()["params"]
    with pytest.raises(ValueError, match="monotone"):
        BBQCalibrator(monotone="yes").fit(s, y)  # type: ignore[arg-type]
