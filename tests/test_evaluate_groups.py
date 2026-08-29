"""Tests for probcal.metrics.evaluate(by=...) -> GroupedMetricReport."""

import numpy as np
import pytest

from probcal._math import expit
from probcal._results import GroupedMetricReport, MetricReport
from probcal.datasets import make_pd_portfolio
from probcal.metrics import evaluate

RNG = np.random.default_rng(73)


def _calibrated(n: int = 800) -> tuple[np.ndarray, np.ndarray]:
    p = expit(RNG.normal(-0.8, 1.2, n))
    y = (RNG.random(n) < p).astype(float)
    return y, p


def test_evaluate_by_none_is_byte_identical_to_0_2_0_fixture() -> None:
    """Fixture values computed on the pre-``by`` code path (make_pd_portfolio(n=800,
    random_state=5), evaluate(..., n_boot=50, metrics=("brier", "ici"))), hard-coded
    here so a future refactor of the ``by=None`` path cannot silently change it.

    Point estimates stay pinned at ``atol=1e-12``: they are computed on the
    unsorted path and are bit-identical across 0.2.0-0.3.0. The CI bounds are
    pinned at ``rtol=1e-9`` instead: 0.3.0 sorts each bootstrap replicate once
    to share that order across metrics, which reorders every weighted sum
    inside a replicate and can move a percentile bound in its last bits (drift
    measured at 3.9e-11 relative on a n=10,000/n_boot=1,000 catalog run; this
    800-row fixture happens to move less than 1e-12, but the guarantee is the
    looser one).
    """
    d = make_pd_portfolio(n=800, random_state=5)
    rep = evaluate(d.y, d.scores, n_boot=50, metrics=("brier", "ici"))
    assert isinstance(rep, MetricReport)
    assert rep.names == ("brier", "ici")
    np.testing.assert_allclose(rep.values, [0.02347598740117542, 0.04125927233150646], atol=1e-12)
    np.testing.assert_allclose(rep.ci_low, [0.022073442779619666, 0.037005032463440724], rtol=1e-9)
    np.testing.assert_allclose(rep.ci_high, [0.025280355485568177, 0.045428648333590414], rtol=1e-9)


def test_evaluate_by_none_matches_omitting_the_argument() -> None:
    d = make_pd_portfolio(n=400, random_state=3)
    a = evaluate(d.y, d.scores, n_boot=40, seed=7, metrics=("brier", "ece"))
    b = evaluate(d.y, d.scores, n_boot=40, seed=7, metrics=("brier", "ece"), by=None)
    assert isinstance(a, MetricReport)
    assert isinstance(b, MetricReport)
    assert a.names == b.names
    np.testing.assert_array_equal(a.values, b.values)
    np.testing.assert_array_equal(a.ci_low, b.ci_low)
    np.testing.assert_array_equal(a.ci_high, b.ci_high)


def test_evaluate_by_returns_grouped_report_in_sorted_order() -> None:
    y, p = _calibrated(900)
    by = np.where(p < 0.3, "low", np.where(p < 0.6, "mid", "high"))
    rep = evaluate(y, p, n_boot=30, seed=11, metrics=("brier", "log_loss"), by=by)
    assert isinstance(rep, GroupedMetricReport)
    assert rep.groups == ("high", "low", "mid")  # sorted lexicographically
    assert len(rep.reports) == 3
    assert isinstance(rep.pooled, MetricReport)
    assert list(rep.counts) == [int((by == g).sum()) for g in rep.groups]


def test_evaluate_by_group_values_match_direct_call_with_offset_seed() -> None:
    y, p = _calibrated(900)
    by = np.where(p < 0.3, "low", np.where(p < 0.6, "mid", "high"))
    seed = 11
    n_boot = 25
    names = ("brier", "log_loss")
    grouped = evaluate(y, p, n_boot=n_boot, seed=seed, metrics=names, by=by)
    for i, g in enumerate(grouped.groups):
        mask = by == g
        direct = evaluate(y[mask], p[mask], n_boot=n_boot, seed=seed + 1000 * i, metrics=names)
        rep = grouped.reports[i]
        assert rep.names == direct.names
        np.testing.assert_array_equal(rep.values, direct.values)
        np.testing.assert_array_equal(rep.ci_low, direct.ci_low)
        np.testing.assert_array_equal(rep.ci_high, direct.ci_high)


def test_evaluate_by_pooled_matches_direct_call_with_seed_unchanged() -> None:
    y, p = _calibrated(900)
    by = np.where(p < 0.5, "low", "high")
    seed = 5
    n_boot = 20
    names = ("brier",)
    grouped = evaluate(y, p, n_boot=n_boot, seed=seed, metrics=names, by=by)
    direct = evaluate(y, p, n_boot=n_boot, seed=seed, metrics=names)
    np.testing.assert_array_equal(grouped.pooled.values, direct.values)
    np.testing.assert_array_equal(grouped.pooled.ci_low, direct.ci_low)
    np.testing.assert_array_equal(grouped.pooled.ci_high, direct.ci_high)


def test_evaluate_by_length_mismatch_raises() -> None:
    y, p = _calibrated(50)
    with pytest.raises(ValueError, match="by must have the same length as y"):
        evaluate(y, p, n_boot=5, by=np.array(["a", "b"]))


def test_evaluate_by_single_class_group_names_the_group() -> None:
    y = np.array([0.0, 0.0, 0.0, 1.0, 1.0, 1.0])
    p = np.array([0.1, 0.2, 0.3, 0.7, 0.8, 0.9])
    by = np.array(["a", "a", "a", "b", "b", "b"])
    with pytest.raises(ValueError, match=r"group 'a': .*both classes"):
        evaluate(y, p, n_boot=5, by=by)


def test_evaluate_by_to_frame_shape() -> None:
    y, p = _calibrated(300)
    by = np.where(p < 0.5, "low", "high")
    names = ("brier", "ece")
    rep = evaluate(y, p, n_boot=10, metrics=names, by=by)
    frame = rep.to_frame()
    # list of dicts (no pandas import assumed) OR a pandas DataFrame; check row count either way.
    n_rows = len(frame) if isinstance(frame, list) else len(frame.index)
    assert n_rows == (1 + len(rep.groups)) * len(names)
    if isinstance(frame, list):
        assert set(frame[0]) == {"group", "metric", "value", "ci_low", "ci_high"}
        assert frame[0]["group"] == "pooled"
    else:
        assert list(frame.columns) == ["group", "metric", "value", "ci_low", "ci_high"]
        assert frame.iloc[0]["group"] == "pooled"


def test_evaluate_by_repr_smoke() -> None:
    y, p = _calibrated(300)
    by = np.where(p < 0.5, "low", "high")
    rep = evaluate(y, p, n_boot=10, metrics=("brier",), by=by)
    text = repr(rep)
    assert "GroupedMetricReport" in text
    assert "pooled" in text
    assert "low" in text and "high" in text
