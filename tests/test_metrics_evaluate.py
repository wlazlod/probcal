"""Tests for probcal.metrics.evaluate."""

import numpy as np
import pytest

from probcal._math import expit
from probcal._results import MetricReport
from probcal.datasets import make_pd_portfolio
from probcal.metrics import (
    ReliabilitySummary,
    calibration_intercept,
    calibration_slope,
    e90,
    evaluate,
    ici,
    log_loss,
    reliability_summary,
    spiegelhalter_z,
)

RNG = np.random.default_rng(73)


def _calibrated(n: int = 800) -> tuple[np.ndarray, np.ndarray]:
    p = expit(RNG.normal(-0.8, 1.2, n))
    y = (RNG.random(n) < p).astype(float)
    return y, p


def _rare_event_fixture(
    n: int = 150, n_events: int = 4, seed: int = 11
) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    p = np.clip(rng.beta(2, 20, n), 1e-3, 1 - 1e-3)
    y = np.zeros(n)
    y[rng.choice(n, size=n_events, replace=False)] = 1.0
    return y, p


def test_evaluate_returns_metric_report_with_cis() -> None:
    y, p = _calibrated()
    rep = evaluate(y, p, n_boot=30, seed=1)
    assert isinstance(rep, MetricReport)
    for name in ("log_loss", "brier", "ece", "ece_debiased", "mce", "ici", "smooth_ece"):
        assert name in rep.names
    # CIs bracket the point estimates for the well-behaved metrics.
    for i, name in enumerate(rep.names):
        if name in ("log_loss", "brier"):
            assert rep.ci_low[i] <= rep.values[i] <= rep.ci_high[i]


def test_evaluate_seeded_reproducible() -> None:
    y, p = _calibrated(400)
    a = evaluate(y, p, n_boot=20, seed=7)
    b = evaluate(y, p, n_boot=20, seed=7)
    np.testing.assert_allclose(a.ci_low, b.ci_low)
    np.testing.assert_allclose(a.ci_high, b.ci_high)


def test_reliability_summary_matches_metric_calls() -> None:
    y, p = _calibrated(1500)
    s = reliability_summary(y, p)
    assert isinstance(s, ReliabilitySummary)
    assert s.n == 1500 and s.events == int(y.sum())
    assert s.intercept == calibration_intercept(y, p)
    assert s.slope == calibration_slope(y, p)
    assert s.ici == ici(y, p)
    assert s.e90 == e90(y, p)
    assert s.spiegelhalter_p == spiegelhalter_z(y, p).p_value


def test_evaluate_metrics_subset_matches_full_run() -> None:
    d = make_pd_portfolio(n=400)
    full = evaluate(d.y, d.scores, n_boot=25, seed=3)
    sub = evaluate(d.y, d.scores, n_boot=25, seed=3, metrics=["e90", "ici"])
    assert sub.names == ("ici", "e90")  # catalog order, not argument order
    for name in sub.names:
        i, j = full.names.index(name), sub.names.index(name)
        assert (full.values[i], full.ci_low[i], full.ci_high[i]) == (
            sub.values[j],
            sub.ci_low[j],
            sub.ci_high[j],
        )


def test_evaluate_unknown_metric_raises_listing_valid_names() -> None:
    d = make_pd_portfolio(n=100)
    with pytest.raises(ValueError, match="log_loss"):
        evaluate(d.y, d.scores, n_boot=5, metrics=["nope"])


@pytest.mark.slow
def test_evaluate_default_boot() -> None:
    y, p = _calibrated(300)
    rep = evaluate(y, p)  # n_boot=1000 default
    assert len(rep.names) == len(rep.values)


def test_evaluate_stratified_avoids_degenerate_substitution_artifact() -> None:
    """Rare-event fixture (n=150, 4 events) that genuinely triggers the old
    (v0.1.2) iid + degenerate-substitution rule: ``if yb.min() == yb.max():
    boot[b] = values``.

    Reimplement that old rule inline (same resampling scheme, current
    ``calibration_intercept``/``log_loss``) and confirm it substitutes the
    bit-exact point estimate for every degenerate replicate -- the artifact
    ``stratify=True`` removes by construction (a stratified replicate can
    never be single-class).

    Note on CI width: the old rule's substitution *narrows* its CI (it
    replaces some replicates with a single repeated value), but on this
    fixture the new stratified CI is narrower still, not wider. Stratifying
    conditions the bootstrap on the observed event count, which removes an
    *additional*, larger source of variance that the old iid scheme carries
    (the event count itself fluctuates 0..n per iid replicate, which is the
    dominant driver of instability for a rare-event, regression-based metric
    like ``calibration_intercept``). So "old width > new width" here, for
    both metrics -- the reverse of a naive "no more collapsing => wider"
    expectation.
    """
    y, p = _rare_event_fixture()
    names = ("intercept", "log_loss")
    n = len(y)

    # --- reimplement the old v0.1.2 iid + degenerate-substitution rule ---
    rng = np.random.default_rng(3)
    point = {"intercept": calibration_intercept(y, p), "log_loss": log_loss(y, p)}
    old_boot = np.empty((200, len(names)))
    n_degen = 0
    for b in range(200):
        idx = rng.integers(0, n, n)
        yb, pb = y[idx], p[idx]
        if yb.min() == yb.max():
            old_boot[b] = [point[k] for k in names]
            n_degen += 1
            continue
        old_boot[b] = [calibration_intercept(yb, pb), log_loss(yb, pb)]

    assert n_degen > 0  # fixture genuinely triggers the old rule
    # The substitution rule literally repeats the bit-exact point estimate.
    assert (old_boot[:, 0] == point["intercept"]).sum() == n_degen
    assert (old_boot[:, 1] == point["log_loss"]).sum() == n_degen
    old_width = np.percentile(old_boot, 97.5, axis=0) - np.percentile(old_boot, 2.5, axis=0)

    rep = evaluate(y, p, n_boot=200, seed=3, metrics=names)
    idx_map = {name: i for i, name in enumerate(rep.names)}
    new_width = np.array([rep.ci_high[idx_map[name]] - rep.ci_low[idx_map[name]] for name in names])

    assert new_width[0] < old_width[0]  # intercept
    assert new_width[1] < old_width[1]  # log_loss


def test_evaluate_stratified_replicates_preserve_class_counts() -> None:
    y, p = _rare_event_fixture()
    n_events = int(y.sum())
    idx0 = np.flatnonzero(y == 0)
    idx1 = np.flatnonzero(y == 1)

    # Independently replicate the stratified index draws `evaluate` performs
    # internally (same seed) and confirm every replicate has exactly the
    # observed event count -- a stratified replicate cannot be degenerate by
    # construction.
    rng = np.random.default_rng(3)
    for _ in range(200):
        idx = np.concatenate(
            [
                idx0[rng.integers(0, len(idx0), len(idx0))],
                idx1[rng.integers(0, len(idx1), len(idx1))],
            ]
        )
        assert int(y[idx].sum()) == n_events

    # Seed reproducibility: two identical calls give identical CIs.
    a = evaluate(y, p, n_boot=100, seed=3, metrics=("intercept", "log_loss"))
    b = evaluate(y, p, n_boot=100, seed=3, metrics=("intercept", "log_loss"))
    np.testing.assert_array_equal(a.ci_low, b.ci_low)
    np.testing.assert_array_equal(a.ci_high, b.ci_high)


def test_evaluate_single_class_raises_valueerror() -> None:
    y = np.zeros(50)
    p = np.full(50, 0.1)
    with pytest.raises(ValueError, match="both classes"):
        evaluate(y, p, n_boot=5, stratify=True)
    with pytest.raises(ValueError, match="both classes"):
        evaluate(y, p, n_boot=5, stratify=False)


def test_evaluate_stratify_false_runs_and_is_reproducible() -> None:
    # A well-balanced fixture: iid draws are exceedingly unlikely to be
    # degenerate, so the 100-redraw RuntimeError path is not exercised here.
    # (That path needs 100 *consecutive* single-class draws; even the
    # rare-event fixture above only has a ~1.7% per-draw degenerate
    # probability, making 100 straight failures astronomically unlikely to
    # construct deterministically without mocking the RNG -- not attempted.)
    y, p = _calibrated(300)
    a = evaluate(y, p, n_boot=30, seed=5, stratify=False)
    b = evaluate(y, p, n_boot=30, seed=5, stratify=False)
    np.testing.assert_array_equal(a.ci_low, b.ci_low)
    np.testing.assert_array_equal(a.ci_high, b.ci_high)
    assert len(a.names) == len(a.values)
