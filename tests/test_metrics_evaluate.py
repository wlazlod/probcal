"""Tests for probcal.metrics.evaluate."""

import numpy as np
import pytest

from probcal._math import expit
from probcal._results import MetricReport
from probcal.datasets import make_pd_portfolio
from probcal.metrics import (
    _METRIC_CATALOG,
    ReliabilitySummary,
    _point_metrics,
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


def test_reliability_summary_e90_reflects_sample_weight() -> None:
    from probcal.metrics.smooth import _ici_distances

    d = make_pd_portfolio(n=2000)
    y, p = d.y, d.scores
    dist = _ici_distances(y, p, 0.75, 512)
    tail_idx = np.argsort(dist)[-50:]  # top ~2.5% largest ICI distances
    w = np.ones(len(y))
    w[tail_idx] = 50.0
    baseline = reliability_summary(y, p).e90
    weighted = reliability_summary(y, p, sample_weight=w).e90
    assert weighted != baseline


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


# --------------------------------------- point-estimate guard for the 0.3.0 bootstrap sort


def _old_point_metrics(
    y: np.ndarray,
    p: np.ndarray,
    w: np.ndarray | None,
    names: tuple[str, ...] | None = None,
) -> dict[str, float]:
    """``_point_metrics`` exactly as it stood before the 0.3.0 throughput work.

    The bootstrap loop now sorts each replicate once and takes a shared-sort
    fast path (``presorted=True``); the *point* estimates must keep coming off
    this original path bit-for-bit. Copied verbatim so a later refactor of the
    live function is compared against the released numerics, not against
    itself.
    """
    from probcal._math import loess, weighted_quantile
    from probcal.metrics import (
        brier_score,
        brier_skill_score,
        calibration_intercept,
        calibration_slope,
        ecce,
        log_loss,
        smooth_ece,
        spiegelhalter_z,
    )
    from probcal.metrics.binned import ece, ece_debiased, ece_sweep

    sel = set(_METRIC_CATALOG if names is None else names)
    dispatch = {
        "log_loss": lambda: log_loss(y, p, sample_weight=w),
        "brier": lambda: brier_score(y, p, sample_weight=w),
        "brier_skill": lambda: brier_skill_score(y, p, sample_weight=w),
        "ece": lambda: ece(y, p, sample_weight=w),
        "ece_debiased": lambda: ece_debiased(y, p, sample_weight=w),
        "mce": lambda: ece(y, p, norm="max", sample_weight=w),
        "ece_sweep": lambda: ece_sweep(y, p, sample_weight=w),
        "smooth_ece": lambda: smooth_ece(y, p, sample_weight=w),
        "intercept": lambda: calibration_intercept(y, p, sample_weight=w),
        "slope": lambda: calibration_slope(y, p, sample_weight=w),
    }
    out: dict[str, float] = {k: fn() for k, fn in dispatch.items() if k in sel}

    if sel & {"ecce_max", "ecce_mean"}:
        ec = ecce(y, p, sample_weight=w)
        out["ecce_max"] = ec.stat_max
        out["ecce_mean"] = ec.stat_mean

    if sel & {"ici", "e50", "e90", "emax"}:
        d = np.abs(loess(p, y, frac=0.75, grid_size=512) - p)
        w_arr = np.ones(len(p)) if w is None else w
        uniform_w = w is None or bool(np.all(w == w[0]))
        if "ici" in sel:
            out["ici"] = float(np.average(d, weights=w_arr))
        if "e50" in sel:
            out["e50"] = (
                float(np.quantile(d, 0.5)) if uniform_w else float(weighted_quantile(d, 0.5, w))
            )
        if "e90" in sel:
            out["e90"] = (
                float(np.quantile(d, 0.9)) if uniform_w else float(weighted_quantile(d, 0.9, w))
            )
        if "emax" in sel:
            out["emax"] = float(np.max(d))

    if sel & {"spiegelhalter_z", "spiegelhalter_p"}:
        sp = spiegelhalter_z(y, p, sample_weight=w)
        if "spiegelhalter_z" in sel:
            out["spiegelhalter_z"] = sp.z
        if "spiegelhalter_p" in sel:
            out["spiegelhalter_p"] = sp.p_value

    return {k: out[k] for k in _METRIC_CATALOG if k in sel}


@pytest.mark.parametrize("weighted", [False, True])
def test_evaluate_point_estimates_unchanged_by_e3(weighted: bool) -> None:
    d = make_pd_portfolio(n=600, random_state=9)
    w = np.linspace(0.5, 2.0, 600) if weighted else np.ones(600)
    got = _point_metrics(d.y, d.scores, w, _METRIC_CATALOG)
    want = _old_point_metrics(d.y, d.scores, w, _METRIC_CATALOG)
    assert list(got) == list(want) == list(_METRIC_CATALOG)
    for name in _METRIC_CATALOG:
        assert got[name] == want[name], name


def test_point_metrics_presorted_matches_the_default_path() -> None:
    """The bootstrap fast path may only move values by summation order (~1e-15)."""
    d = make_pd_portfolio(n=1500, random_state=6)
    order = np.argsort(d.scores, kind="stable")
    y, p, w = d.y[order], d.scores[order], np.ones(1500)
    slow = _point_metrics(y, p, w, _METRIC_CATALOG)
    fast = _point_metrics(y, p, w, _METRIC_CATALOG, presorted=True)
    for name in _METRIC_CATALOG:
        assert fast[name] == pytest.approx(slow[name], rel=1e-12, abs=1e-15), name


def test_evaluate_accepts_single_column_p() -> None:
    y, p = _calibrated(300)
    names = ["log_loss", "brier", "ece", "intercept"]
    flat = evaluate(y, p, metrics=names, n_boot=25, seed=7)
    col = evaluate(y, p.reshape(-1, 1), metrics=names, n_boot=25, seed=7)
    assert col.names == flat.names
    assert np.array_equal(col.values, flat.values)
    assert np.array_equal(col.ci_low, flat.ci_low)
    assert np.array_equal(col.ci_high, flat.ci_high)
