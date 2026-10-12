"""Tests for probcal.metrics.evaluate."""

import numpy as np
import pytest

from probcal._math import expit
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
    full = evaluate(d.y, d.scores, n_boot=25, random_state=3)
    sub = evaluate(d.y, d.scores, n_boot=25, random_state=3, metrics=["e90", "ici"])
    assert sub.names == ("ici", "e90")  # catalog order, not argument order
    for name in sub.names:
        i, j = full.names.index(name), sub.names.index(name)
        assert (full.values[i], full.ci_low[i], full.ci_high[i]) == (
            sub.values[j],
            sub.ci_low[j],
            sub.ci_high[j],
        )


def test_evaluate_iid_bootstrap_seeded_reproducible() -> None:
    # The stratified default is pinned by test_evaluate_metrics_subset_matches_full_run.
    y, p = _calibrated(300)
    names = ("intercept", "log_loss")
    a = evaluate(y, p, n_boot=50, random_state=3, metrics=names, stratify=False)
    b = evaluate(y, p, n_boot=50, random_state=3, metrics=names, stratify=False)
    np.testing.assert_array_equal(a.ci_low, b.ci_low)
    np.testing.assert_array_equal(a.ci_high, b.ci_high)


def test_evaluate_bad_arguments_raise() -> None:
    d = make_pd_portfolio(n=100)
    with pytest.raises(ValueError, match="log_loss"):
        evaluate(d.y, d.scores, n_boot=5, metrics=["nope"])
    with pytest.raises(ValueError, match="n_boot"):
        evaluate(d.y, d.scores, n_boot=-1)
    for stratify in (True, False):
        with pytest.raises(ValueError, match="both classes"):
            evaluate(np.zeros(50), np.full(50, 0.1), n_boot=5, stratify=stratify)


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


def test_evaluate_point_estimates_unchanged_by_e3() -> None:
    d = make_pd_portfolio(n=600, random_state=9)
    w = np.linspace(0.5, 2.0, 600)
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
    flat = evaluate(y, p, metrics=names, n_boot=25, random_state=7)
    col = evaluate(y, p.reshape(-1, 1), metrics=names, n_boot=25, random_state=7)
    assert col.names == flat.names
    assert np.array_equal(col.values, flat.values)
    assert np.array_equal(col.ci_low, flat.ci_low)
    assert np.array_equal(col.ci_high, flat.ci_high)


# ------------------------------------------------------------------ 0.3.4 fixes


def test_evaluate_n_boot_zero_gives_point_estimates_only() -> None:
    # Regression (MET-10): n_boot=0 raised IndexError from np.percentile.
    y, p = _calibrated(300)
    rep = evaluate(y, p, n_boot=0, metrics=("brier", "ece"))
    assert np.all(np.isnan(rep.ci_low)) and np.all(np.isnan(rep.ci_high))
    full = evaluate(y, p, n_boot=5, metrics=("brier", "ece"))
    np.testing.assert_array_equal(rep.values, full.values)


def test_evaluate_seed_keyword_deprecated() -> None:
    y, p = _calibrated(300)
    with pytest.warns(DeprecationWarning, match="random_state"):
        old = evaluate(y, p, n_boot=10, seed=4, metrics=("brier",))
    new = evaluate(y, p, n_boot=10, random_state=4, metrics=("brier",))
    np.testing.assert_array_equal(old.ci_low, new.ci_low)
    with pytest.raises(TypeError, match="only random_state"):
        evaluate(y, p, n_boot=10, seed=4, random_state=5)
