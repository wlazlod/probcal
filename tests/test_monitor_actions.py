"""Tests for probcal.monitor._actions (margin-of-conservatism offsets)."""

import numpy as np
import pytest

from probcal import BetaCalibrator, CalibratedModel, Chain, LogitOffset, make_pd_portfolio
from probcal._math import beta_ppf, expit, logit
from probcal.monitor import AppliedAction, CalibrationMonitor, moc_offset, moc_offset_from_counts


def _batch(n=1000, shift=0.0, slope=1.0, seed=0, event_rate=0.05):
    d = make_pd_portfolio(n=n, event_rate=event_rate, random_state=seed)
    p = d.scores
    true = expit(slope * logit(p) + shift)
    rng = np.random.default_rng(seed + 1000)
    y = (rng.random(n) < true).astype(float)
    return y, p


def _drifted_monitor(shift=0.8, n_batches=6, alpha=0.05):
    mon = CalibrationMonitor(alpha=alpha)
    for k in range(n_batches):
        y, p = _batch(shift=shift, seed=k)
        mon.update(y, p, label=f"m{k}")
    return mon


# ---------------------------------------------------------------- moc_offset


def test_moc_offset_delta_at_least_the_plug_in_estimate() -> None:
    # The CS's upper end is the margin of conservatism: it is at least as
    # large as the predictable plug-in estimate the monitor itself used.
    mon = _drifted_monitor()
    off = moc_offset(mon)
    assert isinstance(off, LogitOffset)
    assert off.fitted_
    assert off.delta_ >= mon.steps_[-1].delta_hat


def test_moc_offset_default_uses_last_step_delta_ci() -> None:
    mon = _drifted_monitor()
    off = moc_offset(mon)
    assert off.delta_ == pytest.approx(mon.steps_[-1].delta_ci[1])


def test_moc_offset_fits_on_last_batch_probabilities() -> None:
    mon = _drifted_monitor()
    off = moc_offset(mon)
    expected_pre_mean = float(np.mean(expit(mon._z[-1])))
    assert off.pre_mean_ == pytest.approx(expected_pre_mean)


def test_moc_offset_level_recomputes_surviving_set() -> None:
    mon = _drifted_monitor()
    off_default = moc_offset(mon)
    off_level = moc_offset(mon, level=1.0 - mon.alpha)
    # Same alpha as the monitor's own -> same threshold -> same result.
    assert off_level.delta_ == pytest.approx(off_default.delta_)

    threshold = -np.log(1.0 - 0.5)
    surviving = mon._cs_grid[mon._cs_max < threshold]
    off_tight = moc_offset(mon, level=0.5)
    assert off_tight.delta_ == pytest.approx(float(surviving.max()))


def test_moc_offset_level_requires_a_monitor_not_a_report() -> None:
    mon = _drifted_monitor()
    report = mon.report()
    with pytest.raises(TypeError, match="CalibrationMonitor"):
        moc_offset(report, level=0.8)


def test_moc_offset_report_input_uses_placeholder_batch() -> None:
    mon = _drifted_monitor()
    report = mon.report()
    off = moc_offset(report)
    assert off.delta_ == pytest.approx(report.steps[-1].delta_ci[1])
    assert off.pre_mean_ == pytest.approx(0.5)


def test_moc_offset_invalid_level_raises() -> None:
    mon = _drifted_monitor()
    with pytest.raises(ValueError, match="level"):
        moc_offset(mon, level=1.5)


def test_moc_offset_no_steps_raises() -> None:
    mon = CalibrationMonitor()
    with pytest.raises(ValueError, match="no batches"):
        moc_offset(mon)


def test_moc_offset_empty_delta_ci_raises() -> None:
    mon = CalibrationMonitor(alpha=0.05, delta_ci_grid=(-0.1, 0.1, 5))
    for k in range(4):
        y, p = _batch(shift=2.0, seed=k)
        mon.update(y, p, label=f"m{k}")
    assert mon.steps_[-1].delta_ci is None
    with pytest.raises(ValueError, match="delta_ci_grid"):
        moc_offset(mon)


def test_moc_offset_empty_surviving_set_at_level_raises() -> None:
    mon = CalibrationMonitor(alpha=0.4, delta_ci_grid=(-0.1, 0.1, 5))
    for k in range(4):
        y, p = _batch(shift=2.0, seed=k)
        mon.update(y, p, label=f"m{k}")
    with pytest.raises(ValueError, match="delta_ci_grid"):
        moc_offset(mon, level=0.999999)


def test_moc_offset_rejects_wrong_type() -> None:
    with pytest.raises(TypeError):
        moc_offset("not a monitor")


# ---------------------------------------------------------------- moc_offset_from_counts


def test_moc_offset_from_counts_transform_mean_matches_jeffreys_quantile() -> None:
    rng = np.random.default_rng(1)
    y = (rng.random(2000) < 0.03).astype(float)
    p = np.full(2000, 0.02)
    off = moc_offset_from_counts(y, p, level=0.9)
    k = float(np.sum(y))
    n = float(len(y))
    q = beta_ppf(0.9, k + 0.5, n - k + 0.5)
    np.testing.assert_allclose(off.transform(p).mean(), q, atol=1e-10)


def test_moc_offset_from_counts_level_half_recovers_approximate_mle() -> None:
    rng = np.random.default_rng(2)
    n = 20000
    y = (rng.random(n) < 0.04).astype(float)
    p = np.full(n, 0.03)
    off = moc_offset_from_counts(y, p, level=0.5)
    mle = float(np.mean(y))
    assert abs(off.post_mean_ - mle) < 0.005


def test_moc_offset_from_counts_weighted() -> None:
    y = np.array([0.0, 0.0, 1.0, 0.0])
    p = np.array([0.05, 0.05, 0.05, 0.05])
    w = np.array([2.0, 3.0, 1.0, 4.0])
    off = moc_offset_from_counts(y, p, level=0.9, sample_weight=w)
    k = float(np.sum(w * y))
    n = float(np.sum(w))
    q = beta_ppf(0.9, k + 0.5, n - k + 0.5)
    assert off.post_mean_ == pytest.approx(q, abs=1e-9)


def test_moc_offset_from_counts_invalid_level_raises() -> None:
    y = np.array([0.0, 1.0])
    p = np.array([0.1, 0.2])
    with pytest.raises(ValueError, match="level"):
        moc_offset_from_counts(y, p, level=0.0)


# ---------------------------------------------------------------- composition


def test_chain_with_moc_offset_serializes_and_inverts() -> None:
    d = make_pd_portfolio(n=1500, random_state=9)
    cal = BetaCalibrator().fit(d.scores, d.y)
    off = moc_offset_from_counts(d.y, cal.predict_proba(d.scores), level=0.9)
    chain = Chain([cal, off])

    payload = chain.to_dict()
    restored = Chain.from_dict(payload)
    q = np.linspace(0.05, 0.5, 10)
    np.testing.assert_allclose(chain.predict_proba(q), restored.predict_proba(q))

    lo, hi = chain.interval_inverse(0.0, 0.3, space="logit", buffer_logit=0.1)
    assert np.isneginf(lo)
    assert np.isfinite(hi) or np.isposinf(hi)


def test_chain_with_monitor_moc_offset_serializes_and_inverts() -> None:
    # The brief's named scenario: Chain([cal, moc_offset(mon)]) -- the
    # monitor-derived offset, not the counts variant covered above.
    d = make_pd_portfolio(n=1500, random_state=10)
    cal = BetaCalibrator().fit(d.scores, d.y)
    mon = _drifted_monitor(shift=0.8, n_batches=5)
    off = moc_offset(mon)
    chain = Chain([cal, off])

    payload = chain.to_dict()
    restored = Chain.from_dict(payload)
    q = np.linspace(0.05, 0.5, 10)
    np.testing.assert_allclose(chain.predict_proba(q), restored.predict_proba(q))

    lo, hi = chain.interval_inverse(0.0, 0.5, space="logit", buffer_logit=0.1)
    assert np.isneginf(lo)
    assert np.isfinite(hi) or np.isposinf(hi)

    # BetaCalibrator has an exact point inverse, so the Chain does too.
    p_target = chain.predict_proba(d.scores[:20])
    s = chain.point_inverse(p_target)
    np.testing.assert_allclose(chain.predict_proba(s), p_target, atol=1e-8)


# ---------------------------------------------------------------- apply_recommendation


class _StubModel:
    """Deterministic sklearn-free model over a single score column (mirrors
    tests/test_golden.py's _StubModel)."""

    def fit(self, X, y):  # noqa: ARG002
        return self

    def predict_proba(self, X):
        s = np.asarray(X)[:, 0]
        return np.column_stack([1.0 - s, s])

    def get_params(self):
        return {"stub": True}


def test_apply_recommendation_none_when_no_alarm() -> None:
    mon = CalibrationMonitor(alpha=0.05)
    for k in range(5):
        y, p = _batch(shift=0.0, seed=500 + k)
        mon.update(y, p, label=f"m{k}")
    action = mon.apply_recommendation()
    assert isinstance(action, AppliedAction)
    assert action.kind == "none"
    assert action.offset is None
    assert action.composed is None
    assert action.monitor is None
    assert action.window == ()


def test_apply_recommendation_re_fit_carries_no_offset() -> None:
    # Slope drift, like test_monitor_sim.py's _recommendation_run.
    mon = CalibrationMonitor(alpha=0.05)
    for k in range(12):
        y, p = _batch(shift=0.0, slope=0.7, seed=100 + k)
        mon.update(y, p, label=f"m{k}")
    rep = mon.report()
    assert rep.recommendation == "re-fit"

    action = mon.apply_recommendation()
    assert action.kind == "re-fit"
    assert action.offset is None
    assert action.composed is None
    assert action.monitor is None
    assert action.window  # non-empty: the suggested re-fit window


def test_apply_recommendation_does_not_mutate_the_old_monitor() -> None:
    mon = _drifted_monitor(shift=0.8, n_batches=6)
    before = mon.to_dict()
    mon.apply_recommendation()
    after = mon.to_dict()
    assert before == after


def test_apply_recommendation_window_uses_onset_index_not_label_lookup() -> None:
    # Regression: labels are documented as opaque and may repeat. Deriving
    # the onset index by looking a batch up BY LABEL (the previous
    # implementation) silently returns the FIRST match and can point at
    # the wrong index; the window must come from estimate_onset's index
    # directly (CalibrationMonitor._onset_index), exactly as report() does.
    from probcal.monitor._onset import estimate_onset

    mon = CalibrationMonitor(alpha=0.05)
    for k in range(4):
        y, p = _batch(n=2000, shift=0.0, seed=k)
        mon.update(y, p, label="m")  # every batch shares the same label
    for k in range(6):
        y, p = _batch(n=2000, shift=0.8, seed=100 + k)
        mon.update(y, p, label="m")
    rep = mon.report()
    assert rep.alarm_at == "m"

    onset_idx = estimate_onset(np.array([s.log_e_increment for s in mon.steps_]))
    action = mon.apply_recommendation()
    assert len(action.window) == len(mon.steps_) - onset_idx
    assert len(action.window) != len(mon.steps_)  # a label lookup would find index 0


def test_apply_recommendation_audit_fingerprints_round_trip() -> None:
    mon = _drifted_monitor(shift=0.8, n_batches=6)
    action = mon.apply_recommendation()

    js = action.to_json()
    restored = AppliedAction.from_json(js)
    assert restored.audit == action.audit
    assert restored.fingerprint() == action.fingerprint()


def test_applied_action_round_trip_with_chain_composed() -> None:
    d = make_pd_portfolio(n=400, random_state=7)
    cal = BetaCalibrator().fit(d.scores, d.y)
    chain = Chain([cal])
    mon = _drifted_monitor(shift=0.8, n_batches=6)

    action = mon.apply_recommendation(target=chain)
    restored = AppliedAction.from_json(action.to_json())
    assert isinstance(restored.composed, Chain)
    assert restored.fingerprint() == action.fingerprint()
    assert restored.composed.fingerprint() == action.composed.fingerprint()


def test_applied_action_round_trip_with_calibrated_model_composed() -> None:
    d = make_pd_portfolio(n=400, random_state=7)
    wrapped = CalibratedModel(_StubModel(), BetaCalibrator(), flow="prefit").fit(
        d.scores.reshape(-1, 1), d.y
    )
    mon = _drifted_monitor(shift=0.8, n_batches=6)
    action = mon.apply_recommendation(target=wrapped)
    js = action.to_json()

    # Lazy load: only a model *reference* was serialized (CalibratedModel.to_dict),
    # never the model itself -- composed comes back with model_ unattached.
    lazy = AppliedAction.from_json(js)
    assert isinstance(lazy.composed, CalibratedModel)
    assert lazy.composed.model_ is None

    # model= reattaches the same model class: fingerprints agree exactly,
    # and prediction is possible again.
    restored = AppliedAction.from_json(js, model=_StubModel())
    assert isinstance(restored.composed, CalibratedModel)
    assert restored.composed.model_ is not None
    assert restored.fingerprint() == action.fingerprint()
    assert restored.composed.fingerprint() == action.composed.fingerprint()
    np.testing.assert_array_equal(
        restored.composed.predict_proba(d.scores.reshape(-1, 1)),
        action.composed.predict_proba(d.scores.reshape(-1, 1)),
    )


def test_apply_recommendation_composes_chain_target() -> None:
    d = make_pd_portfolio(n=400, random_state=7)
    cal = BetaCalibrator().fit(d.scores, d.y)
    chain = Chain([cal])
    mon = _drifted_monitor(shift=0.8, n_batches=6)

    action = mon.apply_recommendation(target=chain)
    assert isinstance(action.composed, Chain)
    assert action.composed.offsets_[-1].delta_ == pytest.approx(action.offset.delta_)
    assert chain.offsets_ == ()  # the original target is untouched


def test_apply_recommendation_composes_calibrated_model_target() -> None:
    d = make_pd_portfolio(n=400, random_state=7)
    wrapped = CalibratedModel(_StubModel(), BetaCalibrator(), flow="prefit").fit(
        d.scores.reshape(-1, 1), d.y
    )
    mon = _drifted_monitor(shift=0.8, n_batches=6)

    action = mon.apply_recommendation(target=wrapped)
    assert isinstance(action.composed, CalibratedModel)
    assert len(action.composed.offsets_) == 1
    assert action.composed.offsets_[-1].delta_ == pytest.approx(action.offset.delta_)
    assert wrapped.offsets_ == []  # the original target is untouched (deep-copied)


def test_apply_recommendation_rejects_unknown_target_type() -> None:
    mon = _drifted_monitor(shift=0.8, n_batches=6)
    with pytest.raises(TypeError, match="Chain"):
        mon.apply_recommendation(target=object())


def test_apply_recommendation_closes_the_drift_loop() -> None:
    """End-to-end: alarm -> apply_recommendation() -> feed the corrected
    stream into the fresh monitor -> no further alarm (spec M4)."""
    mon = CalibrationMonitor(alpha=0.05)
    alarm_batch = None
    batches = []
    for k in range(20):
        shift = 0.5 if k >= 10 else 0.0
        y, p = _batch(n=2000, shift=shift, seed=k)
        batches.append((y, p))
        step = mon.update(y, p, label=f"m{k}")
        if step.alarm and alarm_batch is None:
            alarm_batch = k
    assert alarm_batch is not None and alarm_batch >= 10

    action = mon.apply_recommendation()
    assert action.kind == "re-offset"

    step = None
    for k in range(alarm_batch + 1, 20):
        y, p = batches[k]
        p_corrected = action.offset.transform(p)
        step = action.monitor.update(y, p_corrected, label=f"m{k}")
        assert not step.alarm
